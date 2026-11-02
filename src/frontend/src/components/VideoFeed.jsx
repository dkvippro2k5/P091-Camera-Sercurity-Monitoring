import { useEffect, useRef, useState } from 'react'
import { api } from '../api'

/**
 * One camera's live picture: H.264 over a WebSocket, with the old MJPEG `<img>` as fallback.
 *
 * Why not just keep the `<img>`: MJPEG sends every frame as a standalone JPEG (~10 Mbps for
 * one 480p camera on our footage) and the browser decodes each one on the CPU. That is what
 * held the grid to 10-14fps while the backend was producing 20. H.264 sends keyframes rarely
 * and deltas in between — ~8x less data — and the browser decodes it on the GPU, so the
 * same feed arrives at the rate the server sends it.
 *
 * The fallback is not decoration. It carries three real cases: a browser with no MediaSource
 * support, a host whose FFmpeg has no usable H.264 encoder (the server says so explicitly
 * with a `fallback` message rather than leaving us on a silent socket), and a socket that
 * drops. In all three the tile keeps showing video instead of going black.
 *
 * MSE bookkeeping worth knowing about, since both bugs are silent when they happen:
 *
 *   - `appendBuffer` throws if the previous append has not finished, so fragments queue here
 *     and drain on `updateend` rather than being appended as they arrive.
 *   - A live SourceBuffer grows forever. Old ranges are evicted once past a threshold, or
 *     the tab's memory climbs until the browser kills the buffer out from under us.
 */

// Kept small on purpose: this is a live view, so falling behind is worse than dropping. If
// the queue exceeds this the oldest fragments go — the newest picture is the one that matters.
const MAX_QUEUED_FRAGMENTS = 90
// Seconds of already-played video to keep buffered before evicting. Enough to survive a
// brief stall, small enough that memory stays flat over a long shift.
const BUFFER_RETENTION_SEC = 12
// How far behind live the element may drift before it is nudged forward. MSE plays from
// wherever it is, so a stall silently becomes permanent latency without this.
const MAX_DRIFT_SEC = 2.5
// Backoff before retrying the socket after a clean end (camera restarted, server redeployed).
const RECONNECT_DELAY_MS = 2000

export default function VideoFeed({ camera, className, alt }) {
  const videoRef = useRef(null)
  // 'connecting' until the first fragment lands, then 'live'; 'fallback' switches to MJPEG
  // for the rest of this camera's mount.
  const [mode, setMode] = useState(() =>
    typeof window !== 'undefined' && 'MediaSource' in window ? 'connecting' : 'fallback',
  )

  const cameraId = camera.id

  useEffect(() => {
    if (mode === 'fallback') return undefined

    let disposed = false
    let socket = null
    let mediaSource = null
    let sourceBuffer = null
    let reconnectTimer = null
    // Fragments waiting for the SourceBuffer to become idle. See the note above about
    // appendBuffer throwing on a busy buffer.
    let pending = []
    let sawInitSegment = false
    let announcedLive = false

    const video = videoRef.current
    if (!video) return undefined

    /** Give up on H.264 for this camera and let the MJPEG `<img>` take over. */
    const giveUp = () => {
      if (disposed) return
      setMode('fallback')
    }

    const drain = () => {
      if (!sourceBuffer || sourceBuffer.updating || pending.length === 0) return
      if (!mediaSource || mediaSource.readyState !== 'open') return
      try {
        sourceBuffer.appendBuffer(pending.shift())
      } catch (err) {
        // QuotaExceededError is the recoverable one: the buffer is full, so drop what is
        // buffered and resync. Anything else means this SourceBuffer is unusable.
        if (err.name === 'QuotaExceededError') {
          pending = []
          evictOldRanges(true)
        } else {
          giveUp()
        }
      }
    }

    /**
     * Drop buffered ranges the viewer has already played past.
     *
     * `aggressive` is used after a quota error, where keeping the usual retention window is
     * exactly what filled the buffer in the first place.
     */
    const evictOldRanges = (aggressive = false) => {
      if (!sourceBuffer || sourceBuffer.updating) return
      const buffered = sourceBuffer.buffered
      if (buffered.length === 0) return
      const start = buffered.start(0)
      const keepFrom = video.currentTime - (aggressive ? 1 : BUFFER_RETENTION_SEC)
      if (keepFrom <= start) return
      try {
        sourceBuffer.remove(start, keepFrom)
      } catch {
        // A remove that races the buffer going busy is retried on the next updateend; it is
        // not worth tearing the stream down over.
      }
    }

    /** Skip forward when playback has fallen behind the newest buffered frame. */
    const keepUpWithLive = () => {
      if (!sourceBuffer || sourceBuffer.buffered.length === 0) return
      const end = sourceBuffer.buffered.end(sourceBuffer.buffered.length - 1)
      if (end - video.currentTime > MAX_DRIFT_SEC) {
        video.currentTime = end - 0.1
      }
      if (video.paused) {
        // Autoplay can still be refused despite `muted`; the catch keeps that from becoming
        // an unhandled rejection on every tick.
        video.play().catch(() => {})
      }
    }

    const onUpdateEnd = () => {
      evictOldRanges()
      keepUpWithLive()
      drain()
    }

    const connect = () => {
      if (disposed) return

      mediaSource = new MediaSource()
      video.src = URL.createObjectURL(mediaSource)

      mediaSource.addEventListener('sourceopen', () => {
        if (disposed) return
        // Revoked as soon as the MediaSource is attached: the element holds its own
        // reference, and leaving the object URL alive leaks it for the tab's lifetime.
        URL.revokeObjectURL(video.src)

        socket = new WebSocket(api.videoUrl(cameraId))
        socket.binaryType = 'arraybuffer'

        socket.onmessage = (event) => {
          if (disposed) return

          if (typeof event.data === 'string') {
            // Control frames. `init` carries the codec string MSE needs; `fallback` is the
            // server telling us this camera cannot serve H.264 at all.
            let message
            try {
              message = JSON.parse(event.data)
            } catch {
              return
            }
            if (message.type === 'fallback') {
              giveUp()
              return
            }
            if (message.type === 'init') {
              if (!MediaSource.isTypeSupported(message.mime)) {
                giveUp()
                return
              }
              try {
                sourceBuffer = mediaSource.addSourceBuffer(message.mime)
                // 'sequence' would renumber timestamps and defeat the drift check above;
                // the server's pts are already a clean monotonic timeline.
                sourceBuffer.mode = 'segments'
                sourceBuffer.addEventListener('updateend', onUpdateEnd)
              } catch {
                giveUp()
              }
            }
            return
          }

          // Binary: the first is the init segment, the rest are media fragments. The server
          // guarantees that order, so there is no per-message header to parse.
          if (!sourceBuffer) return
          if (!sawInitSegment) {
            sawInitSegment = true
          } else if (!announcedLive) {
            // A local flag, not a read of `mode`: this closure is created once per connect
            // and would capture a stale `mode` forever. React de-dupes the set anyway; the
            // flag just avoids calling it on every fragment.
            announcedLive = true
            setMode('live')
          }

          pending.push(event.data)
          if (pending.length > MAX_QUEUED_FRAGMENTS) {
            pending = pending.slice(-MAX_QUEUED_FRAGMENTS)
          }
          drain()
        }

        socket.onerror = () => {
          // Errors always precede a close; handled there so the retry lives in one place.
        }

        socket.onclose = () => {
          if (disposed) return
          // Never showed a frame: the camera or the encoder is the problem, not the link,
          // so retrying would just loop. Show MJPEG instead.
          if (!sawInitSegment) {
            giveUp()
            return
          }
          // Played and then dropped — a redeploy or a camera restart. Rebuild the whole
          // MediaSource rather than reusing it: its SourceBuffer holds a timeline the new
          // stream will not continue.
          teardownStream()
          reconnectTimer = setTimeout(connect, RECONNECT_DELAY_MS)
        }
      })
    }

    const teardownStream = () => {
      if (socket) {
        socket.onmessage = socket.onclose = socket.onerror = null
        if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
          socket.close()
        }
        socket = null
      }
      if (sourceBuffer) {
        sourceBuffer.removeEventListener('updateend', onUpdateEnd)
        sourceBuffer = null
      }
      if (mediaSource && mediaSource.readyState === 'open') {
        try {
          mediaSource.endOfStream()
        } catch {
          // Already ending, or a buffer is mid-update. Nothing depends on this succeeding.
        }
      }
      mediaSource = null
      pending = []
      sawInitSegment = false
    }

    connect()

    return () => {
      disposed = true
      if (reconnectTimer) clearTimeout(reconnectTimer)
      teardownStream()
      if (video) video.removeAttribute('src')
    }
    // `mode` is deliberately not a dependency: it is set from inside this effect, and
    // re-running on every 'connecting' -> 'live' transition would tear down the very stream
    // that just went live. The one transition that must rebuild — into 'fallback' — is
    // handled by the early return, which the guard below re-evaluates on a camera change.
  }, [cameraId, mode === 'fallback'])

  if (mode === 'fallback') {
    // The original MJPEG path, unchanged: the browser decodes multipart/x-mixed-replace
    // natively, and a stalled camera holds its last frame rather than throwing.
    return <img className={className} src={camera.stream_url ?? api.streamUrl(cameraId)} alt={alt} />
  }

  return (
    <video
      ref={videoRef}
      className={className}
      // muted is what makes autoplay permissible without a user gesture; playsInline keeps
      // iOS Safari from taking the tile fullscreen on play.
      muted
      autoPlay
      playsInline
      aria-label={alt}
    />
  )
}
