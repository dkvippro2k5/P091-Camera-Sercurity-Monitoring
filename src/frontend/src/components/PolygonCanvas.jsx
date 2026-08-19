import { useRef, useEffect, useState } from 'react'

/**
 * Draw a restricted-area polygon on top of a still frame from the camera.
 *
 * Points are stored **normalized to [0, 1]**, never in pixels. Two reasons, both hard
 * requirements rather than preferences: the backend's `Zone` model rejects anything outside
 * that range outright, and a normalized polygon survives the camera changing resolution —
 * the same config works on a 720p and a 1080p stream, because pixel conversion happens at
 * query time against whatever the frame actually is.
 *
 * Normalization divides by the element's *rendered* size from getBoundingClientRect(), not
 * by the image's natural size. The canvas is CSS-scaled to fit the layout, so the natural
 * size is not what the click landed on.
 */
export default function PolygonCanvas({ imageUrl, points, onChange, onError }) {
  const canvasRef = useRef(null)
  const [size, setSize] = useState({ width: 0, height: 0 })

  // Keep the drawing surface matched to the displayed box, including on window resize —
  // a canvas whose backing store differs from its CSS size renders blurry and, worse,
  // offsets the drawn polygon from where the user clicked.
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const measure = () => {
      const rect = canvas.getBoundingClientRect()
      if (rect.width && rect.height) setSize({ width: rect.width, height: rect.height })
    }
    measure()

    const observer = new ResizeObserver(measure)
    observer.observe(canvas)
    return () => observer.disconnect()
  }, [imageUrl])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !size.width) return

    canvas.width = size.width
    canvas.height = size.height
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, size.width, size.height)
    if (points.length === 0) return

    const pixels = points.map(([x, y]) => [x * size.width, y * size.height])

    ctx.beginPath()
    ctx.moveTo(pixels[0][0], pixels[0][1])
    pixels.slice(1).forEach(([x, y]) => ctx.lineTo(x, y))
    // Close the outline only once it is a real polygon, so 2 points still read as a line
    // being drawn rather than a degenerate shape.
    if (pixels.length > 2) ctx.closePath()

    ctx.strokeStyle = '#f87171'
    ctx.lineWidth = 2.5
    ctx.stroke()

    if (pixels.length > 2) {
      ctx.fillStyle = 'rgba(220, 38, 38, 0.24)'
      ctx.fill()
    }

    // 7px radius with a dark halo. At the previous 5px flat dot, a vertex dropped on a
    // bright region of the frame was indistinguishable from the scene behind it, and the
    // first-point marker that closes the polygon was the one most often lost.
    pixels.forEach(([x, y], index) => {
      ctx.beginPath()
      ctx.arc(x, y, 7, 0, Math.PI * 2)
      ctx.fillStyle = index === 0 ? '#ffffff' : '#dc2626'
      ctx.fill()
      ctx.lineWidth = 2
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.65)'
      ctx.stroke()
    })
  }, [points, size])

  function handleClick(event) {
    const rect = canvasRef.current.getBoundingClientRect()
    const x = (event.clientX - rect.left) / rect.width
    const y = (event.clientY - rect.top) / rect.height
    // Clamp rather than reject: a click a pixel outside the box would otherwise fail the
    // backend's [0,1] validator with an error the user cannot act on.
    const clamp = (v) => Math.min(1, Math.max(0, Number(v.toFixed(4))))
    onChange([...points, [clamp(x), clamp(y)]])
  }

  return (
    <div className="polygon">
      <div className="polygon-stage">
        <img src={imageUrl} alt="Camera preview for zone drawing" onError={onError} />
        <canvas ref={canvasRef} onClick={handleClick} />
      </div>
      <div className="polygon-controls">
        {/* Point count and the 3-minimum requirement are stated up front. Previously the
            minimum only appeared after the first click, so the requirement surfaced as a
            submit-time error instead of a precondition. */}
        <span>
          {points.length === 0
            ? 'Click the preview to place zone points (3 minimum).'
            : `${points.length} of 3 minimum placed.`}
        </span>
        <span className="spacer" />
        <button type="button" className="compact" onClick={() => onChange(points.slice(0, -1))} disabled={!points.length}>
          Undo
        </button>
        <button type="button" className="compact" onClick={() => onChange([])} disabled={!points.length}>
          Clear
        </button>
      </div>
    </div>
  )
}
