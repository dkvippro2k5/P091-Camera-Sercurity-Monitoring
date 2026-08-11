import { useState, useEffect } from 'react'
import PolygonCanvas from './PolygonCanvas'
import { api } from '../api'
import { describeWindow, toWindowString } from '../activeHours'
import { incidentMeta } from '../incidents'

/**
 * The guard's camera screen: what exists, plus the form to add another.
 *
 * The existing-camera list here is now convenience rather than the only route to a delete —
 * the guard owns both this screen and the live grid, whose tiles carry their own delete
 * button. It is kept because this is where someone comes to *change* the setup, and being
 * shown the current roster next to the form is how you notice a camera is already configured
 * before adding it twice.
 */
export default function CameraSetup({ onAdd, onCancel, cameras = [], onDelete }) {
  const [name, setName] = useState('')
  const [sourceType, setSourceType] = useState('demo')
  const [source, setSource] = useState('')
  const [demos, setDemos] = useState([])
  const [intrusionEnabled, setIntrusionEnabled] = useState(false)
  const [polygon, setPolygon] = useState([])
  // Always-armed is the default. The PRD's example window is 22:00-05:00, but defaulting to
  // it would mean a camera added at 14:00 silently watches nothing for eight hours — the
  // failure has to be opted into, not inherited.
  const [hoursEnabled, setHoursEnabled] = useState(false)
  const [startTime, setStartTime] = useState('22:00')
  const [endTime, setEndTime] = useState('05:00')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    api
      .listDemoVideos()
      .then((videos) => {
        setDemos(videos)
        if (videos.length) setSource(videos[0].id)
      })
      .catch((err) => setError(err.message))
  }, [])

  // The polygon is drawn on a still from *this* source, so switching source invalidates
  // whatever was drawn — keeping it would place the zone against a different scene.
  useEffect(() => {
    setPolygon([])
  }, [sourceType, source])

  const canPreview = Boolean(source) && (sourceType !== 'rtsp' || source.startsWith('rtsp://'))
  const previewUrl = canPreview ? api.previewFrameUrl(sourceType, source) : null

  async function handleUpload(event) {
    const file = event.target.files?.[0]
    if (!file) return
    setBusy(true)
    setError(null)
    try {
      const { filename } = await api.uploadVideo(file)
      // The server-generated name is what the backend can resolve; the original filename
      // is never used as a path.
      setSource(filename)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  async function handleSubmit(event) {
    event.preventDefault()
    setError(null)

    if (intrusionEnabled && polygon.length < 3) {
      setError('Draw at least 3 points on the preview to define the restricted zone.')
      return
    }
    if (intrusionEnabled && hoursEnabled && startTime === endTime) {
      setError('Zone hours start and end are the same time. Pick a different end time, or switch to always armed.')
      return
    }

    setBusy(true)
    try {
      await onAdd({
        name: name.trim() || 'Camera',
        source_type: sourceType,
        source,
        intrusion_enabled: intrusionEnabled,
        zone_polygon: intrusionEnabled ? polygon : null,
        // Only meaningful alongside a zone, so it rides with the polygon rather than
        // being sent for a camera that has intrusion switched off.
        zone_active_hours: intrusionEnabled ? toWindowString(hoursEnabled, startTime, endTime) : null,
      })
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const SOURCE_LABELS = { demo: 'Prepared demo', upload: 'Upload video', rtsp: 'RTSP stream' }
  // Recomputed on render rather than memoized: it reads the current clock, and the whole
  // point is that it reflects the moment the operator is looking at it.
  const windowNote = describeWindow(startTime, endTime)

  return (
    <div className="setup">
      <CameraList cameras={cameras} onDelete={onDelete} />

      <div className="setup-head">
        <h2>Add camera</h2>
        <p>
          Fall and abandoned-object detection start automatically. Intrusion detection is optional and needs a
          zone drawn on the camera&apos;s own preview.
        </p>
      </div>

      <form onSubmit={handleSubmit}>
        <label className="field">
          <span>Camera name</span>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Lobby north" />
          <span className="hint">Shown on the tile and in every alert from this camera.</span>
        </label>

        <fieldset>
          <legend>Video source</legend>

          {/* Each option is a full-width card; the label is the target, so the hit area is
              44px rather than the ~13px native radio it wraps. */}
          <div className="choices">
            {['demo', 'upload', 'rtsp'].map((type) => (
              <label key={type} className="choice">
                <input
                  type="radio"
                  name="source_type"
                  value={type}
                  checked={sourceType === type}
                  onChange={() => {
                    setSourceType(type)
                    setSource(type === 'demo' && demos.length ? demos[0].id : '')
                  }}
                />
                {SOURCE_LABELS[type]}
              </label>
            ))}
          </div>

          {sourceType === 'demo' && (
            <label className="field">
              <span className="visually-hidden">Demo clip</span>
              <select value={source} onChange={(e) => setSource(e.target.value)}>
                {demos.map((demo) => (
                  <option key={demo.id} value={demo.id}>
                    {demo.name}
                    {demo.suggested_label ? ` — shows ${demo.suggested_label.replace('_', ' ')}` : ''}
                  </option>
                ))}
              </select>
            </label>
          )}

          {sourceType === 'upload' && (
            <div className="field">
              <input type="file" accept="video/mp4,video/x-matroska,video/quicktime" onChange={handleUpload} />
              {source && <small className="ok">Uploaded as {source}</small>}
            </div>
          )}

          {sourceType === 'rtsp' && (
            <label className="field">
              <span className="visually-hidden">RTSP URL</span>
              <input
                value={source}
                onChange={(e) => setSource(e.target.value)}
                placeholder="rtsp://user:pass@192.168.1.50:554/stream1"
              />
            </label>
          )}
        </fieldset>

        <fieldset>
          <legend>Detection</legend>
          {/*
            Fall and abandoned-object are not toggles: PRD §3.1 has them always on. Intrusion
            is opt-in because it is meaningless without a polygon to be inside of.
          */}
          <p className="always-on">
            <span className="tag">Fall detection</span>
            <span className="tag">Abandoned object</span>
            <span>always active</span>
          </p>

          <label className="toggle">
            <input
              type="checkbox"
              checked={intrusionEnabled}
              onChange={(e) => setIntrusionEnabled(e.target.checked)}
            />
            <span className="toggle-text">
              <strong>Intrusion detection</strong>
              <span className="hint">Alerts when a person stands inside a zone you draw below.</span>
            </span>
          </label>

          {intrusionEnabled &&
            (previewUrl ? (
              <>
                <PolygonCanvas imageUrl={previewUrl} points={polygon} onChange={setPolygon} />

                {/* PRD §3.1: intrusion alerts "trong khung giờ cấm (VD: 22h - 05h)". The
                    window is per-zone and evaluated against wall-clock time, so it belongs
                    next to the polygon it arms rather than in a global setting. */}
                <div className="zone-hours">
                  <span className="fact-label">Zone hours</span>

                  <div className="choices">
                    <label className="choice">
                      <input
                        type="radio"
                        name="zone_hours"
                        checked={!hoursEnabled}
                        onChange={() => setHoursEnabled(false)}
                      />
                      Always armed
                    </label>
                    <label className="choice">
                      <input
                        type="radio"
                        name="zone_hours"
                        checked={hoursEnabled}
                        onChange={() => setHoursEnabled(true)}
                      />
                      Only between set hours
                    </label>
                  </div>

                  {hoursEnabled && (
                    <>
                      <div className="time-range">
                        <label className="field compact-field">
                          <span>From</span>
                          <input type="time" value={startTime} onChange={(e) => setStartTime(e.target.value)} required />
                        </label>
                        <label className="field compact-field">
                          <span>To</span>
                          <input type="time" value={endTime} onChange={(e) => setEndTime(e.target.value)} required />
                        </label>
                      </div>

                      {/* The consequence, not just the state. An operator who sets a night
                          window at midday would otherwise leave this screen believing the
                          zone is live and conclude the feature is broken. aria-live so the
                          note is announced when the times change, not just painted. */}
                      <small className={windowNote.armed ? 'ok' : 'warn'} aria-live="polite">
                        {windowNote.text}
                      </small>
                    </>
                  )}
                </div>
              </>
            ) : (
              <small className="warn">Pick a video source first to draw the zone on its preview.</small>
            ))}
        </fieldset>

        {/* role="alert" so the failure is announced rather than silently rendered next to
            a submit button that appears to do nothing. */}
        {error && (
          <p className="err" role="alert">
            <span aria-hidden="true">▲</span>
            {error}
          </p>
        )}

        <div className="actions">
          <button type="submit" className="primary" disabled={busy || !source}>
            {busy ? 'Starting…' : 'Add camera'}
          </button>
          <button type="button" onClick={onCancel} disabled={busy}>
            Cancel
          </button>
          {/* A disabled button with no stated reason reads as a broken button. */}
          {!source && !busy && <span className="hint">Choose a video source to continue.</span>}
        </div>
      </form>
    </div>
  )
}

/**
 * Configured cameras, with a delete alongside the one on the grid tile.
 *
 * Deliberately not a second live grid — no MJPEG here. Rendering every stream on the
 * configuration screen would double each camera's decode cost for someone who came to check
 * settings, and the setup screen's job is what is *configured*, not what is on screen right
 * now — that is one nav click away on the grid.
 */
function CameraList({ cameras, onDelete }) {
  if (!cameras.length) return null

  return (
    <section className="camera-list">
      <div className="setup-head">
        <h2>Configured cameras</h2>
        <p>
          {cameras.length} camera{cameras.length === 1 ? '' : 's'}, {cameras.filter((c) => c.running).length} currently
          streaming. These are the feeds on your live grid.
        </p>
      </div>

      <ul>
        {cameras.map((camera) => (
          <li key={camera.id} className="camera-row">
            <span className={camera.running ? 'status-chip live' : 'status-chip offline'}>
              <span className="dot" aria-hidden="true" />
              {camera.running ? 'Live' : 'Offline'}
            </span>

            <span className="camera-row-name" title={camera.name}>
              {camera.name}
            </span>

            <span className="camera-row-tags">
              {camera.active_labels.map((label) => (
                <span key={label} className="tag">
                  {incidentMeta(label).short}
                </span>
              ))}
              {/* The window is shown here even though the grid also shows it: this is where
                  it was configured, so it is where you come to check what you set. */}
              {camera.zone_active_hours && <span className="tag">{camera.zone_active_hours}</span>}
            </span>

            <span className="telemetry">{camera.source_type}</span>

            <button
              className="icon compact danger"
              onClick={() => onDelete(camera.id)}
              aria-label={`Remove camera ${camera.name}`}
              title={`Remove ${camera.name}`}
            >
              <span aria-hidden="true">✕</span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  )
}
