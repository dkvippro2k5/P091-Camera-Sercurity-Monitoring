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
  const [imageError, setImageError] = useState(false)

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
    setImageError(false)
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
      <header className="setup-page-head">
        <div>
          <p className="eyebrow">Camera configuration</p>
          <h1>Manage cameras</h1>
          <p>Sources, detection rules, and monitoring zones.</p>
        </div>
        <button type="button" onClick={onCancel}>Return to live grid</button>
      </header>

      <CameraList cameras={cameras} onDelete={onDelete} />

      <form onSubmit={handleSubmit} className="setup-split">
        {/* Left Column: Form Fields */}
        <div className="setup-col-form">
          <div className="setup-form-head">
            <span className="setup-step">01</span>
            <div><h2>Add a camera</h2><p>Choose a source and detection policy.</p></div>
          </div>
          <label className="field">
            <span>Camera name</span>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Lobby north" />
            <span className="hint">Shown on the live tile and in every alert from this camera.</span>
          </label>

          <fieldset>
            <legend>Video source</legend>

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
            <legend>Detection rules</legend>
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
                <span className="hint">Alerts when a person stands inside a zone you draw on the preview.</span>
              </span>
            </label>

            {intrusionEnabled && (
              <div className="zone-hours">
                <span className="fact-label">Active monitoring hours</span>

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
                    Set hours only
                  </label>
                </div>

                {hoursEnabled && (
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
                )}

                {/* Real-time Status Feedback Card */}
                <div className="active-hours-status-card" aria-live="polite">
                  <div className="status-header">
                    <span className="status-label">Monitoring status</span>
                    {!hoursEnabled ? (
                      <span className="status-badge armed">
                        <span className="pulse-dot green" aria-hidden="true" />
                        Armed
                      </span>
                    ) : windowNote.armed ? (
                      <span className="status-badge armed">
                        <span className="pulse-dot green" aria-hidden="true" />
                        Armed
                      </span>
                    ) : (
                      <span className="status-badge disarmed">
                        <span className="badge-dot amber" aria-hidden="true" />
                        Disarmed
                      </span>
                    )}
                  </div>
                  <small className={!hoursEnabled || windowNote.armed ? 'ok' : 'warn'}>
                    {!hoursEnabled ? 'Always armed — 24/7 active monitoring enabled.' : windowNote.text}
                  </small>
                </div>
              </div>
            )}
          </fieldset>

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
            {!source && !busy && <span className="hint">Choose a video source to continue.</span>}
          </div>
        </div>

        {/* Right Column: Visual Live Canvas Sticky Card */}
        <div className="setup-col-preview">
          <div className="setup-preview-card">
            <div className="setup-preview-head">
              <div><span className="setup-step">02</span><h2>Preview and zone</h2></div>
              {previewUrl && intrusionEnabled && (
                <span className={polygon.length < 3 ? 'zone-points-badge warning' : 'zone-points-badge active'}>
                  <span className={polygon.length < 3 ? 'badge-dot amber' : 'badge-dot green'} aria-hidden="true" />
                  {polygon.length < 3
                    ? `${polygon.length}/3 points drawn`
                    : `${polygon.length} points active`}
                </span>
              )}
              {previewUrl && !intrusionEnabled && (
                <span className="status-badge disarmed">
                  <span className="badge-dot amber" aria-hidden="true" />
                  Zone off
                </span>
              )}
            </div>

            {previewUrl && !imageError ? (
              <div className="preview-container">
                {intrusionEnabled ? (
                  <div className="preview-canvas-wrap">
                    <PolygonCanvas imageUrl={previewUrl} points={polygon} onChange={setPolygon} onError={() => setImageError(true)} />
                  </div>
                ) : (
                  <div className="preview-image-wrap">
                    <div className="polygon-stage">
                      <img src={previewUrl} alt="Camera preview frame" onError={() => setImageError(true)} />
                    </div>
                  </div>
                )}
              </div>
            ) : imageError ? (
              <div className="preview-placeholder">
                <span className="preview-placeholder-icon error" aria-hidden="true">!</span>
                <h4>RTSP connection failed</h4>
                <p>Check the stream URL, credentials, and camera network connection.</p>
              </div>
            ) : (
              <div className="preview-placeholder">
                <span className="preview-placeholder-icon" aria-hidden="true">+</span>
                <h4>No source selected</h4>
                <p>The preview will appear when a video source is ready.</p>
              </div>
            )}
          </div>
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
