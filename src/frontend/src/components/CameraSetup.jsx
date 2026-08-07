import { useState, useEffect } from 'react'
import PolygonCanvas from './PolygonCanvas'
import { api } from '../api'

export default function CameraSetup({ onAdd, onCancel }) {
  const [name, setName] = useState('')
  const [sourceType, setSourceType] = useState('demo')
  const [source, setSource] = useState('')
  const [demos, setDemos] = useState([])
  const [intrusionEnabled, setIntrusionEnabled] = useState(false)
  const [polygon, setPolygon] = useState([])
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

    setBusy(true)
    try {
      await onAdd({
        name: name.trim() || 'Camera',
        source_type: sourceType,
        source,
        intrusion_enabled: intrusionEnabled,
        zone_polygon: intrusionEnabled ? polygon : null,
      })
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const SOURCE_LABELS = { demo: 'Prepared demo', upload: 'Upload video', rtsp: 'RTSP stream' }

  return (
    <div className="setup">
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
              <PolygonCanvas imageUrl={previewUrl} points={polygon} onChange={setPolygon} />
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
