import { useState, useEffect, useRef } from 'react'
import PendingSourceTile from './PendingSourceTile'
import { api } from '../api'
import { toWindowString } from '../activeHours'
import { incidentMeta } from '../incidents'

const SOURCE_LABELS = { demo: 'Video mẫu', upload: 'Tải video lên', rtsp: 'Luồng RTSP' }

function deriveRtspName(url) {
  try {
    return new URL(url).hostname || 'Luồng RTSP'
  } catch {
    return 'Luồng RTSP'
  }
}

/**
 * The guard's camera screen: what exists, plus the form to add more.
 *
 * Adding is queue-then-submit rather than one-shot: the operator builds up a list of
 * sources — any mix of RTSP streams, sample clips, and uploaded files — each with its own
 * name, zone and schedule, then adds them all at once. This replaced two separate paths (a
 * single-source form and a no-zone "batch upload for testing" section) that did the same
 * job through different doors; the batch path in particular could never draw a zone; folding
 * multi-upload into the same queue removes that gap instead of caging it off in a toggle.
 *
 * The existing-camera list here is now convenience rather than the only route to a delete —
 * the guard owns both this screen and the live grid, whose tiles carry their own delete
 * button. It is kept because this is where someone comes to *change* the setup, and being
 * shown the current roster next to the form is how you notice a camera is already configured
 * before adding it twice.
 */
export default function CameraSetup({ onAddBatch, onCancel, cameras = [], onDelete }) {
  const [pendingSources, setPendingSources] = useState([])
  const [sourceType, setSourceType] = useState('rtsp')
  const [rtspInput, setRtspInput] = useState('')
  const [demoSelection, setDemoSelection] = useState('')
  const [demos, setDemos] = useState([])
  const [uploadBusy, setUploadBusy] = useState(false)
  const [uploadStatus, setUploadStatus] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const nextKeyRef = useRef(0)

  useEffect(() => {
    api
      .listDemoVideos()
      .then((videos) => {
        setDemos(videos)
        if (videos.length) setDemoSelection(videos[0].id)
      })
      .catch((err) => setError(err.message))
  }, [])

  function addEntry({ source_type, source, name }) {
    setPendingSources((prev) => [
      ...prev,
      {
        key: `src-${nextKeyRef.current++}`,
        name,
        source_type,
        source,
        intrusion_enabled: false,
        polygon: [],
        // Always-armed is the default. The PRD's example window is 22:00-05:00, but
        // defaulting to it would mean a camera added at 14:00 silently watches nothing
        // for eight hours — the failure has to be opted into, not inherited.
        hours_enabled: false,
        start_time: '22:00',
        end_time: '05:00',
        image_error: false,
      },
    ])
  }

  function addRtspEntry() {
    const url = rtspInput.trim()
    if (!url.startsWith('rtsp://')) {
      setError('Nhập URL RTSP hợp lệ, bắt đầu bằng rtsp://')
      return
    }
    setError(null)
    addEntry({ source_type: 'rtsp', source: url, name: deriveRtspName(url) })
    setRtspInput('')
  }

  function addDemoEntry() {
    if (!demoSelection) return
    const demo = demos.find((d) => d.id === demoSelection)
    addEntry({ source_type: 'demo', source: demoSelection, name: demo?.name ?? 'Video mẫu' })
  }

  // Every selected file becomes its own queued camera. Uploads run sequentially (one big
  // multi-part POST per file is simpler than a fan-out, and this is an interactive add
  // flow, not a path that needs to race itself) so `uploadStatus` can report which file is
  // in flight.
  async function handleUploadFiles(event) {
    const files = Array.from(event.target.files || [])
    if (!files.length) return

    setUploadBusy(true)
    setError(null)
    const failures = []

    for (let i = 0; i < files.length; i++) {
      const file = files[i]
      setUploadStatus(`Đang tải ${i + 1}/${files.length}: ${file.name}`)
      try {
        const { filename } = await api.uploadVideo(file)
        addEntry({
          source_type: 'upload',
          source: filename,
          name: file.name.replace(/\.[^/.]+$/, '').slice(0, 64) || `Camera ${i + 1}`,
        })
      } catch (err) {
        failures.push(`${file.name}: ${err.message}`)
      }
    }

    setUploadStatus(null)
    setUploadBusy(false)
    event.target.value = ''
    if (failures.length) setError(failures.join('; '))
  }

  function updateEntry(key, patch) {
    setPendingSources((prev) => prev.map((entry) => (entry.key === key ? { ...entry, ...patch } : entry)))
  }

  function removeEntry(key) {
    setPendingSources((prev) => prev.filter((entry) => entry.key !== key))
  }

  async function handleSubmit(event) {
    event.preventDefault()
    setError(null)

    const invalid = pendingSources.filter((entry) => {
      if (!entry.intrusion_enabled) return false
      if (entry.polygon.length < 3) return true
      if (entry.hours_enabled && entry.start_time === entry.end_time) return true
      return false
    })
    if (invalid.length) {
      setError(
        `Hoàn tất vùng giám sát cho: ${invalid.map((e) => e.name || 'camera chưa đặt tên').join(', ')}. ` +
          'Cần ít nhất 3 điểm trên ảnh xem trước, và giờ bắt đầu/kết thúc phải khác nhau.'
      )
      return
    }

    setBusy(true)
    try {
      await onAddBatch(
        pendingSources.map((entry) => ({
          name: entry.name.trim() || 'Camera mới',
          source_type: entry.source_type,
          source: entry.source,
          intrusion_enabled: entry.intrusion_enabled,
          zone_polygon: entry.intrusion_enabled ? entry.polygon : null,
          zone_active_hours: entry.intrusion_enabled
            ? toWindowString(entry.hours_enabled, entry.start_time, entry.end_time)
            : null,
        }))
      )
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="setup">
      <header className="setup-page-head">
        <div>
          <p className="eyebrow">Cấu hình camera</p>
          <h1>Quản lý camera</h1>
          <p>Nguồn phát, quy tắc phát hiện và vùng giám sát.</p>
        </div>
        <button type="button" onClick={onCancel}>Quay lại màn hình giám sát</button>
      </header>

      <CameraList cameras={cameras} onDelete={onDelete} />

      <form onSubmit={handleSubmit} className="setup-split">
        {/* Left Column: add-source controls */}
        <div className="setup-col-form">
          <div className="setup-form-head">
            <div><h2>Thêm camera</h2><p>Thêm một hoặc nhiều nguồn video, rồi cấu hình từng nguồn ở bên phải.</p></div>
          </div>

          <fieldset>
            <legend>Nguồn video</legend>

            <div className="choices source-choices">
              {['rtsp', 'demo', 'upload'].map((type) => (
                <label key={type} className={`choice ${type === 'rtsp' ? 'choice-primary' : 'choice-secondary'}`}>
                  <input
                    type="radio"
                    name="source_type"
                    value={type}
                    checked={sourceType === type}
                    onChange={() => setSourceType(type)}
                  />
                  <span className="choice-copy">
                    <strong>{SOURCE_LABELS[type]}</strong>
                    <small>
                      {type === 'rtsp' && 'Kết nối camera an ninh trực tiếp'}
                      {type === 'demo' && 'Dùng video mẫu để thử nghiệm'}
                      {type === 'upload' && 'Tải lên một hoặc nhiều tệp video'}
                    </small>
                  </span>
                </label>
              ))}
            </div>

            {sourceType === 'demo' && (
              <div className="field">
                <span className="visually-hidden">Video mẫu</span>
                <div className="inline-add">
                  <select value={demoSelection} onChange={(e) => setDemoSelection(e.target.value)}>
                    {demos.map((demo) => (
                      <option key={demo.id} value={demo.id}>
                        {demo.name}
                        {demo.suggested_label ? ` — minh họa ${incidentMeta(demo.suggested_label).short.toLowerCase()}` : ''}
                      </option>
                    ))}
                  </select>
                  <button type="button" className="compact" onClick={addDemoEntry} disabled={!demoSelection}>
                    Thêm vào danh sách
                  </button>
                </div>
              </div>
            )}

            {sourceType === 'upload' && (
              <div className="field">
                <input
                  type="file"
                  multiple
                  accept="video/mp4,video/x-matroska,video/quicktime"
                  onChange={handleUploadFiles}
                  disabled={uploadBusy}
                />
                {uploadBusy && uploadStatus && <small className="hint">{uploadStatus}</small>}
              </div>
            )}

            {sourceType === 'rtsp' && (
              <div className="field">
                <span className="visually-hidden">RTSP URL</span>
                <div className="inline-add">
                  <input
                    value={rtspInput}
                    onChange={(e) => setRtspInput(e.target.value)}
                    placeholder="rtsp://user:pass@192.168.1.50:554/stream1"
                  />
                  <button type="button" className="compact" onClick={addRtspEntry} disabled={!rtspInput.trim()}>
                    Thêm vào danh sách
                  </button>
                </div>
              </div>
            )}
          </fieldset>

          <p className="always-on">
            <span className="tag">Phát hiện té ngã</span>
            <span className="tag">Vật bị bỏ lại</span>
            <span>luôn hoạt động cho mọi camera</span>
          </p>

          {error && (
            <p className="err" role="alert">
              <span aria-hidden="true">▲</span>
              {error}
            </p>
          )}

          <div className="actions">
            <button type="submit" className="primary" disabled={busy || uploadBusy || pendingSources.length === 0}>
              {busy ? 'Đang khởi động…' : pendingSources.length ? `Thêm ${pendingSources.length} camera` : 'Thêm camera'}
            </button>
            {uploadBusy && <span className="hint">Đang tải video lên…</span>}
            {!uploadBusy && pendingSources.length === 0 && !busy && (
              <span className="hint">Thêm ít nhất một nguồn video để tiếp tục.</span>
            )}
          </div>
        </div>

        {/* Right Column: preview grid, one tile per queued source */}
        <div className={pendingSources.length > 1 ? 'setup-col-preview multi' : 'setup-col-preview'}>
          <div className="setup-preview-card">
            <div className="setup-preview-head">
              <div><span className="setup-step">02</span><h2>Xem trước và vùng giám sát</h2></div>
              {pendingSources.length > 0 && (
                <span className="zone-points-badge active">
                  <span className="badge-dot green" aria-hidden="true" />
                  {pendingSources.length} nguồn đã thêm
                </span>
              )}
            </div>

            {pendingSources.length === 0 ? (
              <div className="preview-placeholder">
                <span className="preview-placeholder-icon" aria-hidden="true">+</span>
                <h4>Chưa có nguồn nào</h4>
                <p>Thêm luồng RTSP, video mẫu hoặc tải video lên ở bên trái để xem trước và vẽ vùng giám sát.</p>
              </div>
            ) : (
              <div className="pending-grid">
                {pendingSources.map((entry) => (
                  <PendingSourceTile
                    key={entry.key}
                    entry={entry}
                    onChange={updateEntry}
                    onRemove={removeEntry}
                    disabled={busy}
                  />
                ))}
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
        <h2>Camera đã cấu hình</h2>
        <p>
          Có {cameras.length} camera, {cameras.filter((c) => c.running).length} camera đang phát trực tiếp trên màn hình giám sát.
        </p>
      </div>

      <ul>
        {cameras.map((camera) => (
          <li key={camera.id} className="camera-row">
            <span className={camera.running ? 'status-chip live' : 'status-chip offline'}>
              <span className="dot" aria-hidden="true" />
              {camera.running ? 'Trực tiếp' : 'Ngoại tuyến'}
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
              aria-label={`Xóa camera ${camera.name}`}
              title={`Xóa ${camera.name}`}
            >
              <span aria-hidden="true">✕</span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  )
}
