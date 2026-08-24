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
  const [sourceType, setSourceType] = useState('rtsp')
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
      setError('Hãy vẽ ít nhất 3 điểm trên ảnh xem trước để xác định khu vực hạn chế.')
      return
    }
    if (intrusionEnabled && hoursEnabled && startTime === endTime) {
      setError('Giờ bắt đầu và kết thúc trùng nhau. Hãy chọn giờ khác hoặc bật giám sát 24/7.')
      return
    }

    setBusy(true)
    try {
      await onAdd({
        name: name.trim() || 'Camera mới',
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

  const SOURCE_LABELS = { demo: 'Video mẫu', upload: 'Tải video lên', rtsp: 'Luồng RTSP' }
  // Recomputed on render rather than memoized: it reads the current clock, and the whole
  // point is that it reflects the moment the operator is looking at it.
  const windowNote = describeWindow(startTime, endTime)

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
        {/* Left Column: Form Fields */}
        <div className="setup-col-form">
          <div className="setup-form-head">
            <div><h2>Thêm camera</h2><p>Chọn nguồn phát và chính sách phát hiện.</p></div>
          </div>
          <label className="field">
            <span>Tên camera</span>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Sảnh phía Bắc" />
            <span className="hint">Hiển thị trên thẻ trực tiếp và mọi cảnh báo của camera này.</span>
          </label>

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
                    onChange={() => {
                      setSourceType(type)
                      setSource(type === 'demo' && demos.length ? demos[0].id : '')
                    }}
                  />
                  <span className="choice-copy">
                    <strong>{SOURCE_LABELS[type]}</strong>
                    <small>
                      {type === 'rtsp' && 'Kết nối camera an ninh trực tiếp'}
                      {type === 'demo' && 'Dùng video mẫu để thử nghiệm'}
                      {type === 'upload' && 'Phân tích một tệp video một lần'}
                    </small>
                  </span>
                </label>
              ))}
            </div>

            {sourceType === 'demo' && (
              <label className="field">
                <span className="visually-hidden">Video mẫu</span>
                <select value={source} onChange={(e) => setSource(e.target.value)}>
                  {demos.map((demo) => (
                    <option key={demo.id} value={demo.id}>
                      {demo.name}
                      {demo.suggested_label ? ` — minh họa ${incidentMeta(demo.suggested_label).short.toLowerCase()}` : ''}
                    </option>
                  ))}
                </select>
              </label>
            )}

            {sourceType === 'upload' && (
              <div className="field">
                <input type="file" accept="video/mp4,video/x-matroska,video/quicktime" onChange={handleUpload} />
                {source && <small className="ok">Đã tải lên với tên {source}</small>}
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
            <legend>Quy tắc phát hiện</legend>
            <p className="always-on">
              <span className="tag">Phát hiện té ngã</span>
              <span className="tag">Vật bị bỏ lại</span>
              <span>luôn hoạt động</span>
            </p>

            <label className="toggle">
              <input
                type="checkbox"
                checked={intrusionEnabled}
                onChange={(e) => setIntrusionEnabled(e.target.checked)}
              />
              <span className="toggle-text">
                <strong>Phát hiện xâm nhập</strong>
                <span className="hint">Cảnh báo khi có người đi vào vùng được vẽ trên ảnh xem trước.</span>
              </span>
            </label>

            {intrusionEnabled && (
              <div className="zone-hours">
                <span className="fact-label">Khung giờ giám sát</span>

                <div className="choices">
                  <label className="choice">
                    <input
                      type="radio"
                      name="zone_hours"
                      checked={!hoursEnabled}
                      onChange={() => setHoursEnabled(false)}
                    />
                    Giám sát 24/7
                  </label>
                  <label className="choice">
                    <input
                      type="radio"
                      name="zone_hours"
                      checked={hoursEnabled}
                      onChange={() => setHoursEnabled(true)}
                    />
                    Chỉ trong khung giờ
                  </label>
                </div>

                {hoursEnabled && (
                  <div className="time-range">
                    <label className="field compact-field">
                      <span>Từ</span>
                      <input type="time" value={startTime} onChange={(e) => setStartTime(e.target.value)} required />
                    </label>
                    <label className="field compact-field">
                      <span>Đến</span>
                      <input type="time" value={endTime} onChange={(e) => setEndTime(e.target.value)} required />
                    </label>
                  </div>
                )}

                {/* Real-time Status Feedback Card */}
                <div className="active-hours-status-card" aria-live="polite">
                  <div className="status-header">
                    <span className="status-label">Trạng thái giám sát</span>
                    {!hoursEnabled ? (
                      <span className="status-badge armed">
                        <span className="pulse-dot green" aria-hidden="true" />
                        Đang bật
                      </span>
                    ) : windowNote.armed ? (
                      <span className="status-badge armed">
                        <span className="pulse-dot green" aria-hidden="true" />
                        Đang bật
                      </span>
                    ) : (
                      <span className="status-badge disarmed">
                        <span className="badge-dot amber" aria-hidden="true" />
                        Đang tắt
                      </span>
                    )}
                  </div>
                  <small className={!hoursEnabled || windowNote.armed ? 'ok' : 'warn'}>
                    {!hoursEnabled ? 'Luôn bật — giám sát liên tục 24/7.' : windowNote.text}
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
              {busy ? 'Đang khởi động…' : 'Thêm camera'}
            </button>
            {!source && !busy && <span className="hint">Chọn nguồn video để tiếp tục.</span>}
          </div>
        </div>

        {/* Right Column: Visual Live Canvas Sticky Card */}
        <div className="setup-col-preview">
          <div className="setup-preview-card">
            <div className="setup-preview-head">
              <div><span className="setup-step">02</span><h2>Xem trước và vùng giám sát</h2></div>
              {previewUrl && intrusionEnabled && (
                <span className={polygon.length < 3 ? 'zone-points-badge warning' : 'zone-points-badge active'}>
                  <span className={polygon.length < 3 ? 'badge-dot amber' : 'badge-dot green'} aria-hidden="true" />
                  {polygon.length < 3
                    ? `Đã vẽ ${polygon.length}/3 điểm`
                    : `${polygon.length} điểm đang hoạt động`}
                </span>
              )}
              {previewUrl && !intrusionEnabled && (
                <span className="status-badge disarmed">
                  <span className="badge-dot amber" aria-hidden="true" />
                  Vùng đang tắt
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
                      <img src={previewUrl} alt="Ảnh xem trước camera" onError={() => setImageError(true)} />
                    </div>
                  </div>
                )}
              </div>
            ) : imageError ? (
              <div className="preview-placeholder">
                <span className="preview-placeholder-icon error" aria-hidden="true">!</span>
                <h4>Kết nối RTSP thất bại</h4>
                <p>Kiểm tra URL luồng, thông tin đăng nhập và kết nối mạng của camera.</p>
              </div>
            ) : (
              <div className="preview-placeholder">
                <span className="preview-placeholder-icon" aria-hidden="true">+</span>
                <h4>Chưa chọn nguồn</h4>
                <p>Ảnh xem trước sẽ xuất hiện khi nguồn video sẵn sàng.</p>
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
