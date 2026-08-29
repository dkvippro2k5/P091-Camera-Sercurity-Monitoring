import PolygonCanvas from './PolygonCanvas'
import { api } from '../api'
import { describeWindow } from '../activeHours'

const SOURCE_LABELS = { demo: 'Video mẫu', upload: 'Tải video lên', rtsp: 'Luồng RTSP' }

/**
 * One queued video in the multi-source setup grid: its own preview, its own optional
 * intrusion zone, its own schedule. Each tile is self-contained so adding, drawing on, or
 * removing one never touches the others' state — the grid is just a map over the queue.
 */
export default function PendingSourceTile({ entry, onChange, onRemove, disabled }) {
  const previewUrl = api.previewFrameUrl(entry.source_type, entry.source)
  const windowNote = describeWindow(entry.start_time, entry.end_time)
  const patch = (fields) => onChange(entry.key, fields)

  return (
    <article className="pending-tile">
      <div className="pending-tile-head">
        <span className="tag">{SOURCE_LABELS[entry.source_type]}</span>
        <input
          className="pending-tile-name"
          value={entry.name}
          onChange={(e) => patch({ name: e.target.value })}
          placeholder="Tên camera"
          disabled={disabled}
          aria-label="Tên camera"
        />
        <div className="pending-tile-head-actions">
          {entry.intrusion_enabled && (
            <span className={entry.polygon.length < 3 ? 'zone-points-badge warning' : 'zone-points-badge active'}>
              <span className={entry.polygon.length < 3 ? 'badge-dot amber' : 'badge-dot green'} aria-hidden="true" />
              {entry.polygon.length < 3 ? `${entry.polygon.length}/3 điểm` : `${entry.polygon.length} điểm`}
            </span>
          )}
          <button
            type="button"
            className="icon compact danger"
            onClick={() => onRemove(entry.key)}
            disabled={disabled}
            aria-label={`Bỏ ${entry.name || 'nguồn này'} khỏi danh sách`}
            title="Bỏ khỏi danh sách"
          >
            <span aria-hidden="true">✕</span>
          </button>
        </div>
      </div>

      <div className="pending-tile-body">
        {entry.image_error ? (
          <div className="pending-tile-placeholder">
            <span className="preview-placeholder-icon error" aria-hidden="true">!</span>
            <p>Không đọc được khung hình từ nguồn này.</p>
            <button type="button" className="compact" onClick={() => patch({ image_error: false })}>
              Thử lại
            </button>
          </div>
        ) : entry.intrusion_enabled ? (
          <PolygonCanvas
            imageUrl={previewUrl}
            points={entry.polygon}
            onChange={(polygon) => patch({ polygon })}
            onError={() => patch({ image_error: true })}
          />
        ) : (
          <div className="polygon-stage">
            <img src={previewUrl} alt={`Ảnh xem trước ${entry.name}`} onError={() => patch({ image_error: true })} />
          </div>
        )}
      </div>

      <div className="pending-tile-foot">
        <label className="toggle">
          <input
            type="checkbox"
            checked={entry.intrusion_enabled}
            onChange={(e) => patch({ intrusion_enabled: e.target.checked })}
            disabled={disabled}
          />
          <span className="toggle-text">
            <strong>Phát hiện xâm nhập</strong>
            <span className="hint">Cảnh báo khi có người đi vào vùng được vẽ trên ảnh xem trước.</span>
          </span>
        </label>

        {entry.intrusion_enabled && (
          <div className="zone-hours">
            <span className="fact-label">Khung giờ giám sát</span>

            <div className="choices">
              <label className="choice">
                <input
                  type="radio"
                  name={`zone_hours_${entry.key}`}
                  checked={!entry.hours_enabled}
                  onChange={() => patch({ hours_enabled: false })}
                />
                Giám sát 24/7
              </label>
              <label className="choice">
                <input
                  type="radio"
                  name={`zone_hours_${entry.key}`}
                  checked={entry.hours_enabled}
                  onChange={() => patch({ hours_enabled: true })}
                />
                Chỉ trong khung giờ
              </label>
            </div>

            {entry.hours_enabled && (
              <div className="time-range">
                <label className="field compact-field">
                  <span>Từ</span>
                  <input
                    type="time"
                    value={entry.start_time}
                    onChange={(e) => patch({ start_time: e.target.value })}
                    required
                  />
                </label>
                <label className="field compact-field">
                  <span>Đến</span>
                  <input
                    type="time"
                    value={entry.end_time}
                    onChange={(e) => patch({ end_time: e.target.value })}
                    required
                  />
                </label>
              </div>
            )}

            <div className="active-hours-status-card" aria-live="polite">
              <div className="status-header">
                <span className="status-label">Trạng thái giám sát</span>
                {!entry.hours_enabled || windowNote.armed ? (
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
              <small className={!entry.hours_enabled || windowNote.armed ? 'ok' : 'warn'}>
                {!entry.hours_enabled ? 'Luôn bật — giám sát liên tục 24/7.' : windowNote.text}
              </small>
            </div>
          </div>
        )}
      </div>
    </article>
  )
}
