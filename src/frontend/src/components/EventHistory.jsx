import { useCallback, useEffect, useState } from 'react'
import { api } from '../api'
import { incidentMeta, formatClock, relativeTime } from '../incidents'

const PAGE_SIZE = 25

// The three states a row can be in, as the manager reads them. `unreviewed` is the absence of
// a verdict rather than a value, which is why it cannot be folded into the other two — see the
// same split in `MonitoringService.incident_history`.
const VERDICTS = [
  { value: '', label: 'Tất cả kết luận' },
  { value: 'unreviewed', label: 'Chưa xử lý' },
  { value: 'real', label: 'Sự cố thật' },
  { value: 'false', label: 'Báo động giả' },
]

/**
 * The manager's event history: every alert the system has ever raised, filterable.
 *
 * Reads `/incidents` — the durable log — rather than `/alerts`, which is the guard's live
 * ring buffer and holds only the current process's tail. On a screen labelled "lịch sử" that
 * difference is the whole point: after a restart the buffer is empty and the log is not.
 *
 * Refreshed by re-fetching rather than by a live stream. A manager reviewing last night's
 * incidents is reading, not responding, so the seconds of latency a poll costs buy nothing
 * back — and the alert SSE stream is guard-gated precisely because a manager has no modal to
 * interrupt. `App` re-mounts nothing here; the refresh is the filter change and the button.
 */
export default function EventHistory({ onInspect }) {
  const [page, setPage] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
  const [offset, setOffset] = useState(0)
  const [filters, setFilters] = useState({ cameraId: '', label: '', verdict: '' })

  const load = useCallback(async () => {
    setLoading(true)
    try {
      setPage(await api.incidents({ limit: PAGE_SIZE, offset, ...filters }))
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [offset, filters])

  useEffect(() => {
    load()
  }, [load])

  // Any filter change returns to the first page. Keeping the offset would land the operator
  // on page 4 of a two-page result and show them an empty list they did not ask for.
  function setFilter(key, value) {
    setOffset(0)
    setFilters((prev) => ({ ...prev, [key]: value }))
  }

  if (error) {
    return (
      <div className="stats-error" role="alert">
        <span className="status-mark danger">!</span>
        <div>
          <h4>Không thể tải lịch sử sự kiện</h4>
          <p>{error}</p>
        </div>
      </div>
    )
  }

  const items = page?.items ?? []
  const total = page?.total ?? 0
  const shownTo = Math.min(offset + PAGE_SIZE, total)

  return (
    <section className="event-history" aria-labelledby="history-title">
      <header className="stage-head">
        <div>
          <h2 id="history-title">Lịch sử sự kiện</h2>
          <p className="stage-subtitle">
            Toàn bộ cảnh báo đã ghi nhận, kể cả trước lần khởi động gần nhất.
          </p>
        </div>
        <button className="compact" onClick={load} disabled={loading}>
          {loading ? 'Đang tải…' : 'Làm mới'}
        </button>
      </header>

      <div className="history-filters">
        <label className="field compact-field">
          <span>Camera</span>
          <select value={filters.cameraId} onChange={(e) => setFilter('cameraId', e.target.value)}>
            <option value="">Tất cả camera</option>
            {(page?.cameras ?? []).map((name) => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>
        </label>

        <label className="field compact-field">
          <span>Loại sự cố</span>
          <select value={filters.label} onChange={(e) => setFilter('label', e.target.value)}>
            <option value="">Tất cả loại</option>
            {(page?.labels ?? []).map((label) => (
              <option key={label} value={label}>{incidentMeta(label).short}</option>
            ))}
          </select>
        </label>

        <label className="field compact-field">
          <span>Kết luận</span>
          <select value={filters.verdict} onChange={(e) => setFilter('verdict', e.target.value)}>
            {VERDICTS.map((option) => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
        </label>
      </div>

      {items.length === 0 ? (
        <div className="manager-empty">
          {loading ? 'Đang tải lịch sử…' : 'Không có sự kiện nào khớp với bộ lọc.'}
        </div>
      ) : (
        <>
          {/* The wrapper scrolls, not the page. A wide table inside a fixed-width column is
              the usual cause of a whole document scrolling sideways on a phone. */}
          <div className="history-table-wrap">
            <table className="visual-table history-table">
              <thead>
                <tr>
                  <th scope="col">Sự cố</th>
                  <th scope="col">Camera</th>
                  <th scope="col">Thời gian</th>
                  <th scope="col">Kết luận</th>
                  <th scope="col">Người xử lý</th>
                </tr>
              </thead>
              <tbody>
                {items.map((row) => (
                  <HistoryRow key={row.alert_id} row={row} onInspect={onInspect} />
                ))}
              </tbody>
            </table>
          </div>

          <div className="history-pager">
            <span className="hint">
              Hiển thị {offset + 1}–{shownTo} trong {total} sự kiện
            </span>
            <button
              className="compact"
              onClick={() => setOffset((prev) => Math.max(0, prev - PAGE_SIZE))}
              disabled={offset === 0 || loading}
            >
              Trang trước
            </button>
            <button
              className="compact"
              onClick={() => setOffset((prev) => prev + PAGE_SIZE)}
              disabled={shownTo >= total || loading}
            >
              Trang sau
            </button>
          </div>
        </>
      )}
    </section>
  )
}

function HistoryRow({ row, onInspect }) {
  const meta = incidentMeta(row.label)

  return (
    <tr>
      <td>
        {/* The row opens the detail view. A button rather than a click handler on the <tr>
            so it is reachable by keyboard and announced as an action. */}
        <button className="history-open" onClick={() => onInspect(row)}>
          {/* The thumbnail slot is rendered on every row and is always the same size, so the
              incident name after it starts at the same x whether or not this row has an
              evidence frame. Older rows have no evidence_url; without the empty slot their
              names slid left to the cell edge and the column read ragged. */}
          <span className="history-thumb" aria-hidden="true">
            {row.evidence_url && (
              <img src={row.evidence_url} alt="" loading="lazy" decoding="async" />
            )}
          </span>
          <span className="history-label">{meta.short}</span>
        </button>
      </td>
      <td>{row.camera_name}</td>
      <td>
        <time dateTime={row.created_at} title={formatClock(row.created_at)}>
          {formatClock(row.created_at)}
        </time>
        <span className="history-relative"> · {relativeTime(row.created_at)}</span>
      </td>
      <td><VerdictBadge verdict={row.operator_verdict} /></td>
      <td className="history-reviewer">{row.judged_by || '—'}</td>
    </tr>
  )
}

/* Text, not colour alone. Whether an incident was confirmed is the single most consequential
   thing on this screen, and a red-vs-green pill is unresolvable for a colour-blind reader. */
function VerdictBadge({ verdict }) {
  if (verdict === 'real') return <span className="history-verdict real">Sự cố thật</span>
  if (verdict === 'false') return <span className="history-verdict false">Báo động giả</span>
  return <span className="history-verdict pending">Chưa xử lý</span>
}
