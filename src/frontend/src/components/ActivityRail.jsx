import {
  incidentMeta,
  relativeTime,
  formatClock,
  triageOrder,
  isAutoSuppressed,
  isPendingTriage,
  requiresGuardReview,
} from '../incidents'

export default function ActivityRail({
  alerts,
  unackedCount,
  onInspect,
  activeId,
  collapsed = false,
  onToggleCollapse,
}) {
  const pendingReview = triageOrder(alerts.filter(requiresGuardReview))
  const pendingVerification = alerts.filter(isPendingTriage).slice().reverse()
  const filtered = alerts.filter((a) => !a.acknowledged && isAutoSuppressed(a)).slice().reverse()
  const resolved = alerts.filter((a) => a.acknowledged).slice().reverse()

  return (
    <>
      {onToggleCollapse && (
        <button
          className={collapsed ? 'rail-drawer-handle collapsed' : 'rail-drawer-handle'}
          onClick={(e) => {
            e.stopPropagation()
            onToggleCollapse()
          }}
          title={collapsed ? 'Mở danh sách sự cố' : 'Đóng danh sách sự cố'}
          aria-label={collapsed ? 'Mở danh sách sự cố' : 'Đóng danh sách sự cố'}
        >
          <svg className="handle-icon" aria-hidden="true" viewBox="0 0 24 24">
            <path d={collapsed ? 'M15 5l-7 7 7 7' : 'M9 5l7 7-7 7'} />
          </svg>
          {unackedCount > 0 && (
            <span className="handle-badge">{unackedCount}</span>
          )}
        </button>
      )}

    <aside className={collapsed ? 'rail drawer-rail collapsed' : 'rail drawer-rail'} aria-label="Danh sách sự cố">

      <div className="rail-head">
        <div><span className="eyebrow">Hàng đợi sự cố</span><h2>Hoạt động</h2></div>
        {unackedCount > 0 && (
          <span className="rail-count">
            {unackedCount}
            <span className="visually-hidden"> đang chờ xử lý</span>
          </span>
        )}
      </div>

      {alerts.length === 0 ? (
        <p className="rail-empty">Chưa có sự cố nào trong ca này.</p>
      ) : (
        <ul className="rail-list">
          {pendingReview.length > 0 && <li className="rail-section-label">Cần xử lý</li>}
          {pendingReview.map((alert) => (
            <RailItem key={alert.id} alert={alert} onInspect={onInspect} activeId={activeId} unacked />
          ))}
          {pendingVerification.length > 0 && <li className="rail-section-label">AI đang xác minh</li>}
          {pendingVerification.map((alert) => (
            <RailItem key={alert.id} alert={alert} onInspect={onInspect} activeId={activeId} pending />
          ))}
          {filtered.length > 0 && <li className="rail-section-label">Đã lọc</li>}
          {filtered.map((alert) => (
            <RailItem key={alert.id} alert={alert} onInspect={onInspect} activeId={activeId} autoFiltered />
          ))}
          {resolved.length > 0 && <li className="rail-section-label">Đã xem</li>}
          {resolved.map((alert) => (
            <RailItem key={alert.id} alert={alert} onInspect={onInspect} activeId={activeId} />
          ))}
        </ul>
      )}
    </aside>
    </>
  )
}

function RailItem({ alert, onInspect, activeId, unacked = false, pending = false, autoFiltered = false }) {
  const meta = incidentMeta(alert.label)
  const classes = ['rail-item', meta.severity]
  const autoSuppressed = autoFiltered || isAutoSuppressed(alert)

  if (unacked) classes.push('unacked')
  if (autoSuppressed) classes.push('auto-suppressed')
  if (pending) classes.push('pending-triage')

  return (
    <li>
      <button
        className={classes.join(' ')}
        onClick={() => onInspect(alert.id)}
        aria-current={alert.id === activeId ? 'true' : undefined}
      >
        <img className="rail-thumb" src={alert.image_data_url} alt="" />
        <span className="rail-text">
          <span className="rail-title">
            {meta.short}
            {!unacked && <span className="visually-hidden"> (đã xem)</span>}
          </span>
          <span className="rail-sub">
            {alert.camera_name} -{' '}
            <time dateTime={alert.created_at} title={formatClock(alert.created_at)}>
              {relativeTime(alert.created_at)}
            </time>
          </span>
          {pending && <span className="rail-badge pending">AI đang xác minh…</span>}
          {autoSuppressed && !pending && <span className="rail-badge">AI đã lọc</span>}
        </span>
      </button>
    </li>
  )
}
