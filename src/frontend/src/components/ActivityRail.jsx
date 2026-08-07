import { incidentMeta, relativeTime, formatClock, triageOrder } from '../incidents'

/**
 * Persistent incident log beside the grid.
 *
 * This is the structural gap the old UI had: `GET /api/v1/alerts` existed and was wired
 * into `api.js`, but nothing ever rendered it. An alert's entire lifetime on screen was
 * the modal — so "what did I acknowledge two minutes ago?" and "how busy has this shift
 * been?" were both unanswerable, and a page refresh lost the lot.
 *
 * Unacknowledged items sort by severity (see `triageOrder`) and sit above the resolved
 * ones, which stay in arrival order as a plain audit trail.
 */
export default function ActivityRail({ alerts, unackedCount, onInspect, activeId }) {
  const pending = triageOrder(alerts.filter((a) => !a.acknowledged))
  const resolved = alerts.filter((a) => a.acknowledged).slice().reverse()

  return (
    <aside className="rail" aria-label="Incident activity">
      <div className="rail-head">
        <h2>Activity</h2>
        {unackedCount > 0 && (
          <span className="rail-count">
            {unackedCount}
            <span className="visually-hidden"> awaiting review</span>
          </span>
        )}
      </div>

      {alerts.length === 0 ? (
        <p className="rail-empty">No incidents recorded this session.</p>
      ) : (
        <ul className="rail-list">
          {pending.map((alert) => (
            <RailItem key={alert.id} alert={alert} onInspect={onInspect} activeId={activeId} unacked />
          ))}
          {resolved.map((alert) => (
            <RailItem key={alert.id} alert={alert} onInspect={onInspect} activeId={activeId} />
          ))}
        </ul>
      )}
    </aside>
  )
}

function RailItem({ alert, onInspect, activeId, unacked = false }) {
  const meta = incidentMeta(alert.label)
  const classes = ['rail-item', meta.severity]
  if (unacked) classes.push('unacked')

  return (
    <li>
      <button
        className={classes.join(' ')}
        onClick={() => onInspect(alert.id)}
        aria-current={alert.id === activeId ? 'true' : undefined}
      >
        {/* The thumbnail is the same anonymized data URL the modal shows (PRD §3.3);
            no second request, and no raw frame anywhere in the client. */}
        <img className="rail-thumb" src={alert.image_data_url} alt="" />
        <span className="rail-text">
          <span className="rail-title">
            {meta.short}
            {!unacked && <span className="visually-hidden"> (reviewed)</span>}
          </span>
          <span className="rail-sub">
            {alert.camera_name} ·{' '}
            <time dateTime={alert.created_at} title={formatClock(alert.created_at)}>
              {relativeTime(alert.created_at)}
            </time>
          </span>
        </span>
      </button>
    </li>
  )
}
