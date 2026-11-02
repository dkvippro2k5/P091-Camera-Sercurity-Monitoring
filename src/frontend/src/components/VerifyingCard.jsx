import { incidentMeta, relativeTime } from '../incidents'

export default function VerifyingCard({ pendingAlerts }) {
  if (!pendingAlerts || pendingAlerts.length === 0) return null

  const alert = pendingAlerts[0]
  const meta = incidentMeta(alert.label)
  const count = pendingAlerts.length

  return (
    <div className="verifying-card" role="status" aria-live="polite">
      <div className="verifying-card-head">
        <span className="verifying-spinner" aria-hidden="true" />
        <span className="verifying-title">AI đang xác minh</span>
        {count > 1 && <span className="verifying-count">+{count - 1} more</span>}
      </div>
      <div className="verifying-card-body">
        {alert.evidence_url && (
          <img className="verifying-thumb" src={alert.evidence_url} alt="" decoding="async" />
        )}
        <div className="verifying-info">
          <strong>{meta.headline}</strong>
          <span>
            {alert.camera_name} • {relativeTime(alert.created_at)}
          </span>
        </div>
      </div>
    </div>
  )
}
