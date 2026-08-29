import { useEffect, useState } from 'react'
import { api } from '../api'
import { useFocusTrap } from '../useFocusTrap'
import { incidentMeta, relativeTime, formatClock, triageMeta } from '../incidents'

export default function AlertModal({ alert, queued = 0, isHistory = false, onVerdict, onClose }) {
  const meta = incidentMeta(alert.label)
  const dialogRef = useFocusTrap(true)
  const [note, setNote] = useState(alert.operator_note || '')
  const [actualEventType, setActualEventType] = useState(alert.actual_event_type || '')
  // The verdict write is a real network round trip to the database (recording history and
  // training context, not just an in-memory flag), so it is not instant. Without this, the
  // guard gets no feedback for that whole span and a slow response reads as the click not
  // having registered — the next queued alert *does* still appear once it resolves, but a
  // frozen-looking modal invites a second click (a duplicate submission) or the guard
  // wrongly concluding the queue is stuck.
  const [submitting, setSubmitting] = useState(false)
  const submit = async (verdict) => {
    if (submitting) return
    setSubmitting(true)
    try {
      await onVerdict(alert.id, verdict, note, actualEventType)
    } finally {
      // No reset-to-false-then-leave-mounted case to worry about here: a successful verdict
      // acknowledges this alert, which drops it out of the queue and unmounts this instance
      // (see the `key={active.id}` at the call site) — so `submitting` only needs to revert
      // on the failure path, where the guard stays on this same alert and must be able to
      // retry.
      setSubmitting(false)
    }
  }

  return (
    <div className="overlay" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <div
        className={`modal ${meta.severity}`}
        ref={dialogRef}
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="alert-heading"
        aria-describedby="alert-action"
      >
        <div className="modal-head">
          <p className="modal-eyebrow">
            <span aria-hidden="true">!</span>
            {isHistory ? 'Sự cố đã xử lý' : 'Cảnh báo ưu tiên'}
          </p>
          <h2 id="alert-heading">{meta.headline}</h2>
          <p className="modal-action" id="alert-action">
            {meta.action}
          </p>
        </div>

        <div className="modal-facts">
          <Fact label="Camera" value={alert.camera_name} />
          {alert.zone_name && <Fact label="Khu vực" value={alert.zone_name} />}
          <Fact label="Thời gian" value={`${formatClock(alert.created_at)} (${relativeTime(alert.created_at)})`} />
        </div>

        <div className="modal-body">
          <EvidenceFrames alert={alert} meta={meta} />

          <TriagePanel alert={alert} />
          <SimilarIncidents alertId={alert.id} />
          {!alert.operator_verdict && (
            <div className="operator-feedback">
              <label htmlFor="actual-event-type">
                Thực tế đã xảy ra điều gì?
                <select id="actual-event-type" value={actualEventType} onChange={(event) => setActualEventType(event.target.value)}>
                  <option value="">Chọn nếu đã xác định</option>
                  <option value="real_incident">Sự cố thật</option>
                  <option value="false_alarm">Báo động giả</option>
                  <option value="normal_activity">Hoạt động bình thường</option>
                  <option value="maintenance">Bảo trì / dịch vụ</option>
                  <option value="other">Khác</option>
                </select>
              </label>
              <label htmlFor="guard-note">
                Ghi chú của bảo vệ
                <textarea id="guard-note" value={note} onChange={(event) => setNote(event.target.value)} maxLength={2000} rows={3} placeholder="Mô tả sự việc…" />
              </label>
            </div>
          )}
        </div>

        <div className="modal-foot">
          {isHistory ? (
            <>
              {alert.operator_verdict ? (
                <>
                  <span className="queued">
                    Đã ghi nhận là {alert.operator_verdict === 'real' ? 'sự cố thật' : 'báo động giả'}
                  </span>
                  <button data-autofocus onClick={onClose}>
                    Đóng
                  </button>
                </>
              ) : (
                <>
                  <span className="queued">
                    {submitting
                      ? 'Đang ghi nhận…'
                      : alert.triage_verdict === 'false_positive'
                        ? 'AI đã lọc — ghi đè kết luận:'
                        : 'Cảnh báo cũ — ghi nhận kết luận:'}
                  </span>
                  <button className="verdict-false" onClick={() => submit('false')} disabled={submitting}>
                    Báo động giả
                  </button>
                  <button className="primary" data-autofocus onClick={() => submit('real')} disabled={submitting}>
                    Xác nhận sự cố
                  </button>
                </>
              )}
            </>
          ) : (
            <>
              {submitting ? (
                <span className="queued" aria-live="polite">Đang ghi nhận…</span>
              ) : (
                queued > 0 && (
                  <span className="queued">
                    Còn {queued} sự cố đang chờ
                  </span>
                )
              )}
              <button className="verdict-false" onClick={() => submit('false')} disabled={submitting}>
                Báo động giả
              </button>
              <button className="primary" data-autofocus onClick={() => submit('real')} disabled={submitting}>
                {submitting ? 'Đang xử lý…' : 'Xác nhận sự cố'}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

function TriagePanel({ alert }) {
  if (!alert.triage_status || alert.triage_status === 'skipped') return null

  if (alert.triage_status === 'pending') {
    return (
      <p className="triage triage-pending" aria-live="polite">
        <span className="fact-label">Đánh giá của AI</span>
        <span className="triage-text">Đang phân tích…</span>
      </p>
    )
  }

  const meta = triageMeta(alert.triage_verdict)
  const observations = Array.isArray(alert.triage_frame_observations) ? alert.triage_frame_observations : []

  return (
    <div className={`triage ${meta.tone}`} aria-live="polite">
      <span className="fact-label">Đánh giá của AI</span>
      <div className="triage-text">
        <strong>{meta.text}</strong>
        {alert.triage_rationale && <span className="triage-rationale"> - {alert.triage_rationale}</span>}
        {!alert.triage_rationale && alert.triage_error && (
          <span className="triage-rationale"> - không thể đánh giá</span>
        )}
        {observations.length > 0 && (
          <ol className="triage-observations" aria-label="Frame observations">
            {observations.map((observation, index) => (
              <li key={`${index}-${observation}`}>{observation}</li>
            ))}
          </ol>
        )}
      </div>
    </div>
  )
}

/**
 * Past guard-judged incidents whose rule-engine reason read similarly to this one's
 * (PLANNED feature, docs/database/schema.sql). Renders nothing rather than an empty-state
 * message when the list comes back empty -- that is the common case (feature off, or
 * nothing judged yet to compare against) and a guard mid-shift does not need a panel
 * explaining that.
 */
function SimilarIncidents({ alertId }) {
  const [items, setItems] = useState([])

  useEffect(() => {
    let cancelled = false
    setItems([])
    api
      .similarIncidents(alertId)
      .then((rows) => {
        if (!cancelled) setItems(rows)
      })
      .catch(() => {
        /* Feature is best-effort: a lookup failure should look like "nothing similar", not
           an error the guard has to deal with while judging a live alert. */
      })
    return () => {
      cancelled = true
    }
  }, [alertId])

  if (items.length === 0) return null

  return (
    <section className="similar-incidents" aria-label="Similar past incidents">
      <span className="fact-label">Similar past events</span>
      <ul>
        {items.map((item) => (
          <li key={item.alert_id}>
            {item.evidence_url && <img src={item.evidence_url} alt={`${item.label} on ${item.camera_name}`} />}
            <div className="similar-incident-text">
              <strong>{item.camera_name}</strong>
              <span>
                {Math.round(item.similarity * 100)}% similar - recorded as{' '}
                {item.operator_verdict === 'real' ? 'a real incident' : 'a false alarm'}
              </span>
            </div>
          </li>
        ))}
      </ul>
    </section>
  )
}

function EvidenceFrames({ alert, meta }) {
  const frames = Array.isArray(alert.evidence_frames) ? alert.evidence_frames : []

  if (frames.length === 0) {
    // Rendered only while this modal is open, so the fetch happens on open rather than when
    // the alert arrived — the images are no longer carried in the alert payload itself.
    if (!alert.evidence_url) return null
    return (
      <img
        className="shot"
        src={alert.evidence_url}
        width="640"
        height="360"
        alt={`${meta.headline} on ${alert.camera_name}`}
      />
    )
  }

  return (
    <section className="evidence-strip" aria-label="VLM evidence frames">
      <div className="evidence-head">
          <span className="fact-label">Chuỗi bằng chứng</span>
        <span>{frames.length} khung hình theo thời gian</span>
      </div>
      <div className="evidence-grid">
        {frames.map((frame) => (
          <figure key={`${frame.phase}-${frame.index}`} className="evidence-frame">
            <img
              src={frame.image_url}
              alt={`${frame.phase} frame ${frame.index} for ${meta.headline}`}
              loading="lazy"
              decoding="async"
            />
            <figcaption>
              <span>{frame.phase}</span>
              <strong>#{frame.index}</strong>
            </figcaption>
          </figure>
        ))}
      </div>
    </section>
  )
}

function Fact({ label, value }) {
  return (
    <div className="fact">
      <span className="fact-label">{label}</span>
      <span className="fact-value">{value}</span>
    </div>
  )
}
