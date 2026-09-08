import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import { useFocusTrap } from '../useFocusTrap'
import { incidentMeta, relativeTime, formatClock, triageMeta, alertPriority, similarIncidentQuery } from '../incidents'

/**
 * `actual_event_type` from the guard's one verdict choice, plus an optional category.
 *
 * The modal used to ask an open question ("Thực tế đã xảy ra điều gì?", optional, default
 * blank) *and* offer two verdict buttons — a guard could not tell which one was the
 * decision, and pressing a button without touching the dropdown wrote a null actual-event.
 * Now there is a single required verdict control ("Sự cố thật" / "Báo động giả") and one
 * Submit button. The category dropdown is a *why*, shown after a verdict is picked, and no
 * longer repeats the verdict itself.
 *
 * The two string values are unchanged on purpose — history rows and the stored-procedure
 * tests already read `real_incident`/`false_alarm`, so this changes where they come from,
 * never what they are. An explicit category still wins over the derived value.
 */
function resolveActualEventType(verdict, selected) {
  if (selected) return selected
  return verdict === 'real' ? 'real_incident' : 'false_alarm'
}

/** Vietnamese labels for the stored `actual_event_type` values, for read-only views. */
const ACTUAL_EVENT_LABELS = {
  real_incident: 'Sự cố thật',
  false_alarm: 'Báo động giả',
  normal_activity: 'Hoạt động bình thường',
  maintenance: 'Bảo trì / dịch vụ',
  other: 'Khác',
}

/**
 * `readOnly` is the manager's view of a past incident: the same evidence and AI assessment a
 * guard sees, with no way to record a judgement. It is distinct from `isHistory`, which is a
 * guard reopening an old alert — that one still offers the verdict buttons, because a guard
 * revisiting an unjudged incident is exactly who should be able to judge it late.
 *
 * The reason it is a prop rather than something inferred from the alert: the verdict route is
 * guard-gated, so a manager pressing those buttons gets a 403. Showing a control whose only
 * outcome is an error is worse than not showing it — and inferring "hide them" from a missing
 * verdict would also hide them from the guard, who is the one person meant to have them.
 */
export default function AlertModal({ alert, queued = 0, isHistory = false, readOnly = false, onVerdict, onClose }) {
  const meta = incidentMeta(alert.label)
  // Wording only — `meta` still owns the headline, the action line and the modal's severity
  // class, so a wrong AI verdict can never restyle a real emergency. See `alertPriority`.
  const priority = alertPriority(alert)
  const dialogRef = useFocusTrap(true)
  const [note, setNote] = useState(alert.operator_note || '')
  const [actualEventType, setActualEventType] = useState(alert.actual_event_type || '')
  // The guard's decision, made before Submit is enabled. Null until chosen; drives both the
  // recorded operator_verdict and the fallback actual_event_type.
  const [verdict, setVerdict] = useState(null)
  // The verdict write is a real network round trip to the database (recording history and
  // training context, not just an in-memory flag), so it is not instant. Without this, the
  // guard gets no feedback for that whole span and a slow response reads as the click not
  // having registered — the next queued alert *does* still appear once it resolves, but a
  // frozen-looking modal invites a second click (a duplicate submission) or the guard
  // wrongly concluding the queue is stuck.
  const [submitting, setSubmitting] = useState(false)
  const submit = async () => {
    if (submitting || !verdict) return
    setSubmitting(true)
    try {
      await onVerdict(alert.id, verdict, note, resolveActualEventType(verdict, actualEventType))
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
          <p className={isHistory ? 'modal-eyebrow' : `modal-eyebrow modal-eyebrow--${priority.tone}`}>
            <span aria-hidden="true">!</span>
            {isHistory ? 'Sự cố đã xử lý' : priority.eyebrow}
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
          <SimilarIncidents alert={alert} />
          {/* Inputs whose only submit path is the guard-gated verdict route, so they are not
              offered to a read-only viewer — a manager filling in a note that can never be
              saved is worse than not being asked. Their recorded values are shown instead. */}
          {readOnly && (alert.operator_note || alert.actual_event_type) && (
            <div className="reason">
              <span className="fact-label">Ghi nhận của bảo vệ</span>
              <span>
                {alert.actual_event_type && (
                  <strong>{ACTUAL_EVENT_LABELS[alert.actual_event_type] ?? alert.actual_event_type}</strong>
                )}
                {alert.operator_note && <span> {alert.operator_note}</span>}
              </span>
            </div>
          )}
          {!readOnly && !alert.operator_verdict && (
            <div className="operator-feedback">
              <fieldset className="verdict-choice">
                <legend>
                  {alert.triage_verdict === 'false_positive'
                    ? 'Kết luận của bạn (ghi đè AI):'
                    : 'Kết luận của bạn:'}
                </legend>
                <label>
                  <input
                    type="radio"
                    name="operator-verdict"
                    value="real"
                    checked={verdict === 'real'}
                    onChange={() => setVerdict('real')}
                    data-autofocus
                  />
                  Sự cố thật
                </label>
                <label>
                  <input
                    type="radio"
                    name="operator-verdict"
                    value="false"
                    checked={verdict === 'false'}
                    onChange={() => setVerdict('false')}
                  />
                  Báo động giả
                </label>
              </fieldset>
              {verdict && (
                <>
                  <label htmlFor="actual-event-type">
                    Phân loại (không bắt buộc)
                    <select id="actual-event-type" value={actualEventType} onChange={(event) => setActualEventType(event.target.value)}>
                      <option value="">Không xác định</option>
                      <option value="normal_activity">Hoạt động bình thường</option>
                      <option value="maintenance">Bảo trì / dịch vụ</option>
                      <option value="other">Khác</option>
                    </select>
                  </label>
                  <label htmlFor="guard-note">
                    Ghi chú của bảo vệ
                    <textarea id="guard-note" value={note} onChange={(event) => setNote(event.target.value)} maxLength={2000} rows={3} placeholder="Mô tả sự việc…" />
                  </label>
                </>
              )}
            </div>
          )}
        </div>

        <div className="modal-foot">
          {readOnly ? (
            <>
              <span className="queued">
                {alert.operator_verdict === 'real' && 'Đã ghi nhận là sự cố thật'}
                {alert.operator_verdict === 'false' && 'Đã ghi nhận là báo động giả'}
                {!alert.operator_verdict && 'Chưa có kết luận của bảo vệ'}
              </span>
              <button data-autofocus onClick={onClose}>
                Đóng
              </button>
            </>
          ) : isHistory && alert.operator_verdict ? (
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
              <span className="queued" aria-live="polite">
                {submitting
                  ? 'Đang ghi nhận…'
                  : isHistory
                    ? alert.triage_verdict === 'false_positive'
                      ? 'AI đã lọc — ghi đè kết luận'
                      : 'Cảnh báo cũ — ghi nhận kết luận'
                    : queued > 0
                      ? `Còn ${queued} sự cố đang chờ`
                      : ''}
              </span>
              <button className="primary" onClick={submit} disabled={submitting || !verdict}>
                {submitting ? 'Đang xử lý…' : 'Gửi kết luận'}
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
function SimilarIncidents({ alert }) {
  const [items, setItems] = useState([])
  // Collapsed by default. The panel is context for a decision, not the decision, and a guard
  // opening a live alert should meet the evidence and the verdict buttons first. Collapsed
  // also means the neighbour thumbnails are never requested unless asked for.
  const [expanded, setExpanded] = useState(false)

  const alertId = alert.id
  const { limit, minSimilarity } = similarIncidentQuery(alert)

  useEffect(() => {
    let cancelled = false
    setItems([])
    setExpanded(false)
    api
      .similarIncidents(alertId, { limit, minSimilarity })
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
  }, [alertId, limit, minSimilarity])

  if (items.length === 0) return null

  // Counted from what came back, never adjusted to agree with the AI. The whole value of the
  // panel is that it can contradict the assessment above it.
  const real = items.filter((item) => item.operator_verdict === 'real').length
  const falseAlarms = items.length - real

  return (
    <section className="similar-incidents" aria-label="Sự cố tương tự đã xảy ra">
      <button
        type="button"
        className="similar-incidents-toggle"
        aria-expanded={expanded}
        aria-controls="similar-incidents-list"
        onClick={() => setExpanded((previous) => !previous)}
      >
        <span className="fact-label">Sự cố tương tự ({items.length})</span>
        <span className="similar-incidents-split">
          {real} thật · {falseAlarms} giả
        </span>
        <span aria-hidden="true">{expanded ? '−' : '+'}</span>
      </button>
      {expanded && (
        <ul id="similar-incidents-list">
          {items.map((item) => (
            <SimilarIncidentRow key={item.alert_id} item={item} />
          ))}
        </ul>
      )}
    </section>
  )
}

/**
 * One past incident, with its evidence strip available on demand.
 *
 * The strip is fetched per row and only when opened, because rebuilding it probes storage
 * frame by frame server-side. The thumbnail alone is the detection-moment frame, which for a
 * fall is the least conclusive one there is — so a guard weighing this as precedent needs the
 * sequence, and gets it in one click rather than by default.
 */
function SimilarIncidentRow({ item }) {
  const [frames, setFrames] = useState(null)
  const [loading, setLoading] = useState(false)
  // Index of the frame open in the lightbox, or null. The strip's thumbnails are ~199px
  // wide inside a 680px modal, which is enough to pick a frame out of the sequence and far
  // too small to judge a posture from — so the thumbnails choose, and this shows.
  const [zoomed, setZoomed] = useState(null)
  // The fetch below resolves after an await, and this row can be gone by then: judging the
  // alert remounts the whole modal (`key={active.id}` at the call site), and collapsing the
  // panel unmounts the list. Same guard `SimilarIncidents` uses on its own lookup.
  const mounted = useRef(true)
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
    }
  }, [])

  const showFrames = async () => {
    if (frames !== null) {
      setFrames(null)
      setZoomed(null)
      return
    }
    setLoading(true)
    try {
      const loaded = await api.incidentFrames(item.alert_id)
      if (mounted.current) setFrames(loaded)
    } catch {
      if (mounted.current) setFrames([]) // Same best-effort stance as the lookup itself.
    } finally {
      if (mounted.current) setLoading(false)
    }
  }

  return (
    <li>
      <div className="similar-incident-row">
        {item.evidence_url && <img src={item.evidence_url} alt={`${item.label} tại ${item.camera_name}`} />}
        <div className="similar-incident-text">
          <strong>{item.camera_name}</strong>
          <span>
            Tương đồng {Math.round(item.similarity * 100)}% — ghi nhận là{' '}
            {item.operator_verdict === 'real' ? 'sự cố thật' : 'báo động giả'}
          </span>
        </div>
        <button type="button" className="similar-incident-frames-toggle" onClick={showFrames} disabled={loading}>
          {loading ? 'Đang tải…' : frames !== null ? 'Ẩn khung hình' : 'Xem khung hình'}
        </button>
      </div>
      {frames !== null && frames.length > 0 && (
        <div className="evidence-grid similar-incident-frames">
          {frames.map((frame, index) => (
            <figure key={`${frame.phase}-${frame.index}`} className="evidence-frame">
              <button
                type="button"
                className="evidence-frame-zoom"
                onClick={() => setZoomed(index)}
                aria-label={`Phóng to khung hình ${frame.index}`}
              >
                <img src={frame.image_url} alt={`Khung hình ${frame.index} của sự cố tương tự`} loading="lazy" decoding="async" />
              </button>
              <figcaption>
                <span>{frame.phase}</span>
                <strong>#{frame.index}</strong>
              </figcaption>
            </figure>
          ))}
        </div>
      )}
      {frames !== null && frames.length === 0 && !loading && (
        <p className="similar-incident-empty">Không còn khung hình được lưu cho sự cố này.</p>
      )}
      {zoomed !== null && frames && frames[zoomed] && (
        <FrameLightbox
          frames={frames}
          index={zoomed}
          caption={`${item.camera_name} — ${item.operator_verdict === 'real' ? 'sự cố thật' : 'báo động giả'}`}
          onNavigate={setZoomed}
          onClose={() => setZoomed(null)}
        />
      )}
    </li>
  )
}

/**
 * One evidence frame at full size, with the rest of the sequence a keypress away.
 *
 * The strip's thumbnails sit three-across inside a 680px modal, so each is about 199px wide
 * — enough to tell frames apart, nowhere near enough to judge a posture, which is the whole
 * question a guard opens a past incident to answer. Widening the modal was the alternative
 * and was rejected: it would relayout the alert dialog every time this panel expands, under
 * a guard's hands, next to the verdict buttons.
 *
 * Escape closes here, unlike the alert dialog that contains it. That trap deliberately
 * ignores Escape because dismissing an alert writes an acknowledgement to the incident
 * record; closing a zoomed photograph writes nothing, so the reflex should just work. It is
 * rendered inside the modal rather than at the document root so the alert's own focus trap
 * keeps containing it, and the arrow keys stay scoped to this sequence.
 */
function FrameLightbox({ frames, index, caption, onNavigate, onClose }) {
  const frame = frames[index]

  useEffect(() => {
    function onKeyDown(event) {
      if (event.key === 'Escape') {
        // Stops the alert dialog's own Tab-trapping handler from also seeing this, and
        // keeps Escape from meaning two things at once.
        event.stopPropagation()
        onClose()
      } else if (event.key === 'ArrowLeft' && index > 0) {
        event.stopPropagation()
        onNavigate(index - 1)
      } else if (event.key === 'ArrowRight' && index < frames.length - 1) {
        event.stopPropagation()
        onNavigate(index + 1)
      }
    }
    document.addEventListener('keydown', onKeyDown, true)
    return () => document.removeEventListener('keydown', onKeyDown, true)
  }, [index, frames.length, onNavigate, onClose])

  return (
    <div
      className="frame-lightbox"
      role="dialog"
      aria-modal="true"
      aria-label={`Khung hình ${frame.index} của ${frames.length}`}
      onMouseDown={(event) => event.target === event.currentTarget && onClose()}
    >
      <div className="frame-lightbox-inner">
        <img src={frame.image_url} alt={`Khung hình ${frame.index}, ${frame.phase}`} />
        <div className="frame-lightbox-bar">
          <button
            type="button"
            onClick={() => onNavigate(index - 1)}
            disabled={index === 0}
            aria-label="Khung hình trước"
          >
            ‹
          </button>
          <span>
            {caption} · <strong>{frame.phase}</strong> · #{frame.index}/{frames.length}
          </span>
          <button
            type="button"
            onClick={() => onNavigate(index + 1)}
            disabled={index === frames.length - 1}
            aria-label="Khung hình sau"
          >
            ›
          </button>
          <button type="button" className="frame-lightbox-close" data-autofocus onClick={onClose}>
            Đóng
          </button>
        </div>
      </div>
    </div>
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
