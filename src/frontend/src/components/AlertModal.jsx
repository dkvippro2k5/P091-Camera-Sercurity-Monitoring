import { useFocusTrap } from '../useFocusTrap'
import { incidentMeta, relativeTime, formatClock, triageMeta } from '../incidents'

export default function AlertModal({ alert, queued = 0, isHistory = false, onVerdict, onClose }) {
  const meta = incidentMeta(alert.label)
  // Moves focus in on open, wraps Tab, and restores focus on close. The dialog declared
  // role="alertdialog" + aria-modal before but never took focus, so a keyboard operator
  // was left tabbing through the grid hidden behind the overlay.
  const dialogRef = useFocusTrap(true)

  return (
    <div className="overlay">
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
            <span aria-hidden="true">▲</span>
            {isHistory ? 'Reviewed incident' : `${meta.severity} priority`}
          </p>
          <h2 id="alert-heading">{meta.headline}</h2>
          <p className="modal-action" id="alert-action">
            {meta.action}
          </p>
        </div>

        {/* Facts as labelled pairs. Previously one run-on sentence — "Lobby (cam_3) ·
            zone north · confidence 62%" — which forces the operator to parse prose at
            the moment they are least able to. */}
        <div className="modal-facts">
          <Fact label="Camera" value={alert.camera_name} />
          {alert.zone_name && <Fact label="Zone" value={alert.zone_name} />}
          <Fact label="Time" value={formatClock(alert.created_at)} />
          <Fact label="Detected" value={relativeTime(alert.created_at)} />
          <Fact label="Confidence" value={`${(alert.confidence * 100).toFixed(0)}%`} />
        </div>

        <div className="modal-body">
          {/*
            The captured frame, exactly as the rule engine saw it.
            width/height are declared so the browser reserves the box before the JPEG
            decodes; without them the modal grew as the image landed and the Acknowledge
            button slid under an already-moving cursor.
          */}
          <img
            className="shot"
            src={alert.image_data_url}
            width="640"
            height="360"
            alt={`${meta.headline} on ${alert.camera_name}`}
          />

          {/* The rule engine's own evidence — which rule fired and on what. */}
          {alert.reason && (
            <p className="reason">
              <span className="fact-label">Rule</span>
              <code>{alert.reason}</code>
            </p>
          )}

          {/* The second-stage assessment. Advisory only — it never suppressed this alert,
              and it must not restyle the modal's severity, or a wrong verdict would
              visually downgrade a real emergency. */}
          <TriagePanel alert={alert} />
        </div>

        <div className="modal-foot">
          {isHistory ? (
            <>
              {alert.operator_verdict && (
                <span className="queued">
                  Recorded as {alert.operator_verdict === 'real' ? 'a real incident' : 'a false alarm'}
                </span>
              )}
              <button data-autofocus onClick={onClose}>
                Close <kbd className="hotkey-badge">Esc</kbd>
              </button>
            </>
          ) : (
            <>
              {queued > 0 && (
                <span className="queued">
                  {queued} more {queued === 1 ? 'incident' : 'incidents'} waiting
                </span>
              )}
              <button className="danger" onClick={() => onVerdict(alert.id, 'false')}>
                False alarm <kbd className="hotkey-badge">F</kbd>
              </button>
              <button className="primary" data-autofocus onClick={() => onVerdict(alert.id, 'real')}>
                Confirm real <kbd className="hotkey-badge">↵ Enter</kbd>
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

/**
 * "AI assessment" — the VLM's reading of the rule engine's evidence plus a short clip.
 *
 * Rendered as a labelled aside rather than anything resembling a decision, because it is
 * advisory: every alert reaches the operator regardless of what this says. `skipped` is
 * shown as nothing at all — an assessment that never ran is not an inconclusive one, and
 * an empty panel is more honest than "unknown".
 */
function TriagePanel({ alert }) {
  if (!alert.triage_status || alert.triage_status === 'skipped') return null

  if (alert.triage_status === 'pending') {
    return (
      <p className="triage triage-pending" aria-live="polite">
        <span className="fact-label">AI assessment</span>
        <span className="triage-text">Assessing…</span>
      </p>
    )
  }

  const meta = triageMeta(alert.triage_verdict)
  return (
    <p className={`triage ${meta.tone}`} aria-live="polite">
      <span className="fact-label">AI assessment</span>
      <span className="triage-text">
        <strong>{meta.text}</strong>
        {typeof alert.triage_confidence === 'number' && alert.triage_verdict !== 'uncertain' && (
          <span className="triage-confidence"> ({(alert.triage_confidence * 100).toFixed(0)}% confident)</span>
        )}
        {alert.triage_rationale && <span className="triage-rationale"> — {alert.triage_rationale}</span>}
        {!alert.triage_rationale && alert.triage_error && (
          <span className="triage-rationale"> — assessment unavailable</span>
        )}
      </span>
    </p>
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
