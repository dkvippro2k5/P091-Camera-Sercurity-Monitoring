import { useFocusTrap } from '../useFocusTrap'
import { incidentMeta, relativeTime, formatClock } from '../incidents'

export default function AlertModal({ alert, queued = 0, isHistory = false, onAcknowledge, onClose }) {
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
            The captured frame, exactly as the rule engine saw it — already anonymized.
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

          {/* The rule engine's own evidence — which rule fired and on what. This is the
              whole explanation an alert carries; there is no generated caption behind it. */}
          {alert.reason && (
            <p className="reason">
              <span className="fact-label">Rule</span>
              <code>{alert.reason}</code>
            </p>
          )}
        </div>

        <div className="modal-foot">
          {isHistory ? (
            <button data-autofocus onClick={onClose}>
              Close
            </button>
          ) : (
            <>
              {queued > 0 && (
                <span className="queued">
                  {queued} more {queued === 1 ? 'incident' : 'incidents'} waiting
                </span>
              )}
              {/* data-autofocus, so Enter resolves the alert the instant it appears —
                  the operator never has to find the button with the mouse first. */}
              <button className="primary" data-autofocus onClick={() => onAcknowledge(alert.id)}>
                Acknowledge
              </button>
            </>
          )}
        </div>
      </div>
    </div>
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
