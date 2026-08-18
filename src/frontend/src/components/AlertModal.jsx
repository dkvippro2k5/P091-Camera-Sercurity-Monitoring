import { useState } from 'react'
import { useFocusTrap } from '../useFocusTrap'
import { incidentMeta, relativeTime, formatClock, triageMeta } from '../incidents'

export default function AlertModal({ alert, queued = 0, isHistory = false, onVerdict, onClose }) {
  const meta = incidentMeta(alert.label)
  const dialogRef = useFocusTrap(true)
  const [note, setNote] = useState(alert.operator_note || '')
  const [actualEventType, setActualEventType] = useState(alert.actual_event_type || '')
  const submit = (verdict) => onVerdict(alert.id, verdict, note, actualEventType)

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
            <span aria-hidden="true">!</span>
            {isHistory ? 'Reviewed incident' : `${meta.severity} priority`}
          </p>
          <h2 id="alert-heading">{meta.headline}</h2>
          <p className="modal-action" id="alert-action">
            {meta.action}
          </p>
        </div>

        <div className="modal-facts">
          <Fact label="Camera" value={alert.camera_name} />
          {alert.zone_name && <Fact label="Zone" value={alert.zone_name} />}
          <Fact label="Time" value={formatClock(alert.created_at)} />
          <Fact label="Detected" value={relativeTime(alert.created_at)} />
          <Fact label="Confidence" value={`${(alert.confidence * 100).toFixed(0)}%`} />
        </div>

        <div className="modal-body">
          <EvidenceFrames alert={alert} meta={meta} />

          {alert.reason && (
            <p className="reason">
              <span className="fact-label">Rule</span>
              <code>{alert.reason}</code>
            </p>
          )}

          <TriagePanel alert={alert} />
          {!alert.operator_verdict && (
            <div className="operator-feedback">
              <label>
                What actually happened?
                <select value={actualEventType} onChange={(event) => setActualEventType(event.target.value)}>
                  <option value="">Select if known</option>
                  <option value="real_incident">Real incident</option>
                  <option value="false_alarm">False alarm</option>
                  <option value="normal_activity">Normal activity</option>
                  <option value="maintenance">Maintenance / service activity</option>
                  <option value="other">Other</option>
                </select>
              </label>
              <label>
                Guard note
                <textarea value={note} onChange={(event) => setNote(event.target.value)} maxLength={2000} rows={3} placeholder="Describe what happened..." />
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
                    Recorded as {alert.operator_verdict === 'real' ? 'a real incident' : 'a false alarm'}
                  </span>
                  <button data-autofocus onClick={onClose}>
                    Close
                  </button>
                </>
              ) : (
                <>
                  <span className="queued">
                    {alert.triage_verdict === 'false_positive'
                      ? 'AI Filtered — Override AI verdict:'
                      : 'Past alert — Record verdict:'}
                  </span>
                  <button className="verdict-false" onClick={() => submit('false')}>
                    False alarm
                  </button>
                  <button className="primary" data-autofocus onClick={() => submit('real')}>
                    Confirm real
                  </button>
                </>
              )}
            </>
          ) : (
            <>
              {queued > 0 && (
                <span className="queued">
                  {queued} more {queued === 1 ? 'incident' : 'incidents'} waiting
                </span>
              )}
              <button className="verdict-false" onClick={() => submit('false')}>
                False alarm
              </button>
              <button className="primary" data-autofocus onClick={() => submit('real')}>
                Confirm real
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
        <span className="fact-label">AI assessment</span>
        <span className="triage-text">Assessing...</span>
      </p>
    )
  }

  const meta = triageMeta(alert.triage_verdict)
  const observations = Array.isArray(alert.triage_frame_observations) ? alert.triage_frame_observations : []

  return (
    <div className={`triage ${meta.tone}`} aria-live="polite">
      <span className="fact-label">AI assessment</span>
      <div className="triage-text">
        <strong>{meta.text}</strong>
        {typeof alert.triage_confidence === 'number' && alert.triage_verdict !== 'uncertain' && (
          <span className="triage-confidence"> ({(alert.triage_confidence * 100).toFixed(0)}% confident)</span>
        )}
        {alert.triage_rationale && <span className="triage-rationale"> - {alert.triage_rationale}</span>}
        {!alert.triage_rationale && alert.triage_error && (
          <span className="triage-rationale"> - assessment unavailable</span>
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

function EvidenceFrames({ alert, meta }) {
  const frames = Array.isArray(alert.evidence_frames) ? alert.evidence_frames : []

  if (frames.length === 0) {
    return (
      <img
        className="shot"
        src={alert.image_data_url}
        width="640"
        height="360"
        alt={`${meta.headline} on ${alert.camera_name}`}
      />
    )
  }

  return (
    <section className="evidence-strip" aria-label="VLM evidence frames">
      <div className="evidence-head">
        <span className="fact-label">Evidence clip</span>
        <span>{frames.length} chronological frames</span>
      </div>
      <div className="evidence-grid">
        {frames.map((frame) => (
          <figure key={`${frame.phase}-${frame.index}`} className="evidence-frame">
            <img src={frame.image_data_url} alt={`${frame.phase} frame ${frame.index} for ${meta.headline}`} />
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
