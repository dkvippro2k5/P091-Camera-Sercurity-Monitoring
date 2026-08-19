import { incidentMeta } from '../incidents'
import { isWithinWindow } from '../activeHours'

/**
 * `onDelete` and `onAddClick` are passed for a guard, who both watches the cameras and
 * configures them. Any other role reaching this screen gets them as `null` and sees the
 * read-only empty state below — nobody is shown a control that would come back a 403.
 *
 * As of the role swap that is a defensive branch rather than a live one: the manager is the
 * only other role and their nav does not offer the grid at all. It is kept because "render
 * a button that cannot work" is the failure it exists to prevent, and a third role would
 * arrive without anyone thinking to re-add it.
 */
export default function GridView({ cameras, alertingCameras, onDelete, onAddClick }) {
  if (cameras.length === 0) {
    // Two different messages, because the existing copy told the reader to add a camera and
    // a guard has no way to. "Add your first camera" beside no button reads as a broken UI;
    // worse, it points the wrong person at the problem while the grid stays dark.
    return (
      <div className="empty">
        <span className="empty-symbol" aria-hidden="true">+</span>
        <h2>No cameras yet</h2>
        {onAddClick ? (
          <button className="primary" onClick={onAddClick}>
            Add your first camera
          </button>
        ) : (
          <p>No cameras have been configured yet — until then there is nothing to monitor.</p>
        )}
      </div>
    )
  }

  const live = cameras.filter((c) => c.running).length

  return (
    <>
      <div className="stage-head grid-stage-head">
        <div>
          <p className="eyebrow">Guard workspace</p>
          <h1>Live monitoring</h1>
        </div>
        <div className="stage-head-actions">
          <span className={live === cameras.length ? 'count healthy' : 'count degraded'}>
            <span className="dot" aria-hidden="true" />
            {live} of {cameras.length} streaming
          </span>
          {onAddClick && <button onClick={onAddClick}>Add camera</button>}
        </div>
      </div>

      <div className="grid">
        {cameras.map((camera) => (
          <CameraTile
            key={camera.id}
            camera={camera}
            alerting={alertingCameras.has(camera.id)}
            onDelete={onDelete}
          />
        ))}
      </div>
    </>
  )
}

/** Whether this camera's zone window covers right now. No window means always armed. */
function intrusionArmed(camera) {
  if (!camera.zone_active_hours) return true
  const [start, end] = camera.zone_active_hours.split('-')
  return isWithinWindow(start, end)
}

function windowStart(window) {
  return String(window ?? '').split('-')[0]
}

function CameraTile({ camera, alerting, onDelete }) {
  return (
    <article className={alerting ? 'tile alerting' : 'tile'}>
      <div className="tile-head">
        {/* Status is a chip with a word in it, not a bare coloured dot. Green-vs-grey at
            8px was the only signal that a camera had stopped, which fails 1.4.1 Use of
            Colour and is invisible at wall-display distance regardless of vision. */}
        <span className={camera.running ? 'status-chip live' : 'status-chip offline'}>
          <span className="dot" aria-hidden="true" />
          {camera.running ? 'Live' : 'Offline'}
        </span>

        <span className="tile-name" title={camera.name}>
          {camera.name}
        </span>

        {/* Destructive, so it is pushed away from the camera name, given a real 32px
            target, and styled as danger on hover. It was a zero-padding underlined
            text link sitting directly beside the name — roughly 18px tall, under even
            the 24px WCAG 2.5.8 floor, for an action that deletes a camera.

            Absent entirely for a guard rather than disabled: a disabled control still
            says "this is yours to use, just not now", which is the wrong story for a
            permission the account will never have. */}
        {onDelete ? (
          <button
            className="icon compact danger spacer"
            onClick={() => onDelete(camera.id)}
            aria-label={`Remove camera ${camera.name}`}
            title={`Remove ${camera.name}`}
          >
            <span aria-hidden="true">✕</span>
          </button>
        ) : (
          <span className="spacer" />
        )}
      </div>

      <div className="feed-wrap">
        {/*
          MJPEG in a plain <img>. The browser decodes the multipart stream natively, so
          there is no player to initialise and a stalled camera simply holds its last
          frame instead of throwing. Frames are annotated with zones/tracks/events by the
          backend pipeline; face blurring was removed on 2026-08-10, so faces here are
          identifiable.
        */}
        <img className="feed" src={camera.stream_url} alt={`${camera.name} live feed`} />

        {alerting && (
          <span className="feed-badge">
            <span aria-hidden="true">▲</span> Incident
          </span>
        )}
      </div>

      <div className="tile-foot">
        {camera.active_labels.map((label) => (
          <span key={label} className="tag">
            {/* Intrusion outside its window is shown struck-through rather than hidden.
                A missing tag reads as "not configured"; the operator needs to see that it
                is configured and currently asleep, or a quiet grid looks like a working one. */}
            {label === 'intrusion' && !intrusionArmed(camera) ? (
              <span className="tag-dormant" title={`Intrusion armed ${camera.zone_active_hours} only`}>
                {incidentMeta(label).short} · off until {windowStart(camera.zone_active_hours)}
              </span>
            ) : (
              incidentMeta(label).short
            )}
          </span>
        ))}
        <span className="telemetry spacer">
          {camera.source_type} · {camera.inference_capacity_fps.toFixed(1)} fps · p95 {(camera.latency_p95_ms ?? 0).toFixed(0)} ms
        </span>
      </div>
    </article>
  )
}
