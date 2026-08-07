import { incidentMeta } from '../incidents'

export default function GridView({ cameras, alertingCameras, onDelete, onAddClick }) {
  if (cameras.length === 0) {
    return (
      <div className="empty">
        <h2>No cameras yet</h2>
        <p>
          Add a camera to start monitoring. Fall and abandoned-object detection run automatically; intrusion
          detection needs a zone drawn on the camera&apos;s preview.
        </p>
        <button className="primary" onClick={onAddClick}>
          Add your first camera
        </button>
      </div>
    )
  }

  const live = cameras.filter((c) => c.running).length

  return (
    <>
      <div className="stage-head">
        <h2>Live grid</h2>
        <span className="count">
          {live} of {cameras.length} streaming
        </span>
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
            the 24px WCAG 2.5.8 floor, for an action that deletes a camera. */}
        <button
          className="icon compact danger spacer"
          onClick={() => onDelete(camera.id)}
          aria-label={`Remove camera ${camera.name}`}
          title={`Remove ${camera.name}`}
        >
          <span aria-hidden="true">✕</span>
        </button>
      </div>

      <div className="feed-wrap">
        {/*
          MJPEG in a plain <img>. The browser decodes the multipart stream natively, so
          there is no player to initialise and a stalled camera simply holds its last
          frame instead of throwing. The frames are already face-blurred and annotated
          with zones/tracks/events by the backend pipeline.
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
            {incidentMeta(label).short}
          </span>
        ))}
        <span className="telemetry spacer">
          {camera.source_type} · {camera.processed_frames.toLocaleString()} frames ·{' '}
          {camera.inference_capacity_fps.toFixed(1)} fps
        </span>
      </div>
    </article>
  )
}
