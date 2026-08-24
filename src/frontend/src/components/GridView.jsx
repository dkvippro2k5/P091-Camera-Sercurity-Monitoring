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
export default function GridView({ cameras, alertingCameras, onDelete, onReplay, onAddClick }) {
  if (cameras.length === 0) {
    // Two different messages, because the existing copy told the reader to add a camera and
    // a guard has no way to. "Add your first camera" beside no button reads as a broken UI;
    // worse, it points the wrong person at the problem while the grid stays dark.
    return (
      <div className="empty">
        {onAddClick ? (
          <button className="primary" onClick={onAddClick}>
            Thêm camera
          </button>
        ) : null}
      </div>
    )
  }

  return (
    <div className="grid">
        {cameras.map((camera) => (
          <CameraTile
            key={camera.id}
            camera={camera}
            alerting={alertingCameras.has(camera.id)}
            onDelete={onDelete}
            onReplay={onReplay}
          />
        ))}
    </div>
  )
}

function CameraTile({ camera, alerting, onDelete, onReplay }) {
  const ended = camera.stopped_reason === 'end_of_source'
  return (
    <article className={alerting ? 'tile alerting' : 'tile'}>
      <div className="feed-wrap">
        {/*
          MJPEG in a plain <img>. The browser decodes the multipart stream natively, so
          there is no player to initialise and a stalled camera simply holds its last
          frame instead of throwing. Frames are annotated with zones/tracks/events by the
          backend pipeline; face blurring was removed on 2026-08-10, so faces here are
          identifiable.
        */}
        <img className={ended ? 'feed ended' : 'feed'} src={camera.stream_url} alt={`Luồng trực tiếp ${camera.name}`} />

        <div className="feed-meta">
          <span className={camera.running ? 'feed-status live' : 'feed-status offline'} aria-label={camera.running ? 'Trực tiếp' : 'Ngoại tuyến'} />
        </div>

        {onDelete && (
          <button className="feed-action delete" onClick={() => onDelete(camera.id)} aria-label={`Xóa camera ${camera.name}`} title={`Xóa ${camera.name}`}>
            <svg aria-hidden="true" viewBox="0 0 24 24">
              <path d="M7 7l10 10M17 7 7 17" />
            </svg>
          </button>
        )}

        {ended && (
          onReplay && (
            <button className="feed-replay" onClick={() => onReplay(camera.id)} aria-label="Phát lại video" title="Phát lại video">
              <svg aria-hidden="true" viewBox="0 0 24 24">
                <path d="M4 4v6h6" />
                <path d="M5.4 15a8 8 0 1 0 .6-7.7L4 10" />
              </svg>
            </button>
          )
        )}

        {alerting && (
          <span className="feed-badge">
            <span aria-hidden="true">▲</span> Sự cố
          </span>
        )}
      </div>

    </article>
  )
}
