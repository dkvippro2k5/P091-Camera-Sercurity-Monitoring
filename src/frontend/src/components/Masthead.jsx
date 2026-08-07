/**
 * Top bar: identity, navigation, and the three system vitals.
 *
 * The vitals were previously three identical 12px grey spans, which meant the
 * unacknowledged-alert count — the only number an operator is expected to *act* on —
 * rendered with exactly the same weight as inference capacity, which they never act on.
 * The count now owns the largest type in the bar and turns red when non-zero, and the
 * region is aria-live so a screen reader announces it changing.
 */
export default function Masthead({ screen, onNavigate, status, unackedCount }) {
  const attention = unackedCount > 0

  return (
    <header className="masthead">
      <div className="brand">
        <span aria-hidden="true">◉</span>
        <span>Camera Monitor</span>
      </div>

      <nav aria-label="Views">
        <button aria-current={screen === 'grid' ? 'page' : undefined} onClick={() => onNavigate('grid')}>
          Live grid
        </button>
        <button aria-current={screen === 'setup' ? 'page' : undefined} onClick={() => onNavigate('setup')}>
          Cameras
        </button>
      </nav>

      <div className="vitals">
        <div className={attention ? 'vital attention' : 'vital'} aria-live="polite">
          <span className="vital-value">{unackedCount}</span>
          <span className="vital-label">{unackedCount === 1 ? 'Needs review' : 'Need review'}</span>
        </div>

        {status && (
          <>
            <div className="vital">
              <span className="vital-value">
                {status.active_cameras}/{status.total_cameras}
              </span>
              <span className="vital-label">Cameras live</span>
            </div>
            <div className="vital">
              <span className="vital-value">{status.average_inference_capacity_fps.toFixed(1)}</span>
              <span className="vital-label">fps capacity</span>
            </div>
          </>
        )}
      </div>
    </header>
  )
}
