/**
 * Top bar: identity, navigation, and the three system vitals.
 *
 * The vitals were previously three identical 12px grey spans, which meant the
 * unacknowledged-alert count — the only number an operator is expected to *act* on —
 * rendered with exactly the same weight as inference capacity, which they never act on.
 * The count now owns the largest type in the bar and turns red when non-zero, and the
 * region is aria-live so a screen reader announces it changing.
 *
 * `screens` is the list of screens this role may reach, computed in `App`. Filtering the nav
 * from the same list the screen switch uses means the two cannot drift into a tab that
 * renders nothing.
 */
const NAV = {
  grid: 'Live grid',
  setup: 'Cameras',
  stats: 'Statistics',
}

const ROLE_LABEL = { guard: 'Guard', manager: 'Manager' }

export default function Masthead({ screen, onNavigate, status, unackedCount, user, screens, onLogout }) {
  const attention = unackedCount > 0
  // A manager sees this count on their dashboard but is not the one clearing it, so the
  // masthead does not shout at them about a queue they cannot work.
  const showBacklog = user?.role === 'guard'

  return (
    <header className="masthead">
      <div className="brand">
        <span aria-hidden="true">◉</span>
        <span>Camera Monitor</span>
      </div>

      <nav aria-label="Views">
        {screens.map((key) => (
          <button key={key} aria-current={screen === key ? 'page' : undefined} onClick={() => onNavigate(key)}>
            {NAV[key]}
          </button>
        ))}
      </nav>

      <div className="vitals">
        {showBacklog && (
          <div className={attention ? 'vital attention' : 'vital'} aria-live="polite">
            <span className="vital-value">{unackedCount}</span>
            <span className="vital-label">{unackedCount === 1 ? 'Needs review' : 'Need review'}</span>
          </div>
        )}

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

        {/* Who is signed in, spelled out with the role. Verdicts are attributed to this
            account (`judged_by` in the labelled history), so an operator working someone
            else's unlocked station needs to be able to see whose name it will carry. */}
        {user && (
          <div className="account">
            <span className="account-name">{user.display_name || user.username}</span>
            <span className="account-role">{ROLE_LABEL[user.role] ?? user.role}</span>
          </div>
        )}
        {user && (
          <button className="compact" onClick={onLogout}>
            Sign out
          </button>
        )}
      </div>
    </header>
  )
}
