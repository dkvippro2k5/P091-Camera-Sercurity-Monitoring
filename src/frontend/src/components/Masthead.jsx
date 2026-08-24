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
  grid: 'Giám sát',
  setup: 'Camera',
  stats: 'Báo cáo',
}

const ROLE_LABEL = { guard: 'Bảo vệ', manager: 'Quản lý' }

export default function Masthead({ screen, onNavigate, status, unackedCount, user, screens, onLogout }) {
  const attention = unackedCount > 0
  const accountInitials = String(user?.display_name || user?.username || 'U')
    .split(/\s+/)
    .map((part) => part[0])
    .join('')
    .slice(0, 2)
    .toUpperCase()
  // A manager sees this count on their dashboard but is not the one clearing it, so the
  // masthead does not shout at them about a queue they cannot work.
  const showBacklog = user?.role === 'guard'

  return (
    <header className="masthead">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true">CM</span>
        <span className="brand-copy"><strong>A-eyes</strong><small>Trung tâm vận hành</small></span>
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
            <span className="vital-label">Chờ xử lý</span>
          </div>
        )}

        {status && (
          <>
            <div className="vital">
              <span className="vital-value">
                {status.active_cameras}/{status.total_cameras}
              </span>
              <span className="vital-label">Camera đang chạy</span>
            </div>
            <div className="vital">
              <span className="vital-value">{status.average_inference_capacity_fps.toFixed(1)}</span>
              <span className="vital-label">Năng lực FPS</span>
            </div>
          </>
        )}

        {/* Who is signed in, spelled out with the role. Verdicts are attributed to this
            account (`judged_by` in the labelled history), so an operator working someone
            else's unlocked station needs to be able to see whose name it will carry. */}
        {user && (
          <div className="account">
            <span className="account-avatar" aria-hidden="true">{accountInitials}</span>
            <span className="account-copy">
              <span className="account-name">{user.display_name || user.username}</span>
              <span className="account-role">{ROLE_LABEL[user.role] ?? user.role}</span>
            </span>
          </div>
        )}
        {user && (
          <button className="compact" onClick={onLogout}>
            Đăng xuất
          </button>
        )}
      </div>
    </header>
  )
}
