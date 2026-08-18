import { useState, useEffect, useCallback, useMemo } from 'react'
import GridView from './components/GridView'
import CameraSetup from './components/CameraSetup'
import AlertModal from './components/AlertModal'
import ActivityRail from './components/ActivityRail'
import Masthead from './components/Masthead'
import Login from './components/Login'
import StatsView from './components/StatsView'
import VerifyingCard from './components/VerifyingCard'
import { api, setUnauthorizedHandler } from './api'
import { triageOrder, isAutoSuppressed, isPendingTriage, requiresGuardReview } from './incidents'

const MAX_HISTORY = 60

// Which screens each role may reach. The guard runs the room — live grid plus camera setup;
// the manager reads statistics and is offered nothing else.
//
// The API enforces the same boundary. Shared monitoring health is exposed separately so a
// manager can keep the masthead current without receiving camera footage or incident data.
const SCREENS = {
  guard: ['grid', 'setup'],
  manager: ['stats'],
}

export default function App() {
  // `undefined` while the session check is in flight, `null` once known to be signed out.
  // Collapsing the two would flash the login screen on every refresh of an authenticated
  // session, which on a wall display looks like the system dropping out.
  const [user, setUser] = useState(undefined)
  const [screen, setScreen] = useState('grid')
  const [cameras, setCameras] = useState([])
  const [status, setStatus] = useState(null)
  // Every alert this session knows about, acknowledged or not. The modal shows the most
  // urgent unacknowledged one; the rail shows all of them.
  const [alerts, setAlerts] = useState([])
  // Set when the operator opens a past alert from the rail, so re-reading a resolved
  // incident does not require a new one to fire.
  const [inspectingId, setInspectingId] = useState(null)
  // Whether the guard has collapsed the activity rail sidebar on the live grid page (defaults to true so grid is un-obscured).
  const [railCollapsed, setRailCollapsed] = useState(true)

  // The alert half of the app — modal, rail, backlog badge, incident stream — is the guard's.
  const isGuard = user?.role === 'guard'

  const unacked = useMemo(() => triageOrder(alerts.filter(requiresGuardReview)), [alerts])
  const pendingAlerts = useMemo(() => alerts.filter(isPendingTriage), [alerts])
  const inspecting = useMemo(() => alerts.find((a) => a.id === inspectingId) ?? null, [alerts, inspectingId])
  // Only a guard gets interrupted. A manager reading statistics cannot action a fall — the
  // verdict route is the guard's — so a modal over their dashboard would be an interruption
  // whose only available response is to dismiss it, training them to dismiss alerts.
  const active = isGuard ? (inspecting ?? unacked[0] ?? null) : null

  // Everything authenticated hangs off this, so signing out clears the screen rather than
  // leaving the last shift's cameras and incidents painted behind the login form.
  const signedOut = useCallback(() => {
    setUser(null)
    setCameras([])
    setStatus(null)
    setAlerts([])
    setInspectingId(null)
    setScreen('grid')
  }, [])

  // One handler for every 401, registered once. Background polls (`loadStatus` every 3s)
  // only `console.error`, so without this a revoked session would leave a frozen grid on
  // screen with no indication that anything had happened.
  useEffect(() => {
    setUnauthorizedHandler(signedOut)
    return () => setUnauthorizedHandler(null)
  }, [signedOut])

  // Restore an existing session before deciding what to render. A 401 here is the normal
  // "not signed in" path, not an error worth logging.
  useEffect(() => {
    api
      .me()
      .then(setUser)
      .catch(() => setUser(null))
  }, [])

  const loadCameras = useCallback(async () => {
    try {
      setCameras(await api.listCameras())
    } catch (err) {
      console.error('Failed to load cameras:', err)
    }
  }, [])

  const loadStatus = useCallback(async () => {
    try {
      setStatus(await api.status())
    } catch (err) {
      console.error('Failed to load status:', err)
    }
  }, [])

  // Seeded once from GET /alerts so a page refresh mid-shift does not start the operator
  // with a blank incident history the backend had been retaining all along.
  const loadAlerts = useCallback(async () => {
    try {
      setAlerts((await api.listAlerts()).slice(-MAX_HISTORY))
    } catch (err) {
      console.error('Failed to load alerts:', err)
    }
  }, [])

  useEffect(() => {
    if (!user) return

    // Cameras and alerts feed only guard surfaces, so a manager fetches neither. `loadStatus`
    // runs for both: it is the masthead's camera vitals, and for a manager it is also the only
    // thing that still ends their session when they sign out in another tab — `auth.revoked`
    // arrives over the SSE stream a manager no longer opens, so the 401 from the next poll is
    // what drops them to the login screen (`api.js` → `setUnauthorizedHandler`). Cross-tab
    // logout is therefore instant for a guard and within 3s for a manager, by design.
    if (isGuard) {
      loadCameras()
      loadAlerts()
    }
    loadStatus()
    const interval = setInterval(loadStatus, 3000)
    return () => clearInterval(interval)
  }, [user, isGuard, loadCameras, loadStatus, loadAlerts])

  useEffect(() => {
    // Not opened until there is a session. An EventSource started while signed out would
    // 401, retry on its own schedule, and keep retrying forever — the reconnect loop this
    // whole path is arranged to avoid.
    //
    // Guards only, and the reason is not just that a manager renders no modal or rail. Every
    // `alert.created` payload carries `image_data_url`, a full annotated JPEG, and since the
    // face anonymizer was removed those frames show identifiable people. Streaming them to a
    // role with nothing to do with them would be footage leaving the building for no reason.
    if (!isGuard) return

    // SSE rather than polling: the pop-up is meant to interrupt, and EventSource
    // reconnects on its own if the backend restarts.
    const events = new EventSource(api.eventsUrl())

    events.addEventListener('alert.created', (e) => {
      const alert = JSON.parse(e.data)
      setAlerts((prev) => (prev.some((a) => a.id === alert.id) ? prev : [...prev, alert].slice(-MAX_HISTORY)))
    })

    // The AI assessment lands seconds after the alert itself — the alert is never held
    // waiting for it. Merge by id so an open modal fills in its assessment panel in place.
    events.addEventListener('alert.verdict', (e) => {
      const alert = JSON.parse(e.data)
      setAlerts((prev) => prev.map((a) => (a.id === alert.id ? { ...a, ...alert } : a)))
    })

    // Logout, from this tab or another one. Filtered by username because `_publish` fans out
    // to every subscriber with no per-user index — unfiltered, one guard signing out would
    // sign out everyone watching.
    events.addEventListener('auth.revoked', (e) => {
      try {
        if (JSON.parse(e.data).username === user.username) signedOut()
      } catch {
        /* malformed payload — better to stay signed in than to sign out on a parse error */
      }
    })

    events.addEventListener('camera.added', loadCameras)
    events.addEventListener('camera.removed', loadCameras)
    events.onerror = () => console.warn('Alert stream interrupted; EventSource will retry')

    return () => events.close()
  }, [user, isGuard, loadCameras, signedOut])

  async function handleLogin(username, password) {
    const account = await api.login(username, password)
    setUser(account)
    // Managers land on statistics, guards on the grid: each role's first screen is the one
    // it can actually act on.
    setScreen(account.role === 'manager' ? 'stats' : 'grid')
  }

  async function handleLogout() {
    try {
      await api.logout()
    } catch (err) {
      // The cookie is gone either way; staying on an authenticated screen because the
      // request failed would be worse than dropping to login.
      console.warn('Logout request failed:', err)
    } finally {
      signedOut()
    }
  }

  async function handleAddCamera(payload) {
    await api.addCamera(payload)
    await loadCameras()
    setScreen('setup')
  }

  async function handleDeleteCamera(id) {
    if (!window.confirm('Remove this camera? Its live feed and zone configuration are deleted.')) return
    await api.deleteCamera(id)
    await loadCameras()
  }

  async function handleVerdict(id, verdict, operatorNote = '', actualEventType = '') {
    let failure = null
    try {
      await api.submitVerdict(id, verdict, operatorNote, actualEventType)
    } catch (err) {
      // Surfaced rather than swallowed: a 403 means the judgement was not recorded, and the
      // guard needs to know that rather than believing the incident is dealt with.
      failure = err
    } finally {
      // Mark rather than drop, so the incident stays in the rail as history. Runs even
      // if the POST failed: a failed judgement must not trap the operator behind a modal
      // they cannot dismiss.
      setAlerts((prev) => prev.map((a) => (a.id === id ? {
        ...a,
        acknowledged: true,
        operator_verdict: verdict,
        operator_note: operatorNote || null,
        actual_event_type: actualEventType || null,
      } : a)))
      setInspectingId(null)
      loadStatus()
    }
    if (failure) window.alert(`That judgement was not recorded: ${failure.message}`)
  }

  const alertingCameras = useMemo(() => new Set(unacked.map((a) => a.camera_id)), [unacked])

  if (user === undefined) {
    return (
      <div className="login-page">
        <p className="hint">Checking your session…</p>
      </div>
    )
  }

  if (user === null) return <Login onLogin={handleLogin} />

  // A role that cannot reach the current screen is put back on its own first screen — not on
  // the grid. `screen` initialises to 'grid', and a manager restoring a session by refreshing
  // never passes through `handleLogin`, so a hardcoded 'grid' fallback here leaves them on a
  // screen no `current ===` branch matches: a blank stage under a working masthead.
  const allowed = SCREENS[user.role] ?? SCREENS.guard
  const current = allowed.includes(screen) ? screen : allowed[0]

  return (
    <div className={`app role-${user.role}`}>
      <Masthead
        screen={current}
        onNavigate={setScreen}
        status={status}
        unackedCount={unacked.length}
        user={user}
        screens={allowed}
        onLogout={handleLogout}
      />

      <div className={`workspace screen-${current} ${current === 'setup' ? 'no-rail' : ''} ${railCollapsed && current !== 'setup' ? 'rail-collapsed' : ''}`}>
        <main className="stage">
          <div className="stage-inner">
            {/* An explicit branch, not a ternary. It used to be `screen === 'grid' ? grid :
                setup`, where *any* unrecognised screen rendered the camera form — so adding a
                third screen silently showed the add-camera page to someone who asked for
                statistics. With three branches an unknown screen renders nothing instead,
                which is why the `allowed[0]` fallback above has to be right. */}
            {current === 'grid' && (
              <GridView
                cameras={cameras}
                alertingCameras={alertingCameras}
                onDelete={isGuard ? handleDeleteCamera : null}
                onAddClick={isGuard ? () => setScreen('setup') : null}
              />
            )}
            {current === 'setup' && (
              <CameraSetup
                onAdd={handleAddCamera}
                onCancel={() => setScreen('grid')}
                cameras={cameras}
                onDelete={handleDeleteCamera}
              />
            )}
            {current === 'stats' && <StatsView status={status} />}
          </div>
        </main>

        {/* The rail is the guard's incident history. Hidden on setup screen to maximize space. */}
        {isGuard && current !== 'setup' && (
          <ActivityRail
            alerts={alerts}
            unackedCount={unacked.length}
            onInspect={setInspectingId}
            activeId={active?.id ?? null}
            collapsed={railCollapsed}
            onToggleCollapse={() => setRailCollapsed((prev) => !prev)}
          />
        )}
      </div>

      {isGuard && pendingAlerts.length > 0 && <VerifyingCard pendingAlerts={pendingAlerts} />}

      {active && (
        <AlertModal
          alert={active}
          queued={unacked.filter((a) => a.id !== active.id).length}
          isHistory={active.acknowledged || isAutoSuppressed(active)}
          onVerdict={handleVerdict}
          onClose={() => setInspectingId(null)}
        />
      )}
    </div>
  )
}
