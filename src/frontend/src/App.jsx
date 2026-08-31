import { useState, useEffect, useCallback, useMemo } from 'react'
import GridView from './components/GridView'
import CameraSetup from './components/CameraSetup'
import AlertModal from './components/AlertModal'
import ActivityRail from './components/ActivityRail'
import Masthead from './components/Masthead'
import Login from './components/Login'
import StatsView from './components/StatsView'
import EventHistory from './components/EventHistory'
import VerifyingCard from './components/VerifyingCard'
import NotificationToggle from './components/NotificationToggle'
import { api, clearApiCache, setUnauthorizedHandler } from './api'
import { triageOrder, isAutoSuppressed, isPendingTriage, requiresGuardReview } from './incidents'

const MAX_HISTORY = 60

// Which screens each role may reach. The guard runs the room — live grid plus camera setup;
// the manager reads statistics and the event history.
//
// The API enforces the same boundary, and draws it at read-versus-act rather than at the two
// roles: the history screen and the evidence stills it links to sit on the shared operator
// gate (`require_operator`), while live footage and every write — verdicts, acknowledgements,
// camera configuration — stay guard-only. So a manager can review what happened and cannot
// change any of it, which is what the read-only detail view below relies on.
const SCREENS = {
  guard: ['grid', 'setup'],
  manager: ['stats', 'history'],
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
  // The manager's history detail, held as the row itself rather than an id. It comes from
  // `/incidents` (the durable log), not from `alerts` — so there is no list here to look an
  // id up in, and reusing `inspectingId` would silently resolve to `null` and render nothing.
  const [historyRow, setHistoryRow] = useState(null)
  const [dismissedAlertIds, setDismissedAlertIds] = useState(() => new Set())
  // Whether the guard has collapsed the activity rail sidebar on the live grid page (defaults to true so grid is un-obscured).
  // Always start closed on a fresh page load; the handle remains available to pull it out.
  const [railCollapsed, setRailCollapsed] = useState(true)
  // Which of the manager's screens have been opened at least once this session.
  //
  // Those two are rendered once and then kept mounted, hidden rather than unmounted, so
  // switching tabs does not throw their state away. Unmounting them meant every return to
  // the report screen restarted from `data === null` — a "Đang chuẩn bị báo cáo…" placeholder
  // and a fresh `/statistics` call that re-reads three JSONL files on the server — and every
  // return to the history screen forgot which filter and page the operator was on. Neither
  // is a page you visit once: a manager flips between them.
  //
  // Tracked in a set rather than rendering both from the start so the work is still deferred
  // until the screen is actually asked for. A guard never adds an entry here at all.
  const [visitedScreens, setVisitedScreens] = useState(() => new Set())

  // The alert half of the app — modal, rail, backlog badge, incident stream — is the guard's.
  const isGuard = user?.role === 'guard'

  const unacked = useMemo(() => triageOrder(alerts.filter(requiresGuardReview)), [alerts])
  const pendingAlerts = useMemo(() => alerts.filter(isPendingTriage), [alerts])
  const inspecting = useMemo(() => alerts.find((a) => a.id === inspectingId) ?? null, [alerts, inspectingId])
  // Only a guard gets interrupted. A manager reading statistics cannot action a fall — the
  // verdict route is the guard's — so a modal over their dashboard would be an interruption
  // whose only available response is to dismiss it, training them to dismiss alerts.
  const active = isGuard ? (inspecting ?? unacked.find((alert) => !dismissedAlertIds.has(alert.id)) ?? null) : null

  // Everything authenticated hangs off this, so signing out clears the screen rather than
  // leaving the last shift's cameras and incidents painted behind the login form.
  const signedOut = useCallback(() => {
    setUser(null)
    setCameras([])
    setStatus(null)
    setAlerts([])
    setInspectingId(null)
    setHistoryRow(null)
    setDismissedAlertIds(new Set())
    // Unmounts the kept-alive manager screens, which is what actually discards the statistics
    // and incident rows they were holding. Without this the next operator to sign in at the
    // same terminal would briefly see the previous one's report behind the login form.
    setVisitedScreens(new Set())
    // The component state above lives and dies with the mount; the `/statistics` cache does
    // not, so it has to be emptied by hand or it would outlive the session that fetched it.
    clearApiCache()
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
    const interval = setInterval(() => {
      loadStatus()
      if (isGuard) loadCameras()
    }, 3000)
    return () => clearInterval(interval)
  }, [user, isGuard, loadCameras, loadStatus, loadAlerts])

  useEffect(() => {
    // Not opened until there is a session. An EventSource started while signed out would
    // 401, retry on its own schedule, and keep retrying forever — the reconnect loop this
    // whole path is arranged to avoid.
    //
    // Guards only. Alert payloads no longer carry the frame itself — just an
    // `/api/v1/evidence/{id}` URL, which `require_guard` gates on the server, so footage
    // still never reaches a manager. Keeping the stream guard-only stays correct regardless:
    // a manager renders no modal and no rail, so the events would have no reader.
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
    events.addEventListener('camera.updated', loadCameras)
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

  // Sequential, not Promise.all: `MultiStreamManager` starts each worker (RTSP/file open +
  // model load) synchronously in `add_camera`, so firing every request at once would just
  // queue them behind each other on the server anyway while looking like a burst of failures
  // if one is slow. One partial failure should not lose the cameras that did succeed, so
  // failures are collected and reported together after every payload has been tried.
  //
  // Navigation only follows success: this is the only add path now (the setup screen queues
  // every source — one RTSP camera or twenty uploads — through this same call), so throwing
  // after navigating away would mean CameraSetup unmounts before it can show the error it was
  // just handed. On a total failure the screen stays put and the queue survives; on a partial
  // failure the grid is still where the cameras that *did* start actually show up, so the
  // miss is surfaced with an alert instead of trapping the operator behind an error banner
  // for cameras that are already running.
  async function handleAddCameraBatch(payloads) {
    const failures = []
    let addedCount = 0
    for (const payload of payloads) {
      try {
        await api.addCamera(payload)
        addedCount += 1
      } catch (err) {
        failures.push(`${payload.name}: ${err.message}`)
      }
    }
    await loadCameras()
    if (addedCount > 0) {
      setScreen('grid')
      if (failures.length) window.alert(`Một số camera không thể thêm được:\n${failures.join('\n')}`)
    } else if (failures.length) {
      throw new Error(failures.join('; '))
    }
  }

  async function handleDeleteCamera(id) {
    if (!window.confirm('Xóa camera này? Luồng trực tiếp và cấu hình vùng giám sát cũng sẽ bị xóa.')) return
    await api.deleteCamera(id)
    await loadCameras()
  }

  async function handleReplayCamera(id) {
    try {
      await api.replayCamera(id)
      await loadCameras()
    } catch (err) {
      window.alert(`Không thể phát lại video: ${err.message}`)
    }
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
      setDismissedAlertIds((prev) => {
        const next = new Set(prev)
        next.delete(id)
        return next
      })
      loadStatus()
    }
    if (failure) window.alert(`Không thể ghi nhận kết luận: ${failure.message}`)
  }

  const alertingCameras = useMemo(() => new Set(unacked.map((a) => a.camera_id)), [unacked])

  function handleCloseAlert() {
    if (active) {
      setDismissedAlertIds((prev) => new Set(prev).add(active.id))
    }
    setInspectingId(null)
  }

  // A role that cannot reach the current screen is put back on its own first screen — not on
  // the grid. `screen` initialises to 'grid', and a manager restoring a session by refreshing
  // never passes through `handleLogin`, so a hardcoded 'grid' fallback here leaves them on a
  // screen no `current ===` branch matches: a blank stage under a working masthead.
  //
  // Computed above the early returns below, not after them: `visitedScreens` is filled from an
  // effect keyed on it, and a hook cannot sit after a conditional return. `user` is still
  // `undefined`/`null` on those paths, so `allowed` falls back to the guard's list and
  // `current` is a value nothing renders from — harmless, because the returns below run first.
  const allowed = SCREENS[user?.role] ?? SCREENS.guard
  const current = allowed.includes(screen) ? screen : allowed[0]

  // Records a screen the first time it is actually shown, which is what lets the two manager
  // screens stay mounted afterwards. Guarded on `user` so the login and session-check renders
  // above do not record 'grid' for a session that does not exist yet, and it only ever grows
  // — `signedOut` is what empties it, so one operator's visits do not survive into the next
  // one's session on a shared terminal.
  useEffect(() => {
    if (!user) return
    setVisitedScreens((prev) => (prev.has(current) ? prev : new Set(prev).add(current)))
  }, [user, current])

  if (user === undefined) {
    return (
      <div className="login-page">
        <p className="hint">Đang kiểm tra phiên đăng nhập…</p>
      </div>
    )
  }

  if (user === null) return <Login onLogin={handleLogin} />

  return (
    <div className={`app role-${user.role} ${active ? 'modal-open' : ''}`}>
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
              <>
              <NotificationToggle />
              <GridView
                cameras={cameras}
                alertingCameras={alertingCameras}
                onDelete={isGuard ? handleDeleteCamera : null}
                onReplay={isGuard ? handleReplayCamera : null}
                onAddClick={isGuard ? () => setScreen('setup') : null}
              />
              </>
            )}
            {current === 'setup' && (
              <CameraSetup
                onAddBatch={handleAddCameraBatch}
                onCancel={() => setScreen('grid')}
                cameras={cameras}
                onDelete={handleDeleteCamera}
              />
            )}
            {/* The manager's two screens are hidden rather than unmounted, so flipping between
                them keeps the report already fetched and the history on the page and filter
                the operator had chosen. Unmounting reset both to their empty state and fired a
                fresh request every time — on `/statistics` that is three JSONL files re-read
                on the server for a screen the operator was looking at seconds ago.

                `visitedScreens` gates the first mount so nothing is fetched for a screen never
                opened; after that the wrapper only toggles `hidden`. The guard's screens keep
                unmounting: the grid holds live video sockets that should close when it is not
                on screen, and the setup form is a task you finish and leave. */}
            {visitedScreens.has('stats') && (
              <div hidden={current !== 'stats'}>
                <StatsView status={status} />
              </div>
            )}
            {visitedScreens.has('history') && (
              <div hidden={current !== 'history'}>
                <EventHistory onInspect={setHistoryRow} />
              </div>
            )}
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

      {/* The manager's read-only detail view: same evidence and AI assessment, no verdict
          controls — those post to a guard-gated route, so offering them here would only
          produce a 403. `readOnly` is what selects that footer; see the note on AlertModal
          for why it is a separate flag from `isHistory`.

          The row is reshaped to `Alert` here rather than inside `EventHistory`: `/incidents`
          keys the id as `alert_id` and the modal reads `alert.id`, and doing it at this one
          boundary keeps the history component free of the modal's field names. */}
      {historyRow && (
        <AlertModal
          key={historyRow.alert_id}
          alert={{ ...historyRow, id: historyRow.alert_id }}
          isHistory
          readOnly
          onVerdict={() => {}}
          onClose={() => setHistoryRow(null)}
        />
      )}

      {active && (
        <AlertModal
          // Keyed so a fresh AlertModal instance mounts per alert: without this, switching
          // from one queued alert to the next reuses the same component instance, and its
          // local state (note, actualEventType, the submitting flag below) would leak from
          // the alert just judged onto the one that replaces it.
          key={active.id}
          alert={active}
          queued={unacked.filter((a) => a.id !== active.id).length}
          isHistory={active.acknowledged || isAutoSuppressed(active)}
          onVerdict={handleVerdict}
          onClose={handleCloseAlert}
        />
      )}
    </div>
  )
}
