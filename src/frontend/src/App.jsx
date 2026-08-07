import { useState, useEffect, useCallback, useMemo } from 'react'
import GridView from './components/GridView'
import CameraSetup from './components/CameraSetup'
import AlertModal from './components/AlertModal'
import ActivityRail from './components/ActivityRail'
import Masthead from './components/Masthead'
import { api } from './api'
import { triageOrder } from './incidents'

const MAX_HISTORY = 60

export default function App() {
  const [screen, setScreen] = useState('grid')
  const [cameras, setCameras] = useState([])
  const [status, setStatus] = useState(null)
  // Every alert this session knows about, acknowledged or not. The modal shows the most
  // urgent unacknowledged one; the rail shows all of them. Previously only the modal
  // existed, so acknowledging an alert erased it from the interface with no history.
  const [alerts, setAlerts] = useState([])
  // Set when the operator opens a past alert from the rail, so re-reading a resolved
  // incident does not require a new one to fire.
  const [inspectingId, setInspectingId] = useState(null)

  const unacked = useMemo(() => triageOrder(alerts.filter((a) => !a.acknowledged)), [alerts])
  const inspecting = useMemo(() => alerts.find((a) => a.id === inspectingId) ?? null, [alerts, inspectingId])
  const active = inspecting ?? unacked[0] ?? null

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

  // Seeded once from GET /alerts, which the UI never called before — a page refresh
  // during a shift used to start the operator with a blank incident history even
  // though the backend had been retaining it all along.
  const loadAlerts = useCallback(async () => {
    try {
      setAlerts((await api.listAlerts()).slice(-MAX_HISTORY))
    } catch (err) {
      console.error('Failed to load alerts:', err)
    }
  }, [])

  useEffect(() => {
    loadCameras()
    loadStatus()
    loadAlerts()
    const interval = setInterval(loadStatus, 3000)
    return () => clearInterval(interval)
  }, [loadCameras, loadStatus, loadAlerts])

  useEffect(() => {
    // SSE rather than polling: the pop-up is meant to interrupt, and EventSource
    // reconnects on its own if the backend restarts.
    const events = new EventSource(api.eventsUrl())

    events.addEventListener('alert.created', (e) => {
      const alert = JSON.parse(e.data)
      setAlerts((prev) => (prev.some((a) => a.id === alert.id) ? prev : [...prev, alert].slice(-MAX_HISTORY)))
    })

    events.addEventListener('camera.added', loadCameras)
    events.addEventListener('camera.removed', loadCameras)
    events.onerror = () => console.warn('Alert stream interrupted; EventSource will retry')

    return () => events.close()
  }, [loadCameras])

  async function handleAddCamera(payload) {
    await api.addCamera(payload)
    await loadCameras()
    setScreen('grid')
  }

  async function handleDeleteCamera(id) {
    if (!window.confirm('Remove this camera? Its live feed and zone configuration are deleted.')) return
    await api.deleteCamera(id)
    await loadCameras()
  }

  async function handleAcknowledge(id) {
    try {
      await api.acknowledge(id)
    } finally {
      // Mark rather than drop, so the incident stays in the rail as history. Runs even
      // if the POST failed: a failed acknowledge must not trap the operator behind a
      // modal they cannot dismiss.
      setAlerts((prev) => prev.map((a) => (a.id === id ? { ...a, acknowledged: true } : a)))
      setInspectingId(null)
      loadStatus()
    }
  }

  const alertingCameras = useMemo(() => new Set(unacked.map((a) => a.camera_id)), [unacked])

  return (
    <div className="app">
      <Masthead screen={screen} onNavigate={setScreen} status={status} unackedCount={unacked.length} />

      <div className="workspace">
        <main className="stage">
          <div className="stage-inner">
            {screen === 'grid' ? (
              <GridView
                cameras={cameras}
                alertingCameras={alertingCameras}
                onDelete={handleDeleteCamera}
                onAddClick={() => setScreen('setup')}
              />
            ) : (
              <CameraSetup onAdd={handleAddCamera} onCancel={() => setScreen('grid')} />
            )}
          </div>
        </main>

        <ActivityRail
          alerts={alerts}
          unackedCount={unacked.length}
          onInspect={setInspectingId}
          activeId={active?.id ?? null}
        />
      </div>

      {active && (
        <AlertModal
          alert={active}
          queued={unacked.filter((a) => a.id !== active.id).length}
          isHistory={active.acknowledged}
          onAcknowledge={handleAcknowledge}
          onClose={() => setInspectingId(null)}
        />
      )}
    </div>
  )
}
