const BASE = '/api/v1'

/**
 * Called when any request comes back 401, so the app can drop to the login screen.
 *
 * A module-level hook rather than a thrown-and-caught error type: a 401 can surface from a
 * background poll (`loadStatus` runs every 3s) whose caller only ever `console.error`s, so
 * relying on each call site to notice would leave a signed-out operator staring at a frozen
 * grid. `App` registers the one handler at mount.
 */
let onUnauthorized = () => {}

/**
 * Absolute ws:// or wss:// URL for a path under BASE.
 *
 * WebSocket, unlike fetch, has no relative form and no `credentials` option — it always
 * sends the origin's cookies, and always needs a full URL. Deriving the scheme from
 * `location.protocol` rather than hardcoding it is what keeps this working behind TLS.
 */
function wsUrl(path) {
  const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${scheme}//${window.location.host}${BASE}${path}`
}

export function setUnauthorizedHandler(handler) {
  onUnauthorized = typeof handler === 'function' ? handler : () => {}
}

/**
 * The last `/statistics` body this session saw, so reopening the report screen paints from it
 * instead of a placeholder while the fresh copy is on its way. See `api.statistics`.
 *
 * Deliberately module-level rather than component state: the point is to survive the screen
 * being unmounted. That also means it survives a *session* change, which it must not — the
 * report is one operator's view of their own site. `clearApiCache` is what empties it, and
 * `App.signedOut` calls it on every logout and every 401.
 */
let _statisticsCache = null

function cacheStatistics(body) {
  _statisticsCache = body
  return body
}

/**
 * Drop everything cached from a previous session.
 *
 * Called on sign-out and on any 401. Without it, the next operator to sign in at a shared
 * terminal would see the previous one's figures for the moment before the refresh lands.
 */
export function clearApiCache() {
  _statisticsCache = null
}

async function request(path, options = {}) {
  const response = await fetch(`${BASE}${path}`, {
    ...options,
    // The session is an HttpOnly cookie — see `src/services/sessions.py`. Same-origin in
    // both dev (Vite proxy) and prod (nginx), but `fetch` still needs telling: its default
    // is `same-origin`, which sends cookies, yet being explicit is what keeps this working
    // if the frontend is ever served from a different host than the API.
    credentials: 'include',
    // Merged, never replaced. `addCamera`/`submitVerdict` pass `Content-Type: application/json`,
    // and `uploadVideo` deliberately passes *no* headers so the browser can set the multipart
    // boundary itself — overwriting either one breaks that call.
    headers: { ...(options.headers ?? {}) },
  })

  if (!response.ok) {
    // Handled before the throw so it fires even for callers that swallow the error.
    if (response.status === 401) onUnauthorized()

    // FastAPI puts the useful message in `detail`; surfacing the raw status alone would
    // hide "zone polygon needs 3 points" behind a bare 422.
    let detail = response.statusText
    try {
      const body = await response.json()
      detail = body.detail ?? detail
    } catch {
      /* non-JSON error body */
    }
    const error = new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
    // Carried so a caller can tell "you may not do this" (403) from "that was invalid" (422).
    // The verdict handler uses it to explain a refused judgement instead of failing silently.
    error.status = response.status
    throw error
  }
  return response.status === 204 ? null : response.json()
}

export const api = {
  // --- Auth -----------------------------------------------------------------------------
  login: (username, password) =>
    request('/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    }),

  logout: () => request('/auth/logout', { method: 'POST' }),

  // Who am I — the app calls this on load to restore a session the browser still holds a
  // cookie for, so a refresh mid-shift does not force a re-login.
  me: () => request('/auth/me'),

  // --- Guard surface --------------------------------------------------------------------
  listCameras: () => request('/cameras'),
  status: () => request('/monitoring/status'),
  listAlerts: () => request('/alerts'),
  notificationSettings: () => request('/notifications'),
  setNotificationEnabled: (enabled) => request('/notifications', { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled }) }),

  deleteCamera: (id) => request(`/cameras/${id}`, { method: 'DELETE' }),
  replayCamera: (id) => request(`/cameras/${id}/replay`, { method: 'POST' }),
  acknowledge: (id) => request(`/alerts/${id}/acknowledge`, { method: 'POST' }),

  // Empty list whenever the feature is off or nothing similar has been judged yet -- never
  // an error the modal needs to handle specially. See docs/database/schema.sql (PLANNED).
  similarIncidents: (id) => request(`/alerts/${id}/similar`),

  // The HITL judgement (PRD §3.2). Distinct from acknowledge: this one records whether the
  // incident was real, and is the only call that appends to the labelled history.
  submitVerdict: (id, verdict, operatorNote = '', actualEventType = '') =>
    request(`/alerts/${id}/verdict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        verdict,
        operator_note: operatorNote.trim() || null,
        actual_event_type: actualEventType.trim() || null,
      }),
    }),

  // --- Camera setup, the guard's -----------------------------------------------------------
  listDemoVideos: () => request('/demo-videos'),

  addCamera: (payload) =>
    request('/cameras', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),

  uploadVideo: async (file) => {
    const form = new FormData()
    form.append('file', file)
    return request('/uploads', { method: 'POST', body: form })
  },

  // --- Manager surface ----------------------------------------------------------------------
  /**
   * The manager's aggregate report.
   *
   * `cached` asks for the last response this session received, if there is one, and refreshes
   * it in the background — the screen paints immediately with slightly stale numbers instead
   * of a placeholder. Only the *first* load of the screen passes it; the WebSocket-driven
   * refresh does not, because that call exists precisely to fetch numbers the cache does not
   * have yet and serving it from the cache would make the live update never arrive.
   *
   * Worth being explicit about the staleness this introduces: a reader can see figures up to
   * one refresh old for the moment the screen opens. That is the trade being made — this
   * response costs three JSONL files re-read on the server, and the alternative is a spinner
   * on every visit to a screen the operator was looking at a minute ago. The revalidation
   * lands within the same second, and `/manager/live` keeps it current after that.
   */
  statistics: (cached = false) => {
    if (!cached) return request('/statistics').then(cacheStatistics)
    if (_statisticsCache) {
      // Fire and forget: the caller already has something to render, and a failed refresh
      // must not reject a promise that was resolved from the cache.
      request('/statistics').then(cacheStatistics).catch(() => {})
      return Promise.resolve(_statisticsCache)
    }
    return request('/statistics').then(cacheStatistics)
  },

  // The durable event log, paged. Distinct from `listAlerts`, which is the guard's live ring
  // buffer: this one survives a restart and is what the history screen reads. Undefined
  // filters are dropped rather than sent as the string "undefined", which the route's
  // `verdict` pattern would reject with a 422.
  incidents: ({ limit = 50, offset = 0, cameraId, label, verdict } = {}) => {
    const params = new URLSearchParams({ limit, offset })
    if (cameraId) params.set('camera_id', cameraId)
    if (label) params.set('label', label)
    if (verdict) params.set('verdict', verdict)
    return request(`/incidents?${params}`)
  },

  previewFrameUrl: (sourceType, source) =>
    `${BASE}/preview-frame?source_type=${sourceType}&source=${encodeURIComponent(source)}`,

  streamUrl: (id) => `${BASE}/cameras/${id}/stream`,
  snapshotUrl: (id) => `${BASE}/cameras/${id}/snapshot`,
  eventsUrl: () => `${BASE}/events`,
  managerLiveUrl: () => wsUrl('/manager/live'),
  // H.264/fMP4 socket the grid prefers over `streamUrl`'s MJPEG. See `VideoFeed`.
  videoUrl: (id) => wsUrl(`/cameras/${id}/video`),
}
