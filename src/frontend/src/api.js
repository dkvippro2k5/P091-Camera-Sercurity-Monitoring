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
  statistics: () => request('/statistics'),

  previewFrameUrl: (sourceType, source) =>
    `${BASE}/preview-frame?source_type=${sourceType}&source=${encodeURIComponent(source)}`,

  streamUrl: (id) => `${BASE}/cameras/${id}/stream`,
  snapshotUrl: (id) => `${BASE}/cameras/${id}/snapshot`,
  eventsUrl: () => `${BASE}/events`,
  managerLiveUrl: () => wsUrl('/manager/live'),
  // H.264/fMP4 socket the grid prefers over `streamUrl`'s MJPEG. See `VideoFeed`.
  videoUrl: (id) => wsUrl(`/cameras/${id}/video`),
}
