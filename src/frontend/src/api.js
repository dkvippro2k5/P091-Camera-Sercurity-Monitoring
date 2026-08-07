const BASE = '/api/v1'

async function request(path, options = {}) {
  const response = await fetch(`${BASE}${path}`, options)
  if (!response.ok) {
    // FastAPI puts the useful message in `detail`; surfacing the raw status alone would
    // hide "zone polygon needs 3 points" behind a bare 422.
    let detail = response.statusText
    try {
      const body = await response.json()
      detail = body.detail ?? detail
    } catch {
      /* non-JSON error body */
    }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return response.status === 204 ? null : response.json()
}

export const api = {
  listDemoVideos: () => request('/demo-videos'),
  listCameras: () => request('/cameras'),
  status: () => request('/monitoring/status'),
  listAlerts: () => request('/alerts'),

  addCamera: (payload) =>
    request('/cameras', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),

  deleteCamera: (id) => request(`/cameras/${id}`, { method: 'DELETE' }),
  acknowledge: (id) => request(`/alerts/${id}/acknowledge`, { method: 'POST' }),

  uploadVideo: async (file) => {
    const form = new FormData()
    form.append('file', file)
    return request('/uploads', { method: 'POST', body: form })
  },

  previewFrameUrl: (sourceType, source) =>
    `${BASE}/preview-frame?source_type=${sourceType}&source=${encodeURIComponent(source)}`,

  streamUrl: (id) => `${BASE}/cameras/${id}/stream`,
  eventsUrl: () => `${BASE}/events`,
}
