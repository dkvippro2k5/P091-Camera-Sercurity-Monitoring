/**
 * Presentation metadata for the three PRD §3.1 incident labels.
 *
 * One module rather than a const in each component: `GridView`, `AlertModal` and
 * `ActivityRail` all render the same labels, and they previously disagreed — the grid
 * said "Fall", the modal said "FALL DETECTED", and neither knew the label's urgency.
 *
 * `rank` exists because the alert queue was strictly FIFO. A fall is a suspected medical
 * emergency; an abandoned bag is something to go look at within the hour. Ordering those
 * by arrival time means a collapsed person can wait behind a rucksack, which is a triage
 * failure the UI was silently causing. Lower rank surfaces first.
 */

export const INCIDENTS = {
  fall_detection: {
    short: 'Ngã',
    headline: 'Phát hiện té ngã',
    rank: 0,
    severity: 'critical',
    // Shown under the headline so the operator knows what the alert is asking of them.
    // The label alone doesn't say "go now" vs "review later".
    action: 'Có thể là cấp cứu — hãy kiểm tra ngay.',
  },
  intrusion: {
    short: 'Xâm nhập',
    headline: 'Phát hiện xâm nhập',
    rank: 1,
    severity: 'high',
    action: 'Có người đã vào khu vực hạn chế.',
  },
  abandoned_object: {
    short: 'Vật bị bỏ lại',
    headline: 'Phát hiện vật bị bỏ lại',
    rank: 2,
    severity: 'medium',
    action: 'Có vật không có người trông coi. Hãy kiểm tra khi có thể.',
  },
}

const FALLBACK = { short: 'Sự cố', headline: 'Sự cố', rank: 9, severity: 'medium', action: '' }

export function incidentMeta(label) {
  return INCIDENTS[label] ?? { ...FALLBACK, short: label, headline: label }
}

/**
 * Severity first, then oldest-first within a severity band.
 *
 * Returns a new array — callers hold this in React state, where mutating in place
 * would skip the re-render.
 */
export function triageOrder(alerts) {
  return [...alerts].sort((a, b) => {
    const rankDelta = incidentMeta(a.label).rank - incidentMeta(b.label).rank
    if (rankDelta !== 0) return rankDelta
    return new Date(a.created_at) - new Date(b.created_at)
  })
}

/**
 * How the second-stage AI assessment reads on screen.
 *
 * Deliberately plain wording. "Looks like a false alarm" is a hedge because that is what
 * the verdict is — advisory input to the guard's decision, not the decision. `tone` drives
 * a subtle border colour only; it must never restyle the modal's severity, or a wrong
 * verdict would visually downgrade a real emergency.
 */
const TRIAGE = {
  real: { text: 'Có vẻ là sự cố thật', tone: 'triage-real' },
  false_positive: { text: 'Có vẻ là báo động giả', tone: 'triage-false' },
  uncertain: { text: 'Chưa thể kết luận', tone: 'triage-uncertain' },
}

export function triageMeta(verdict) {
  return TRIAGE[verdict] ?? TRIAGE.uncertain
}

/**
 * Confidence bands shared by the modal's eyebrow and the similar-incident lookup.
 *
 * `TRIAGE_HIGH` is 0.85 and `TRIAGE_LOW` marks the point under which the model is not really
 * asserting anything. These two only choose wording and how many neighbours to show, so they
 * stay constants — unlike the auto-suppression threshold (`isAutoSuppressed`), which governs
 * whether a guard sees the alert at all and is therefore a served setting.
 */
export const TRIAGE_HIGH = 0.85
export const TRIAGE_LOW = 0.6

/**
 * The line above the headline, graded by what the VLM actually concluded.
 *
 * Only the eyebrow. `incidentMeta`'s headline, action and severity stay keyed to the label,
 * for the reason given on TRIAGE above: the modal's severity styling must never follow the
 * AI verdict, or a confidently wrong "false positive" would visually downgrade a real
 * medical emergency. Wording can hedge; the colour of a fall alert cannot.
 *
 * Every path that is not a completed triage — VLM off, queue saturated, call failed, or a
 * null confidence — returns the original fixed string, so a deployment with VLM disabled
 * reads exactly as it did before this existed.
 */
export function alertPriority(alert) {
  if (alert.triage_status === 'pending') {
    return { eyebrow: 'Đang xác minh', tone: 'neutral' }
  }
  if (alert.triage_status !== 'complete' || typeof alert.triage_confidence !== 'number') {
    return { eyebrow: 'Cảnh báo ưu tiên', tone: 'high' }
  }

  const confidence = alert.triage_confidence
  if (alert.triage_verdict === 'real') {
    if (confidence >= TRIAGE_HIGH) return { eyebrow: 'Cảnh báo khẩn cấp', tone: 'critical' }
    if (confidence >= TRIAGE_LOW) return { eyebrow: 'Cảnh báo ưu tiên', tone: 'high' }
    return { eyebrow: 'Cảnh báo — cần xác minh', tone: 'medium' }
  }
  if (alert.triage_verdict === 'false_positive') {
    // A false-positive confident enough to auto-suppress never reaches the modal (see
    // isAutoSuppressed and its configurable threshold), so this only ever describes one the
    // system was not confident enough to act on.
    return { eyebrow: 'Nghi báo động giả — ưu tiên thấp', tone: 'low' }
  }
  return { eyebrow: 'Chưa rõ — cần xác minh', tone: 'medium' }
}

/**
 * How many past incidents to offer, and how close they must be to be worth offering.
 *
 * Both scale with the VLM's confidence, for one reason: when the assessment is strong the
 * guard needs a quick corroboration and weak matches only muddy it, so we ask for fewer and
 * demand they be closer. When it is weak the guard is doing the real work of deciding, and
 * more context — even slightly looser context — is what helps.
 *
 * `limit` is a ceiling, never a quota. The floor is applied to whatever comes back, so a
 * camera with no close history returns fewer rows, or none. That is the honest answer, and
 * it is why the displayed real/false split is always a measurement of what was retrieved
 * rather than a ratio anyone chose.
 */
export function similarIncidentQuery(alert) {
  const confidence = typeof alert?.triage_confidence === 'number' ? alert.triage_confidence : null
  if (alert?.triage_status === 'complete' && confidence !== null) {
    if (confidence >= TRIAGE_HIGH) return { limit: 3, minSimilarity: 0.8 }
    if (confidence >= TRIAGE_LOW) return { limit: 5, minSimilarity: 0.7 }
  }
  return { limit: 5, minSimilarity: 0.65 }
}

/**
 * "just now" / "4m ago" — a wall-mounted display is read at a glance, and an absolute
 * "02:14:07" carries no date, so across a shift boundary it cannot be placed in time.
 * The absolute timestamp is still rendered alongside, as a <time> element.
 */
export function relativeTime(iso, now = Date.now()) {
  const seconds = Math.max(0, Math.round((now - new Date(iso).getTime()) / 1000))
  if (seconds < 10) return 'vừa xong'
  if (seconds < 60) return `${seconds} giây trước`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes} phút trước`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours} giờ trước`
  return `${Math.floor(hours / 24)} ngày trước`
}

export function formatClock(iso) {
  const date = new Date(iso)
  return `${date.getHours()}h${String(date.getMinutes()).padStart(2, '0')}`
}

/**
 * The confidence at or above which a false-positive verdict is auto-hidden from the guard.
 *
 * Unlike TRIAGE_HIGH/TRIAGE_LOW (which pick wording), this one governs whether a guard ever
 * sees the alert -- a confidently wrong false-positive here is a real incident dropped from
 * the queue silently -- so it *is* a setting: the backend serves it in `monitoring/status`
 * from VLM_AUTO_SUPPRESS_CONFIDENCE, and `setAutoSuppressConfidence` below is called once
 * when that loads. The literal is only the value used before the first status poll.
 *
 * A module-level setter rather than a prop because `requiresGuardReview` is passed around as
 * a bare predicate (`alerts.filter(requiresGuardReview)`).
 */
let autoSuppressConfidence = 0.9

export function setAutoSuppressConfidence(value) {
  if (typeof value === 'number' && value >= 0 && value <= 1) autoSuppressConfidence = value
}

/**
 * High-confidence false positives filtered by VLM.
 * They do not open the big modal and do not count as active attention backlog.
 */
export function isAutoSuppressed(alert) {
  return (
    alert.triage_status === 'complete' &&
    alert.triage_verdict === 'false_positive' &&
    typeof alert.triage_confidence === 'number' &&
    alert.triage_confidence >= autoSuppressConfidence
  )
}

/**
 * Alerts currently being verified by VLM.
 * Shown as a small corner "AI verifying" card while pending.
 */
export function isPendingTriage(alert) {
  return !alert.acknowledged && alert.triage_status === 'pending'
}

/**
 * Alerts requiring guard attention in the big popup modal.
 * Includes real, uncertain, skipped, failed, or low-confidence false positives.
 */
export function requiresGuardReview(alert) {
  if (alert.acknowledged) return false
  if (isPendingTriage(alert)) return false
  if (isAutoSuppressed(alert)) return false
  return true
}
