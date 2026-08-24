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
 * High-confidence false positives filtered by VLM.
 * They do not open the big modal and do not count as active attention backlog.
 */
export function isAutoSuppressed(alert) {
  return (
    alert.triage_status === 'complete' &&
    alert.triage_verdict === 'false_positive' &&
    typeof alert.triage_confidence === 'number' &&
    alert.triage_confidence >= 0.85
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
