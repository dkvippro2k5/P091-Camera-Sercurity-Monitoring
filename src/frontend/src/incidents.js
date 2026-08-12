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
    short: 'Fall',
    headline: 'Fall detected',
    rank: 0,
    severity: 'critical',
    // Shown under the headline so the operator knows what the alert is asking of them.
    // The label alone doesn't say "go now" vs "review later".
    action: 'Possible medical emergency — check on the person now.',
  },
  intrusion: {
    short: 'Intrusion',
    headline: 'Intrusion',
    rank: 1,
    severity: 'high',
    action: 'Someone entered a restricted zone.',
  },
  abandoned_object: {
    short: 'Abandoned object',
    headline: 'Abandoned object',
    rank: 2,
    severity: 'medium',
    action: 'An item was left unattended. Inspect when free.',
  },
}

const FALLBACK = { short: 'Incident', headline: 'Incident', rank: 9, severity: 'medium', action: '' }

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
  real: { text: 'Appears to be a real incident', tone: 'triage-real' },
  false_positive: { text: 'Looks like a false alarm', tone: 'triage-false' },
  uncertain: { text: 'Inconclusive', tone: 'triage-uncertain' },
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
  if (seconds < 10) return 'just now'
  if (seconds < 60) return `${seconds}s ago`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  return `${Math.floor(hours / 24)}d ago`
}

export function formatClock(iso) {
  return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}
