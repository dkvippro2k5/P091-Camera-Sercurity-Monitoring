/**
 * The zone arming window (PRD §3.1's "khung giờ cấm"), as the setup screen reasons about it.
 *
 * Mirrors `src/vision/zone_config.py:is_zone_active` — same wire format 'HH:MM-HH:MM', same
 * midnight-wrap rule. Kept as its own module rather than inline in the form because the
 * wrap case is the part that is easy to get subtly wrong, and because the *reason* it
 * exists is a UX failure worth naming: the PRD's own example window is 22:00-05:00, so an
 * operator configuring a camera during the working day sets a window that is correct and
 * then sees nothing happen. Without the "not armed right now" note below, the only
 * available conclusion is that intrusion detection is broken.
 */

/** Minutes since midnight for 'HH:MM'. NaN when the field is empty or half-typed. */
function minutes(hhmm) {
  const [h, m] = String(hhmm).split(':')
  return Number(h) * 60 + Number(m)
}

/** True when `now` falls inside [start, end], treating end < start as wrapping midnight. */
export function isWithinWindow(start, end, now = new Date()) {
  const from = minutes(start)
  const to = minutes(end)
  if (Number.isNaN(from) || Number.isNaN(to)) return true

  const current = now.getHours() * 60 + now.getMinutes()
  // Inclusive on both ends, matching the Python side: `start <= current <= end`.
  return from <= to ? current >= from && current <= to : current >= from || current <= to
}

/** The wire value the backend stores, or null for an always-armed zone. */
export function toWindowString(enabled, start, end) {
  if (!enabled) return null
  if (!start || !end) return null
  return `${start}-${end}`
}

/**
 * One sentence telling the operator what this window means *right now*.
 *
 * Deliberately states the consequence ("will not alert") rather than just the state
 * ("inactive"), because the operator is about to walk away from this screen believing
 * the camera is watching.
 */
export function describeWindow(start, end, now = new Date()) {
  if (!start || !end) return { armed: true, text: 'Pick a start and end time.' }
  if (start === end) {
    return {
      armed: false,
      text: 'Start and end are the same time — pick a different end time, or switch to always armed.',
    }
  }

  const wraps = minutes(end) < minutes(start)
  const span = wraps ? `${start} until ${end} the next morning` : `${start} until ${end}`

  if (isWithinWindow(start, end, now)) {
    return { armed: true, text: `Armed ${span}. That window is active right now.` }
  }
  return {
    armed: false,
    text: `Armed ${span}. It is outside that window now, so this zone will not alert until ${start}.`,
  }
}
