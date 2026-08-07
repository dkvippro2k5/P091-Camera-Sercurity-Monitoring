import { useEffect, useRef } from 'react'

const FOCUSABLE = 'button:not([disabled]), [href], input:not([disabled]), select, textarea, [tabindex]:not([tabindex="-1"])'

/**
 * Confine keyboard focus to a dialog while it is open, and restore it afterwards.
 *
 * The alert dialog previously declared `role="alertdialog"` and `aria-modal="true"` but
 * never moved focus into itself. That combination is worse than no ARIA at all: a screen
 * reader announces a modal dialog while the user's focus is still on the page behind it,
 * Tab walks through the camera grid they cannot see, and Enter can fire "remove camera"
 * on a control the overlay is covering.
 *
 * Escape is deliberately NOT wired to close. Dismissing an alert POSTs an
 * acknowledgement that is recorded against the incident, so it has to be an intentional
 * act — a reflexive Escape must not clear a fall alert. WCAG 2.1.2 (No Keyboard Trap) is
 * still satisfied because the Acknowledge button is inside the trap and reachable by Tab,
 * which is the documented way out.
 */
export function useFocusTrap(isOpen) {
  const containerRef = useRef(null)
  const restoreToRef = useRef(null)

  useEffect(() => {
    if (!isOpen) return
    const container = containerRef.current
    if (!container) return

    // Remember where focus came from so it can go back — otherwise acknowledging an
    // alert drops the user at the top of the document and they lose their place.
    restoreToRef.current = document.activeElement

    const focusables = () => Array.from(container.querySelectorAll(FOCUSABLE))
    const initial = container.querySelector('[data-autofocus]') ?? focusables()[0]
    initial?.focus()

    function onKeyDown(event) {
      if (event.key !== 'Tab') return
      const items = focusables()
      if (items.length === 0) {
        event.preventDefault()
        return
      }
      const first = items[0]
      const last = items[items.length - 1]
      // Wrap in both directions. Without the shift-Tab branch, focus escapes backwards
      // out of the dialog on the very first keypress.
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }

    document.addEventListener('keydown', onKeyDown, true)
    // The page behind a modal must not scroll under it.
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'

    return () => {
      document.removeEventListener('keydown', onKeyDown, true)
      document.body.style.overflow = previousOverflow
      // Guard against the element having been unmounted while the dialog was open.
      const restoreTo = restoreToRef.current
      if (restoreTo instanceof HTMLElement && document.contains(restoreTo)) restoreTo.focus()
    }
  }, [isOpen])

  return containerRef
}
