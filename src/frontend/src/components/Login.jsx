import { useState } from 'react'

/**
 * The sign-in screen. Deliberately the whole viewport, not a modal over the grid: there is
 * nothing behind it to return to, and a dimmed live feed under a login box would show an
 * unauthenticated operator exactly the footage the login exists to protect.
 *
 * Reuses the setup form's CSS (`.field`, `.err`, `.actions`) so it inherits the same 44px
 * targets and 16px inputs rather than growing a second, less careful set.
 */
export default function Login({ onLogin }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  async function handleSubmit(event) {
    event.preventDefault()
    setError(null)
    setBusy(true)
    try {
      await onLogin(username.trim(), password)
    } catch (err) {
      setError(err.message)
      // Only the password is cleared. Retyping a correct username after a typo in the
      // password is pure friction, and on a control-room keyboard it is friction during
      // an incident.
      setPassword('')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="login-page">
      <div className="login-shell">
        <div className="login-visual" aria-hidden="true">
          <div className="login-radar"><span /><span /><span /></div>
          <div className="login-visual-label">P-091 · Vision operations</div>
        </div>

        <form className="login-card" onSubmit={handleSubmit}>
          <div className="brand login-brand">
            <span className="brand-mark" aria-hidden="true">CM</span>
            <span className="brand-copy"><strong>Camera Monitor</strong><small>Secure workspace</small></span>
          </div>

          <div className="login-heading">
            <p className="eyebrow">Authorized access</p>
            <h1>Sign in to continue</h1>
          </div>

          <label className="field">
            <span>Username</span>
            <input
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="username"
              autoFocus
              required
            />
          </label>

          <label className="field">
            <span>Password</span>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
              required
            />
          </label>

        {/* role="alert" so a screen reader announces the failure instead of leaving the
            operator pressing a button that appears to do nothing. */}
          {error && (
            <p className="err" role="alert">
              <span aria-hidden="true">▲</span>
              {error}
            </p>
          )}

          <div className="actions">
            <button type="submit" className="primary" disabled={busy || !username.trim() || !password}>
              {busy ? 'Signing in…' : 'Sign in'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
