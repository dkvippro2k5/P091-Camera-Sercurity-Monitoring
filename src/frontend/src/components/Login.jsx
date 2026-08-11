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
      <form className="login-card" onSubmit={handleSubmit}>
        <div className="brand login-brand">
          <span aria-hidden="true">◉</span>
          <span>Camera Monitor</span>
        </div>

        <h1>Sign in</h1>
        <p className="hint">
          Guards monitor the live grid, judge incidents, and configure the cameras. Managers review
          statistics.
        </p>

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

        {/* There is no signup route by design — accounts are seeded by an administrator. A
            login screen with no way forward and no explanation reads as broken. */}
        <p className="hint">Accounts are created by an administrator with <code>scripts/seed_users.py</code>.</p>
      </form>
    </div>
  )
}
