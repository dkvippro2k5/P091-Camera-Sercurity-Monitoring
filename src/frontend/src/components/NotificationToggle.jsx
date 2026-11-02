import { useEffect, useState } from 'react'
import { api } from '../api'

export default function NotificationToggle() {
  const [settings, setSettings] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    api.notificationSettings().then(setSettings).catch(() => setError('Không tải được cấu hình thông báo'))
  }, [])

  async function toggle() {
    setBusy(true)
    setError('')
    try {
      setSettings(await api.setNotificationEnabled(!settings.enabled))
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  if (!settings) return <section className="notification-card" aria-busy="true">Đang tải thông báo…</section>
  return (
    <section className={`notification-card ${settings.enabled ? 'is-on' : ''}`}>
      <div className="notification-copy">
        <span className="notification-icon" aria-hidden="true">✈</span>
        <div><h2>Thông báo Telegram</h2><p>{settings.configured ? 'Chỉ gửi cảnh báo VLM xác nhận là sự kiện thật.' : 'Chưa cấu hình bot Telegram.'}</p></div>
      </div>
      <button className="notification-switch" role="switch" aria-checked={settings.enabled} disabled={!settings.configured || busy} onClick={toggle}>
        <span>{settings.enabled ? 'Đang bật' : 'Đang tắt'}</span><i aria-hidden="true" />
      </button>
      {error && <small className="notification-error">{error}</small>}
    </section>
  )
}
