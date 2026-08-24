import { useEffect, useState } from 'react'
import { api } from '../api'
import { incidentMeta } from '../incidents'

function pct(value) {
  return value === null || value === undefined ? '—' : `${(value * 100).toFixed(1)}%`
}

function number(value) {
  return new Intl.NumberFormat().format(value ?? 0)
}

export default function StatsView({ status }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    let socket
    let reconnectTimer

    const connect = () => {
      if (cancelled) return
      socket = new WebSocket(api.managerLiveUrl())
      socket.onmessage = () => {
        // The event stream is the invalidation signal; the statistics endpoint remains the
        // single source of truth for the aggregate report and heat-map data.
        api.statistics().then((body) => !cancelled && setData(body)).catch(() => {})
      }
      socket.onclose = () => {
        if (!cancelled) reconnectTimer = setTimeout(connect, 3000)
      }
    }

    connect()
    return () => {
      cancelled = true
      clearTimeout(reconnectTimer)
      socket?.close()
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    api.statistics()
      .then((body) => !cancelled && setData(body))
      .catch((err) => !cancelled && setError(err.message))
    return () => { cancelled = true }
  }, [])

  if (error) {
    return <div className="stats-error" role="alert"><span className="status-mark danger">!</span><div><h4>Không thể tải thống kê</h4><p>{error}</p></div></div>
  }
  if (!data) return <div className="stats-loading">Đang chuẩn bị báo cáo…</div>

  const { report, operational, heatmap } = data
  const total = operational.total_alerts || 0
  const reviewed = operational.reviewed_alerts || 0
  const reviewRate = total ? reviewed / total : null
  const activeCameras = status?.active_cameras ?? null
  const totalCameras = status?.total_cameras ?? null
  const cameraRows = operational.by_camera || []
  const labelRows = report.by_label || []
  const days = operational.by_day || []

  return (
    <section className="manager-dashboard" aria-labelledby="manager-page-title">
      <header className="manager-hero">
        <div>
          <p className="eyebrow"><span className="live-dot" /> Thông tin vận hành</p>
          <h1 id="manager-page-title">Tổng quan quản lý</h1>
          <p className="hero-copy">Theo dõi chất lượng phát hiện, khối lượng xử lý và các điểm cần chú ý.</p>
        </div>
        <div className="hero-stamp">
          <span>Hoạt động gần nhất</span>
          <strong>{operational.last_alert ? new Date(operational.last_alert).toLocaleString('vi-VN') : 'Chưa có cảnh báo'}</strong>
        </div>
      </header>

      {report.warnings?.length > 0 && (
        <section className="manager-warning" role="note">
          <span className="status-mark warning">!</span>
          <div><strong>Lưu ý khi đọc mẫu dữ liệu</strong><p>{report.warnings[0]}</p></div>
          {report.warnings.length > 1 && <span className="warning-count">+{report.warnings.length - 1} cảnh báo</span>}
        </section>
      )}

      <section className="manager-kpis" aria-label="Executive metrics">
        <MetricCard label="Độ chính xác phát hiện" value={pct(report.rule_precision)} detail="Sự cố đã xem được phân loại đúng" tone="good" />
        <MetricCard label="Tỷ lệ báo động giả" value={pct(report.false_alarm_rate)} detail="Trên các cảnh báo đã gắn nhãn" tone={report.false_alarm_rate > 0.3 ? 'bad' : 'neutral'} />
        <MetricCard label="Mức độ xử lý" value={pct(reviewRate)} detail={`${number(reviewed)}/${number(total)} cảnh báo đã kết luận`} tone="neutral" />
        <MetricCard label="Hàng đợi trực tiếp" value={number(operational.live_unacknowledged)} detail="Cảnh báo đang chờ bảo vệ xử lý" tone={operational.live_unacknowledged ? 'bad' : 'good'} />
      </section>

      <section className="manager-primary-grid">
        <TrendChart days={days} />
        <ReviewPulse reviewed={reviewed} total={total} activeCameras={activeCameras} totalCameras={totalCameras} />
      </section>

      <section className="manager-secondary-grid">
        <IncidentMix rows={labelRows} />
        <CameraHotspots rows={cameraRows} />
      </section>

      <HeatMap points={heatmap?.points || []} />

      {report.comparisons?.length > 0 && <TriagePanel rows={report.comparisons} agreement={report.triage_agreement} coverage={report.triage_coverage} />}

      <footer className="manager-footer">
        <span>Khoảng dữ liệu</span>
        <strong>{operational.first_alert ? new Date(operational.first_alert).toLocaleDateString('vi-VN') : '—'} đến {operational.last_alert ? new Date(operational.last_alert).toLocaleDateString('vi-VN') : '—'}</strong>
        <span className="footer-separator" />
        <span>Người xử lý</span>
        <strong>{operational.reviewers?.length ? operational.reviewers.join(', ') : 'Chưa ghi nhận'}</strong>
        <span className="footer-separator" />
        <span>Độ bao phủ</span>
        <strong className="footer-muted">Không thể đo từ các cảnh báo đã hiển thị</strong>
      </footer>
    </section>
  )
}

function HeatMap({ points }) {
  const max = Math.max(...points.map((point) => point.total || 0), 1)
  return <article className="viz-panel heatmap-panel">
    <div className="panel-heading"><div><p className="eyebrow">Sơ đồ mô phỏng</p><h2>Bản đồ nhiệt sự cố</h2></div><span className="panel-stat">{points.length} camera</span></div>
    {points.length ? <div className="heatmap-canvas" role="img" aria-label="Simulated incident heat map">
      {points.map((point) => <span key={point.camera_id} className="heatmap-point" title={`${point.camera_name}: ${point.total} alerts`} style={{ left: `${point.x}%`, top: `${point.y}%`, '--intensity': point.total / max }}><b>{point.total}</b></span>)}
    </div> : <EmptyState text="Chưa ghi nhận vị trí sự cố" />}
  </article>
}

function MetricCard({ label, value, detail, tone }) {
  return <article className={`manager-metric ${tone}`}>
    <span className="metric-label">{label}</span>
    <strong className="metric-value">{value}</strong>
    <span className="metric-detail">{detail}</span>
  </article>
}

function TrendChart({ days }) {
  const values = days.map((day) => day.total || 0)
  const peak = Math.max(...values, 1)
  const width = 680
  const height = 220
  const padX = 28
  const padY = 24
  const points = values.map((value, index) => {
    const x = values.length <= 1 ? width / 2 : padX + (index / (values.length - 1)) * (width - padX * 2)
    const y = height - padY - (value / peak) * (height - padY * 2)
    return [x, y]
  })
  const line = points.map(([x, y]) => `${x},${y}`).join(' ')
  const area = points.length ? `${padX},${height - padY} ${line} ${width - padX},${height - padY}` : ''

  return <article className="viz-panel trend-panel">
    <div className="panel-heading"><div><p className="eyebrow">Tín hiệu theo thời gian</p><h2>Hoạt động cảnh báo</h2></div><span className="panel-stat">Tổng {number(values.reduce((sum, value) => sum + value, 0))}</span></div>
    {points.length ? <div className="trend-chart"><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Daily alert activity trend">
      {[0.25, 0.5, 0.75].map((fraction) => <line key={fraction} x1={padX} x2={width - padX} y1={height - padY - fraction * (height - padY * 2)} y2={height - padY - fraction * (height - padY * 2)} className="chart-gridline" />)}
      <polygon points={area} className="chart-area" />
      <polyline points={line} className="chart-line" />
      {points.map(([x, y], index) => <circle key={index} cx={x} cy={y} r="4" className="chart-point"><title>{days[index].date}: {days[index].total} alerts</title></circle>)}
    </svg><div className="chart-labels">{days.map((day) => <span key={day.date}>{day.date}</span>)}</div></div> : <EmptyState text="Không có cảnh báo trong khoảng này" />}
  </article>
}

function ReviewPulse({ reviewed, total, activeCameras, totalCameras }) {
  const ratio = total ? Math.round((reviewed / total) * 100) : 0
  return <article className="viz-panel pulse-panel">
    <div className="panel-heading"><div><p className="eyebrow">Mức độ sẵn sàng</p><h2>Tiến độ xử lý</h2></div><span className="panel-icon">↗</span></div>
    <div className="pulse-main"><div className="donut" style={{ '--progress': `${ratio * 3.6}deg` }}><div><strong>{ratio}%</strong><span>đã xem</span></div></div><div className="pulse-copy"><strong>{reviewed ? `Đã ghi nhận ${number(reviewed)} kết luận` : 'Chưa có kết luận'}</strong><span>{total ? `Còn ${number(total - reviewed)} cảnh báo cần xử lý` : 'Hàng đợi đã trống'}</span></div></div>
    <div className="pulse-foot"><span><i className="pulse-key green" />Camera trực tiếp</span><strong>{activeCameras !== null && totalCameras !== null ? `${activeCameras}/${totalCameras}` : '—'}</strong></div>
  </article>
}

function IncidentMix({ rows }) {
  const max = Math.max(...rows.map((row) => row.total || row.reviewed || 0), 1)
  return <article className="viz-panel mix-panel"><div className="panel-heading"><div><p className="eyebrow">Chất lượng phân loại</p><h2>Cơ cấu sự cố</h2></div><span className="panel-stat">{rows.length} loại</span></div>{rows.length ? <div className="mix-list">{rows.map((row) => { const total = row.total || row.reviewed || 0; const precision = row.precision; const meta = incidentMeta(row.label); return <div className="mix-row" key={row.label}><div className="mix-row-top"><strong>{meta.short}</strong><span>{number(total)} đã xem · chính xác {pct(precision)}</span></div><div className="mix-track"><span style={{ width: `${(total / max) * 100}%` }} /><b style={{ left: `${Math.min(100, (precision || 0) * 100)}%` }} /></div></div>})}</div> : <EmptyState text="Chưa có sự cố được xử lý" />}</article>
}

function CameraHotspots({ rows }) {
  const max = Math.max(...rows.map((row) => row.total || 0), 1)
  return <article className="viz-panel camera-panel"><div className="panel-heading"><div><p className="eyebrow">Điểm cần chú ý</p><h2>Camera có nhiều sự cố</h2></div><span className="panel-stat">{rows.length} camera</span></div>{rows.length ? <div className="hotspot-list">{rows.slice().sort((a, b) => b.total - a.total).map((row) => <div className="hotspot-row" key={row.camera_id}><div className="hotspot-label"><strong>{row.camera_name}</strong><span>{number(row.total)} cảnh báo · {row.total ? Math.round((row.reviewed / row.total) * 100) : 0}% đã xem</span></div><div className="hotspot-track"><span style={{ width: `${(row.total / max) * 100}%` }} /></div></div>)}</div> : <EmptyState text="Chưa ghi nhận cảnh báo camera" />}</article>
}

function TriagePanel({ rows, agreement, coverage }) {
  return <section className="viz-panel triage-panel"><div className="panel-heading"><div><p className="eyebrow">Decision support</p><h2>VLM policy replay</h2><p className="panel-description">Historical replay only. It shows how automated review would have changed surfaced alerts.</p></div><div className="triage-summary"><strong>{pct(agreement)}</strong><span>agreement · {number(coverage)} cases</span></div></div><div className="policy-grid">{rows.map((row) => { const missed = row.real_incidents_suppressed > 0; const delta = (row.precision - row.baseline_precision) * 100; return <div className={`policy-row ${missed ? 'risk' : ''}`} key={row.policy}><strong>{row.policy}</strong><span>{pct(row.baseline_precision)} <i>→</i> {pct(row.precision)}</span><span className={delta >= 0 ? 'delta positive' : 'delta'}>{delta >= 0 ? '+' : ''}{delta.toFixed(1)} pts</span><span className="policy-result">{missed ? `${row.real_incidents_suppressed} real missed` : `${row.false_alarms_suppressed} false alarms filtered`}</span></div>})}</div></section>
}

function EmptyState({ text }) { return <div className="manager-empty">{text}</div> }
