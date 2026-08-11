import { useState, useEffect } from 'react'
import { api } from '../api'
import { incidentMeta } from '../incidents'

/**
 * The manager's numbers: PRD §4 accuracy on the left, operational volume on the right.
 *
 * No charting library — the two views that matter here are a small table and a bar per day,
 * both of which are a `<div>` with a width. Adding a dependency for that would triple the
 * frontend's dependency count for one screen.
 *
 * Two rules the rest of the codebase is emphatic about and this screen must honour:
 *
 * - **`precision` is `float | None`, and `None` is not `0.0`.** `eval_report._precision`
 *   returns `None` when nothing has been reviewed specifically so that "nobody has looked
 *   at this camera yet" cannot be read as "every alert from it was wrong". Rendering it as
 *   `0%` would undo that at the last step, so `pct()` prints `—`.
 * - **Colour is never the only encoding.** Every value is labelled in text; the severity
 *   hues are decoration on top of a name that is already there.
 */
function pct(value) {
  return value === null || value === undefined ? '—' : `${(value * 100).toFixed(1)}%`
}

export default function StatsView() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    api
      .statistics()
      .then((body) => !cancelled && setData(body))
      .catch((err) => !cancelled && setError(err.message))
    return () => {
      cancelled = true
    }
  }, [])

  if (error) {
    return (
      <p className="err" role="alert">
        <span aria-hidden="true">▲</span>
        {error}
      </p>
    )
  }
  if (!data) return <p className="hint">Loading statistics…</p>

  const { report, operational } = data

  return (
    <div className="stats">
      <div className="stage-head">
        <h2>Statistics</h2>
        <span className="count">
          {operational.total_alerts} alert{operational.total_alerts === 1 ? '' : 's'} recorded
        </span>
      </div>

      {/* Caveats first, not last. They are a first-class `warnings` list on the report for
          exactly this reason: a figure whose caveat sits below the fold gets quoted without
          it, and every number on this page is small-sample early on. */}
      {report.warnings?.length > 0 && (
        <section className="stats-warnings" role="note">
          <h3>Read these first</h3>
          <ul>
            {report.warnings.map((warning, i) => (
              <li key={i}>{warning}</li>
            ))}
          </ul>
        </section>
      )}

      <section className="stats-block">
        <h3>Right now</h3>
        <div className="stat-tiles">
          <Tile value={operational.live_unacknowledged} label="Awaiting a guard" attention={operational.live_unacknowledged > 0} />
          <Tile value={operational.total_alerts} label="Alerts all time" />
          <Tile value={operational.reviewed_alerts} label="Judged by a guard" />
          <Tile value={operational.unreviewed_alerts} label="Never judged" />
        </div>
        <p className="hint">
          &ldquo;Awaiting a guard&rdquo; counts only alerts still held in memory — the shift&apos;s live backlog.
          &ldquo;Never judged&rdquo; counts the whole log, including alerts too old for anyone to action now.
        </p>
      </section>

      <section className="stats-block">
        <h3>Detection accuracy (PRD §4)</h3>
        <div className="stat-tiles">
          <Tile value={pct(report.rule_precision)} label="Rule precision" />
          <Tile value={pct(report.false_alarm_rate)} label="False-alarm rate" />
          <Tile value={report.reviewed_rows} label="Reviewed alerts" />
          <Tile value={report.cameras} label="Cameras reviewed" />
        </div>

        {/* The report refuses to print recall and says why, in its own words. Reproduced
            verbatim rather than paraphrased — the refusal is the finding. */}
        <p className="stats-note">
          <strong>Recall: not computable.</strong> {report.recall_note}
        </p>

        {report.by_label?.length > 0 && (
          <table className="stats-table">
            <caption className="visually-hidden">Precision by incident label</caption>
            <thead>
              <tr>
                <th scope="col">Incident</th>
                <th scope="col">Reviewed</th>
                <th scope="col">Real</th>
                <th scope="col">False alarms</th>
                <th scope="col">Precision</th>
              </tr>
            </thead>
            <tbody>
              {report.by_label.map((row) => (
                <tr key={row.label}>
                  <th scope="row">
                    {incidentMeta(row.label).short}
                    {/* Stated in words, not implied by a faint colour. A precision figure
                        from four reviews is not a weaker number, it is a different kind of
                        claim, and the reader has to be told which one they are looking at. */}
                    {row.low_sample && <span className="hint"> too few to trust</span>}
                  </th>
                  <td>{row.reviewed}</td>
                  <td>{row.real}</td>
                  <td>{row.false_alarms}</td>
                  <td>{pct(row.precision)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      {report.comparisons?.length > 0 && (
        <section className="stats-block">
          <h3>What AI triage would have done</h3>
          <p className="hint">
            Nothing was actually suppressed — every alert reached a guard. This replays each policy over alerts
            whose outcome is already known, which is the only way to price the trade-off before committing to it.
            Agreement with guards: {pct(report.triage_agreement)} over {report.triage_coverage} triaged alert
            {report.triage_coverage === 1 ? '' : 's'}.
          </p>

          <table className="stats-table">
            <caption className="visually-hidden">Rules alone versus rules plus VLM triage</caption>
            <thead>
              <tr>
                <th scope="col">Policy</th>
                <th scope="col">Precision before</th>
                <th scope="col">Precision after</th>
                <th scope="col">False alarms hidden</th>
                <th scope="col">Real incidents hidden</th>
              </tr>
            </thead>
            <tbody>
              {report.comparisons.map((row) => (
                <tr key={row.policy}>
                  <th scope="row">{row.policy}</th>
                  <td>{pct(row.baseline_precision)}</td>
                  <td>{pct(row.precision)}</td>
                  <td>{row.false_alarms_suppressed}</td>
                  {/* The number that decides whether triage should ever gate the dashboard.
                      Any value above zero is a medical emergency the system would have
                      swallowed, so it is called out rather than left to blend in. */}
                  <td className={row.real_incidents_suppressed > 0 ? 'stat-bad' : undefined}>
                    {row.real_incidents_suppressed}
                    {row.real_incidents_suppressed > 0 && ' — would have been missed'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      <section className="stats-block">
        <h3>Alerts by camera</h3>
        {operational.by_camera.length === 0 ? (
          <p className="hint">No alerts recorded yet.</p>
        ) : (
          <table className="stats-table">
            <thead>
              <tr>
                <th scope="col">Camera</th>
                <th scope="col">Alerts</th>
                <th scope="col">Reviewed</th>
                <th scope="col">Unreviewed</th>
              </tr>
            </thead>
            <tbody>
              {operational.by_camera.map((row) => (
                <tr key={row.camera_id}>
                  <th scope="row">{row.camera_name}</th>
                  <td>{row.total}</td>
                  <td>{row.reviewed}</td>
                  <td>{row.total - row.reviewed}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      {operational.by_day.length > 0 && <ByDay days={operational.by_day} />}

      <section className="stats-block">
        <h3>Coverage</h3>
        <dl className="stats-meta">
          <dt>Incident types seen</dt>
          <dd>
            {operational.by_label.length === 0
              ? '—'
              : operational.by_label.map((row) => `${incidentMeta(row.label).short} (${row.total})`).join(', ')}
          </dd>
          <dt>Guards who have judged alerts</dt>
          <dd>{operational.reviewers.length ? operational.reviewers.join(', ') : '—'}</dd>
          <dt>First alert</dt>
          <dd>{operational.first_alert ? new Date(operational.first_alert).toLocaleString() : '—'}</dd>
          <dt>Most recent alert</dt>
          <dd>{operational.last_alert ? new Date(operational.last_alert).toLocaleString() : '—'}</dd>
        </dl>
      </section>
    </div>
  )
}

function Tile({ value, label, attention }) {
  return (
    <div className={attention ? 'vital attention' : 'vital'}>
      <span className="vital-value">{value}</span>
      <span className="vital-label">{label}</span>
    </div>
  )
}

/**
 * Daily alert volume as plain bars. Each carries its own count in text beside it, so the
 * bar is redundant reinforcement rather than the only way to read the value.
 */
function ByDay({ days }) {
  const peak = Math.max(...days.map((d) => d.total), 1)
  return (
    <section className="stats-block">
      <h3>Alerts per day</h3>
      <ul className="bars">
        {days.map((day) => (
          <li key={day.date}>
            <span className="bar-label">{day.date}</span>
            <span className="bar-track">
              <span className="bar-fill" style={{ width: `${(day.total / peak) * 100}%` }} />
            </span>
            <span className="bar-value">{day.total}</span>
          </li>
        ))}
      </ul>
    </section>
  )
}
