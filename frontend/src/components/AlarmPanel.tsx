import type { AlarmItem } from '../types'

interface Props {
  alarms: AlarmItem[]
  live: boolean
  onClear: () => void
}

function AlarmRow({ alarm }: { alarm: AlarmItem }) {
  const sev = alarm.severity.toLowerCase()
  return (
    <li className={`alarm sev-${sev}`}>
      <div className="alarm-head">
        <span className="alarm-kind">{alarm.kind.replace(':', ' · ').replace(/_/g, ' ')}</span>
        <span className={`sev-tag sev-${sev}`}>{alarm.severity}</span>
      </div>
      <div className="alarm-msg">{alarm.message}</div>
      <div className="alarm-meta">
        {new Date(alarm.timestamp * 1000).toLocaleString()} · confidence {Math.round(alarm.confidence * 100)}%
      </div>
      {alarm.snapshot && (
        <a href={alarm.snapshot} target="_blank" rel="noreferrer">
          <img className="thumb" src={alarm.snapshot} alt={`Snapshot for ${alarm.kind}`} loading="lazy" />
        </a>
      )}
    </li>
  )
}

export default function AlarmPanel({ alarms, live, onClear }: Props) {
  return (
    <aside className="alarms">
      <header>
        <h2>
          Alarms <span className="count">{alarms.length}</span>
        </h2>
        <span className={`dot ${live ? 'on' : 'off'}`} title={live ? 'Receiving live updates' : 'Reconnecting…'} />
        <button type="button" onClick={onClear} disabled={!alarms.length}>
          Clear
        </button>
      </header>
      {alarms.length === 0 ? (
        <p className="empty">No alarms yet.</p>
      ) : (
        <ul>
          {alarms.map((a) => (
            <AlarmRow key={a.id} alarm={a} />
          ))}
        </ul>
      )}
    </aside>
  )
}
