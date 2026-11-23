import type { Status } from '../types'

const LABELS: Record<string, string> = {
  idle: 'No camera',
  connecting: 'Connecting…',
  running: 'Live',
  reconnecting: 'Connection lost, retrying…',
}

export function StatusBadge({ status, unreachable }: { status: Status | null; unreachable: boolean }) {
  if (unreachable) return <span className="badge bad">Server unreachable</span>
  if (!status) return <span className="badge">…</span>
  const cls = status.state === 'running' ? 'good' : status.state === 'idle' ? '' : 'warn'
  return (
    <span className={`badge ${cls}`}>
      {LABELS[status.state] ?? status.state}
      {status.state === 'running' && ` · ${status.fps} analysed fps`}
    </span>
  )
}

export default function VideoPanel({ status }: { status: Status | null }) {
  const state = status?.state ?? 'idle'
  return (
    <div className="video">
      {state === 'idle' ? (
        <div className="placeholder">Enter an RTSP URL above and press Start</div>
      ) : (
        <>
          {/* key restarts the MJPEG connection whenever a new camera session begins */}
          <img key={status?.session} src="/stream.mjpg" alt="Live camera view" />
          {state !== 'running' && (
            <div className="overlay">{state === 'connecting' ? 'Connecting to camera…' : 'Connection lost, retrying…'}</div>
          )}
        </>
      )}
      {status?.url && <div className="video-url">{status.url}</div>}
      {status?.error && <div className="video-error">{status.error}</div>}
    </div>
  )
}
