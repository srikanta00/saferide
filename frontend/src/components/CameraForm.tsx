import { useState } from 'react'
import { startCamera, stopCamera } from '../api'
import type { Status } from '../types'

interface Props {
  status: Status | null
  onChange: (s: Status) => void
}

export default function CameraForm({ status, onChange }: Props) {
  const [url, setUrl] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const active = !!status && status.state !== 'idle'

  async function run(action: () => Promise<Status>) {
    setBusy(true)
    setError(null)
    try {
      onChange(await action())
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <form
      className="camera-form"
      onSubmit={(e) => {
        e.preventDefault()
        run(() => startCamera(url))
      }}
    >
      <input
        type="text"
        value={url}
        onChange={(e) => setUrl(e.target.value)}
        placeholder="rtsp://user:pass@192.168.1.10:554/stream1"
        aria-label="Camera RTSP URL"
        autoComplete="off"
        spellCheck={false}
      />
      <button type="submit" className="primary" disabled={busy || !url.trim()}>
        {active ? 'Switch camera' : 'Start'}
      </button>
      {active && (
        <button type="button" disabled={busy} onClick={() => run(stopCamera)}>
          Stop
        </button>
      )}
      {error && (
        <p className="form-error" role="alert">
          {error}
        </p>
      )}
    </form>
  )
}
