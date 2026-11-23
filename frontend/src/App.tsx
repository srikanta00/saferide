import AlarmPanel from './components/AlarmPanel'
import CameraForm from './components/CameraForm'
import VideoPanel, { StatusBadge } from './components/VideoPanel'
import { useAlarms, useStatus } from './hooks'

export default function App() {
  const { status, setStatus, unreachable } = useStatus()
  const { alarms, live, clear } = useAlarms()

  return (
    <div className="app">
      <header className="topbar">
        <h1>SafeRide</h1>
        <StatusBadge status={status} unreachable={unreachable} />
      </header>
      <CameraForm status={status} onChange={setStatus} />
      <main>
        <VideoPanel status={status} />
        <AlarmPanel alarms={alarms} live={live} onClear={clear} />
      </main>
    </div>
  )
}
