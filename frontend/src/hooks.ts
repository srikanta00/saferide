import { useCallback, useEffect, useState } from 'react'
import { clearAlarms, getAlarms, getStatus } from './api'
import type { AlarmItem, Status } from './types'

const MAX_ALARMS = 200

/** Polls the backend for camera status. `refresh` lets callers update it right after an action. */
export function useStatus(intervalMs = 2000) {
  const [status, setStatus] = useState<Status | null>(null)
  const [unreachable, setUnreachable] = useState(false)

  const refresh = useCallback(async () => {
    try {
      setStatus(await getStatus())
      setUnreachable(false)
    } catch {
      setUnreachable(true)
    }
  }, [])

  useEffect(() => {
    refresh()
    const t = setInterval(refresh, intervalMs)
    return () => clearInterval(t)
  }, [refresh, intervalMs])

  return { status, setStatus, unreachable, refresh }
}

/** Alarm list: loaded once, then kept live through Server-Sent Events. */
export function useAlarms() {
  const [alarms, setAlarms] = useState<AlarmItem[]>([])
  const [live, setLive] = useState(false)

  useEffect(() => {
    let cancelled = false
    const load = () =>
      getAlarms()
        .then((items) => !cancelled && setAlarms(items))
        .catch(() => undefined)

    const es = new EventSource('/api/events')
    es.onopen = () => {
      setLive(true)
      load() // also covers alarms missed while disconnected
    }
    es.onerror = () => setLive(false)
    es.addEventListener('alarm', (e) => {
      const item = JSON.parse((e as MessageEvent).data) as AlarmItem
      setAlarms((prev) =>
        prev.some((a) => a.id === item.id) ? prev : [item, ...prev].slice(0, MAX_ALARMS),
      )
    })
    return () => {
      cancelled = true
      es.close()
    }
  }, [])

  const clear = useCallback(async () => {
    await clearAlarms()
    setAlarms([])
  }, [])

  return { alarms, live, clear }
}
