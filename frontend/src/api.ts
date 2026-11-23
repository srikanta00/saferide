import type { AlarmItem, Status } from './types'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init)
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      /* keep the HTTP status text */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export const getStatus = () => request<Status>('/api/status')

export const startCamera = (url: string) =>
  request<Status>('/api/camera', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
  })

export const stopCamera = () => request<Status>('/api/camera', { method: 'DELETE' })

export const getAlarms = () => request<AlarmItem[]>('/api/alarms?limit=200')

export const clearAlarms = () => request<{ ok: boolean }>('/api/alarms', { method: 'DELETE' })
