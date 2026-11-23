export type CameraState = 'idle' | 'connecting' | 'running' | 'reconnecting'

export interface Status {
  state: CameraState
  url: string | null
  fps: number
  error: string | null
  session: number
}

export interface AlarmItem {
  id: number
  kind: string
  message: string
  severity: 'CRITICAL' | 'HIGH' | 'MEDIUM' | string
  confidence: number
  timestamp: number
  time: string
  snapshot: string | null
}
