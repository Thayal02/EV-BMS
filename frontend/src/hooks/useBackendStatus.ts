import { useEffect, useState } from 'react'
import { getLiveness } from '../api/health'

export type BackendConnectionState = 'checking' | 'online' | 'offline'

export function useBackendStatus(): BackendConnectionState {
  const [state, setState] = useState<BackendConnectionState>('checking')

  useEffect(() => {
    let cancelled = false

    getLiveness()
      .then(() => {
        if (!cancelled) setState('online')
      })
      .catch(() => {
        if (!cancelled) setState('offline')
      })

    return () => {
      cancelled = true
    }
  }, [])

  return state
}
