import type { BackendConnectionState } from '../hooks/useBackendStatus'

const STATE_STYLES: Record<BackendConnectionState, { dot: string; label: string }> = {
  checking: { dot: 'bg-amber-400 animate-pulse', label: 'Checking backend...' },
  online: { dot: 'bg-emerald-400', label: 'Backend connected' },
  offline: { dot: 'bg-red-500', label: 'Backend unreachable' },
}

export function BackendStatusBadge({ state }: { state: BackendConnectionState }) {
  const { dot, label } = STATE_STYLES[state]

  return (
    <div className="flex items-center gap-2 rounded-full border border-slate-800 bg-slate-900 px-3 py-1 text-sm text-slate-300">
      <span className={`h-2 w-2 rounded-full ${dot}`} aria-hidden="true" />
      <span>{label}</span>
    </div>
  )
}
