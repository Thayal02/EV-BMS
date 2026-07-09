import { BackendStatusBadge } from '../components/BackendStatusBadge'
import { useBackendStatus } from '../hooks/useBackendStatus'

export function DashboardPage() {
  const backendStatus = useBackendStatus()

  return (
    <div className="min-h-screen bg-slate-950">
      <header className="border-b border-slate-800 px-6 py-4">
        <div className="mx-auto flex max-w-6xl items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold text-slate-100">
              Agentic AI Battery Management System
            </h1>
            <p className="text-sm text-slate-400">
              SOH · RUL · Failure Risk decision support for EV battery packs
            </p>
          </div>
          <BackendStatusBadge state={backendStatus} />
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-10">
        <div className="rounded-lg border border-dashed border-slate-800 p-8 text-center text-slate-400">
          Upload a battery dataset to begin analysis. Dataset upload, SOH/RUL/
          failure prediction, explainability, reports, and the AI battery
          expert chat will appear here as each feature is built.
        </div>
      </main>
    </div>
  )
}
