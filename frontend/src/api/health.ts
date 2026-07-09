import { apiClient } from './client'

export interface HealthStatus {
  status: string
}

export function getLiveness(): Promise<HealthStatus> {
  return apiClient.get<HealthStatus>('/api/v1/health')
}
