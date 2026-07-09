/**
 * TypeScript mirror of shared/schemas/battery_metadata.schema.json.
 * Keep in sync with that file and backend/app/schemas/battery.py manually -
 * see shared/README.md.
 */

export type Chemistry = 'NMC' | 'NCA' | 'LFP' | 'LMO' | 'LTO' | 'Other'

export type FormFactor = 'cylindrical' | 'pouch' | 'prismatic'

export interface TemperatureRange {
  min: number
  max: number
}

export interface BatteryMetadata {
  battery_id: string
  manufacturer?: string | null
  model_name?: string | null
  /**
   * Rated pack energy capacity in kWh. Intentionally unconstrained - any
   * pack size is accepted, not just the reference sizes used during model
   * development.
   */
  nominal_capacity_kwh: number
  nominal_voltage_v: number
  chemistry: Chemistry
  pack_configuration?: string | null
  cell_count?: number | null
  form_factor?: FormFactor | null
  manufacture_year?: number | null
  cycle_count_at_upload?: number | null
  operating_temperature_range_c?: TemperatureRange | null
  notes?: string | null
}
