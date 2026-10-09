export interface Envelope<T> {
  success: boolean
  data: T
  error: { code: string; message: string; suggestion?: string } | null
}

export interface User {
  id: string
  email: string
  name: string
  consent: Consent
  created_at?: string | null
}

export interface Consent {
  save_mood_history: boolean
  camera_access: boolean
  store_images: boolean
  consent_updated_at?: string
}

export interface Distribution {
  [emotion: string]: number
}

export interface LatencyBreakdown {
  [stage: string]: number
}

export interface TextSignal {
  emotion: string
  confidence: number
  distribution: Distribution
  explanation?: {
    method: string | null
    contributing_tokens?: { token: string; score: number }[]
    note?: string
    error?: string
  } | null
  latency_ms?: number
}

export interface VisionSignal {
  expression?: {
    label: string
    display_label: string
    confidence: number
    distribution: Distribution
  }
  emotion?: string
  confidence?: number
  distribution?: Distribution
  explanation?: { method: string | null; note?: string; error?: string } | null
  latency_ms?: number
  framing?: string
  error?: string
  message?: string | null
}

export interface FusionInfo {
  mode: 'multimodal' | 'text_only' | 'vision_only'
  text_weight: number
  vision_weight: number
  formula: string
  weights_source: string
  latency_ms: number
}

export interface CrisisResult {
  level: 'LOW' | 'MODERATE' | 'HIGH'
  confidence: number
  probabilities: { LOW: number; MODERATE: number; HIGH: number }
  signals?: { signal: string; weight: number }[]
  disclaimer: string
}

export interface CrisisResources {
  country: string | null
  message: string
  resources: { name: string; description: string; contact: string; type: string; source: string }[]
  international: { name: string; description: string; contact: string; type: string; source: string }[]
  emergency: { description: string; type: string }
  note?: string | null
}

export interface StressorResult {
  category: string | null
  confidence: number
  intensity: 'low' | 'medium' | 'high' | string
  matched_cues: { cue: string; category: string }[]
  method: string
  note?: string
}

export interface WellnessResult {
  state: 'STABLE' | 'NEEDS_ATTENTION' | 'HIGH_CONCERN'
  score: number
  components: Record<string, number>
  explanations: string[]
  disclaimer: string
}

export interface Intervention {
  id: string
  name: string
  kind: string
  duration_minutes: number
  description: string
  steps: string[]
  priority: number
  base_priority?: number
  rationale: string[]
  disclaimer: string
}

export interface InterventionResult {
  crisis_mode: boolean
  headline: string
  interventions: Intervention[]
  learning_note: string
  disclaimer: string
}

export interface ChatResult {
  crisis: CrisisResult | null
  crisis_resources: CrisisResources | null
  emotion: string
  confidence: number
  distribution: Distribution
  text_signal: TextSignal | null
  vision_signal: VisionSignal | null
  fusion: FusionInfo
  stressor: StressorResult | null
  wellness: WellnessResult
  interventions: InterventionResult
  trend_context: { declining: boolean } | null
  saved: { saved: boolean; id?: string; reason?: string; message?: string } | null
  latency_ms: LatencyBreakdown
  disclaimer: string
}

export interface MoodRecord {
  id: string
  timestamp: string
  text_emotion?: string | null
  text_confidence?: number | null
  vision_emotion?: string | null
  vision_confidence?: number | null
  fused_emotion: string
  fused_confidence: number
  valence?: number
  stressor?: string | null
  severity?: string | null
  wellness_state?: string | null
  crisis_level?: string | null
  intervention_id?: string | null
}

export interface DailyPoint {
  date: string
  valence: number
  checkins: number
}

export interface TrendInfo {
  slope_per_day?: number
  direction?: string
  recent_7d_mean?: number
  previous_7d_mean?: number | null
  change_vs_previous_7d?: number | null
  declining?: boolean
  weekday_means?: Record<string, number | null>
  recurring_patterns?: { weakest_weekdays: string[]; note: string } | null
}

export interface TrendsResult {
  available: boolean
  message?: string
  records?: number
  daily_series?: DailyPoint[]
  trend?: TrendInfo
  emotion_distribution?: Record<string, number>
  stressor_distribution?: Record<string, number>
  weekly?: {
    recent_7d_mean_valence: number | null
    previous_7d_mean_valence: number | null
    recent_checkins: number
    previous_checkins: number
  }
  intervention_effectiveness?: {
    intervention_id: string
    ratings: Record<string, number>
    total: number
    mean_score: number
  }[]
}

export interface ForecastResult {
  available: boolean
  reason?: string
  message?: string
  observations?: number
  min_observations_required: number
  method?: string
  horizon_days?: number
  daily_series: DailyPoint[]
  trend?: TrendInfo
  forecast?: { date: string; forecast_valence: number; potential_stress_period: boolean }[]
  stress_periods?: string[]
  disclaimer: string
}

export interface ModelsStatus {
  device: {
    selected: string
    cuda_available: boolean
    cuda_device_name?: string | null
    note?: string | null
    torch_version?: string | null
    error?: string
  }
  models: Record<string, Record<string, unknown>>
  saved_metrics: Record<string, Record<string, unknown>>
  checked_at: string
}
