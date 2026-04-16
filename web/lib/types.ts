export const SCORED_CLASSES = [
  "comedone_open",
  "comedone_closed",
  "papule",
  "pustule",
  "nodule_cyst",
] as const;
export const ALL_CLASSES = [...SCORED_CLASSES, "post_acne_mark"] as const;

export type ScoredClass = (typeof SCORED_CLASSES)[number];
export type AnyClass = (typeof ALL_CLASSES)[number];

export type HayashiBadge = "clear" | "almost_clear" | "mild" | "moderate" | "severe";
export type ExperimentDecision = "KEEP" | "DISCARD" | "FAILED" | "PENDING_KEEP_RULE";

export interface Detection {
  class_name: AnyClass;
  confidence: number;
  bbox: [number, number, number, number, number];
}

export interface InferenceResponse {
  detections: Detection[];
  counts: Record<AnyClass, number>;
  hayashi_badge: HayashiBadge;
}

export interface ExperimentRow {
  run_id: string;
  timestamp: string;
  decision: ExperimentDecision;
  discard_reason: string | null;
  research_val_scored_map50_95: number;
  locked_eval_scored_map50_95: number;
  locked_eval_nodule_cyst_recall: number;
  locked_eval_per_class_precision: Record<ScoredClass, number>;
  locked_eval_per_class_recall?: Record<ScoredClass, number>;
  research_val_per_class_precision?: Record<ScoredClass, number>;
  research_val_per_class_recall?: Record<ScoredClass, number>;
  train_duration_seconds?: number;
  timeout?: boolean;
  crashed?: boolean;
  preprocessing_hash?: string;
  notes?: string;
  preprocessing?: "uncropped" | "cropped" | null;
  weights_path?: string | null;
}

export interface ActiveCheckpoint {
  weights_path: string | null;
  preprocessing: "uncropped" | "cropped" | null;
  iteration_id: string | null;
  weights_exists: boolean;
}

export interface InferenceCheckpointOption {
  run_id: string;
  decision: ExperimentDecision;
  discard_reason: string | null;
  timestamp: string;
  research_val_scored_map50_95: number;
  locked_eval_scored_map50_95: number;
  locked_eval_nodule_cyst_recall: number;
  preprocessing: "uncropped" | "cropped" | null;
  weights_path: string | null;
  weight_file: string | null;
  weights_exists: boolean;
  usable: boolean;
  unusable_reason: string | null;
  official_keep: boolean;
  exploratory_only: boolean;
}

export interface IterationCompareResponse {
  iteration_a: InferenceCheckpointOption;
  iteration_b: InferenceCheckpointOption;
  a: InferenceResponse;
  b: InferenceResponse;
  deltas: {
    counts_diff: Record<AnyClass, number>;
    badge_change: {
      from: HayashiBadge;
      to: HayashiBadge;
    };
  };
}

export interface IterationDetail {
  experiment_row: ExperimentRow | Record<string, unknown> | null;
  train_diff_text: string;
  prompt_text: string;
  codex_response_text: string;
  worker_stdout_tail: string;
}
