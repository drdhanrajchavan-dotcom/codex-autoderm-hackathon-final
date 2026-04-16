"""Shared type contracts for AutoDerm research and inference outputs."""

from typing import Literal, Optional, TypedDict


SCORED_CLASSES = ["comedone_open", "comedone_closed", "papule", "pustule", "nodule_cyst"]
ALL_CLASSES = SCORED_CLASSES + ["post_acne_mark"]


class PerClassMetrics(TypedDict):
    comedone_open: float
    comedone_closed: float
    papule: float
    pustule: float
    nodule_cyst: float


class ExperimentRow(TypedDict):
    run_id: str
    timestamp: str
    decision: Literal["KEEP", "DISCARD", "FAILED"]
    discard_reason: Optional[str]
    research_val_scored_map50_95: float
    locked_eval_scored_map50_95: float
    locked_eval_nodule_cyst_recall: float
    locked_eval_per_class_precision: PerClassMetrics
    locked_eval_per_class_recall: PerClassMetrics
    research_val_per_class_precision: PerClassMetrics
    research_val_per_class_recall: PerClassMetrics
    train_duration_seconds: float
    timeout: bool
    crashed: bool
    preprocessing_hash: str
    notes: str


class Detection(TypedDict):
    class_name: str
    confidence: float
    bbox: list[float]


class InferenceResponse(TypedDict):
    detections: list[Detection]
    counts: dict[str, int]
    hayashi_badge: Literal["clear", "almost_clear", "mild", "moderate", "severe"]
