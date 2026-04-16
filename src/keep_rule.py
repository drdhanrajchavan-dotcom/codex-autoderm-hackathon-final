"""Held-out-primary keep rule for AutoDerm autoresearch."""

from __future__ import annotations

from src.types import ExperimentRow, SCORED_CLASSES


REQUIRED_TOP_LEVEL_FIELDS = [
    "research_val_scored_map50_95",
    "locked_eval_scored_map50_95",
    "locked_eval_nodule_cyst_recall",
    "locked_eval_per_class_precision",
]

LOCKED_EVAL_REGRESSION_TOLERANCE = 0.005
RESEARCH_VAL_IMPROVEMENT_THRESHOLD = 0.003
LOCKED_EVAL_IMPROVEMENT_THRESHOLD = 0.005
NODULE_CYST_RECALL_FLOOR = 0.50
CLASS_PRECISION_DROP_LIMIT = 0.10


def has_required_fields(row: dict) -> tuple[bool, str | None]:
    """Returns (True, None) if all required fields present, else (False, reason)."""
    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in row:
            return (False, f"missing field: {field}")
    pcp = row["locked_eval_per_class_precision"]
    if not isinstance(pcp, dict):
        return (False, "locked_eval_per_class_precision is not a dict")
    for cls in SCORED_CLASSES:
        if cls not in pcp:
            return (False, f"missing field: locked_eval_per_class_precision[{cls}]")
    return (True, None)


def passes_locked_eval_regression(candidate: dict, reference: dict) -> tuple[bool, str | None]:
    """Criterion 1."""
    delta = candidate["locked_eval_scored_map50_95"] - reference["locked_eval_scored_map50_95"]
    if delta < -LOCKED_EVAL_REGRESSION_TOLERANCE:
        return (
            False,
            f"locked_eval regression {delta:.4f} exceeds tolerance {LOCKED_EVAL_REGRESSION_TOLERANCE}",
        )
    return (True, None)


def passes_improvement_threshold(candidate: dict, reference: dict) -> tuple[bool, str | None]:
    """Criterion 2."""
    rv_delta = candidate["research_val_scored_map50_95"] - reference["research_val_scored_map50_95"]
    le_delta = candidate["locked_eval_scored_map50_95"] - reference["locked_eval_scored_map50_95"]
    if rv_delta >= RESEARCH_VAL_IMPROVEMENT_THRESHOLD or le_delta >= LOCKED_EVAL_IMPROVEMENT_THRESHOLD:
        return (True, None)
    return (False, f"improvement threshold not met: rv_delta={rv_delta:.4f}, le_delta={le_delta:.4f}")


def passes_nodule_recall_guardrail(candidate: dict) -> tuple[bool, str | None]:
    """Criterion 3."""
    nr = candidate["locked_eval_nodule_cyst_recall"]
    if nr < NODULE_CYST_RECALL_FLOOR:
        return (False, f"nodule_cyst recall {nr:.4f} below floor {NODULE_CYST_RECALL_FLOOR}")
    return (True, None)


def passes_class_precision_guardrail(candidate: dict, reference: dict) -> tuple[bool, str | None]:
    """Criterion 4."""
    cand_pcp = candidate["locked_eval_per_class_precision"]
    ref_pcp = reference["locked_eval_per_class_precision"]
    for cls in SCORED_CLASSES:
        drop = ref_pcp[cls] - cand_pcp[cls]
        if drop > CLASS_PRECISION_DROP_LIMIT:
            return (False, f"{cls} precision drop {drop:.4f} exceeds limit {CLASS_PRECISION_DROP_LIMIT}")
    return (True, None)


def candidate_beats_reference(candidate: dict, reference: dict) -> tuple[bool, str | None]:
    """
    Returns (True, None) if candidate should be KEPT, (False, discard_reason) otherwise.
    Checks required fields, then 4 criteria in order. First failure short-circuits.
    """
    ok, reason = has_required_fields(candidate)
    if not ok:
        return (False, reason)
    ok, reason = has_required_fields(reference)
    if not ok:
        return (False, f"reference invalid: {reason}")
    for check in [passes_locked_eval_regression, passes_improvement_threshold]:
        ok, reason = check(candidate, reference)
        if not ok:
            return (False, reason)
    ok, reason = passes_nodule_recall_guardrail(candidate)
    if not ok:
        return (False, reason)
    ok, reason = passes_class_precision_guardrail(candidate, reference)
    if not ok:
        return (False, reason)
    return (True, None)
