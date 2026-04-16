import pytest

from src.keep_rule import candidate_beats_reference


def make_row(le=0.250, rv=0.270, nr=0.60, pcp=None):
    if pcp is None:
        pcp = {
            "comedone_open": 0.50,
            "comedone_closed": 0.50,
            "papule": 0.50,
            "pustule": 0.50,
            "nodule_cyst": 0.50,
        }
    return {
        "research_val_scored_map50_95": rv,
        "locked_eval_scored_map50_95": le,
        "locked_eval_nodule_cyst_recall": nr,
        "locked_eval_per_class_precision": pcp,
    }


def test_clear_keep():
    ref = make_row(le=0.250, rv=0.270)
    cand = make_row(le=0.260, rv=0.275)
    ok, reason = candidate_beats_reference(cand, ref)
    assert ok, f"Expected KEEP, got DISCARD: {reason}"


def test_discard_locked_eval_regression():
    ref = make_row(le=0.280)
    cand = make_row(le=0.270)  # regression of 0.010
    ok, reason = candidate_beats_reference(cand, ref)
    assert not ok
    assert "locked_eval regression" in reason


def test_discard_missing_field():
    ref = make_row()
    cand = make_row()
    del cand["locked_eval_nodule_cyst_recall"]
    ok, reason = candidate_beats_reference(cand, ref)
    assert not ok
    assert "missing field" in reason


def test_discard_nodule_recall_guardrail():
    ref = make_row()
    cand = make_row(le=0.260, rv=0.275, nr=0.45)  # below 0.50
    ok, reason = candidate_beats_reference(cand, ref)
    assert not ok
    assert "nodule_cyst recall" in reason


def test_discard_class_precision_drop():
    ref_pcp = {
        "comedone_open": 0.50,
        "comedone_closed": 0.45,
        "papule": 0.60,
        "pustule": 0.55,
        "nodule_cyst": 0.40,
    }
    cand_pcp = {
        "comedone_open": 0.50,
        "comedone_closed": 0.45,
        "papule": 0.45,
        "pustule": 0.55,
        "nodule_cyst": 0.40,
    }  # papule drop 0.15
    ref = make_row(pcp=ref_pcp)
    cand = make_row(le=0.260, rv=0.275, pcp=cand_pcp)
    ok, reason = candidate_beats_reference(cand, ref)
    assert not ok
    assert "papule precision drop" in reason


def test_historical_iter003_vs_iter013():
    """
    The case that motivated the rule change. Per-class precisions stubbed identical
    because the historical breakdown isn't on hand; this test verifies the
    locked_eval improvement axis specifically.
    """
    ref = make_row(le=0.2551, rv=0.2732, nr=0.5417)  # iter_013
    cand = make_row(le=0.2830, rv=0.2723, nr=0.5833)  # iter_003
    ok, reason = candidate_beats_reference(cand, ref)
    assert ok, f"Expected iter_003 to KEEP under new rule, got DISCARD: {reason}"


def test_keep_on_locked_eval_improvement_only():
    ref = make_row(le=0.250, rv=0.270)
    cand = make_row(le=0.258, rv=0.271)  # rv +0.001 (below 0.003), le +0.008 (above 0.005)
    ok, reason = candidate_beats_reference(cand, ref)
    assert ok, f"Expected KEEP on le-only improvement: {reason}"


def test_discard_on_neither_improvement_threshold():
    ref = make_row(le=0.250, rv=0.270)
    cand = make_row(le=0.251, rv=0.271)  # both improvements below threshold
    ok, reason = candidate_beats_reference(cand, ref)
    assert not ok
    assert "improvement threshold" in reason
