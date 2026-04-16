import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src import api_server


CROPPED_HASH = "cropped-hash"
UNCROPPED_HASH = "uncropped-hash"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _configure_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    runs_dir = tmp_path / ".pulled_artifacts" / "runs"
    active_checkpoint_path = tmp_path / "config" / "active_checkpoint.json"
    runs_dir.mkdir(parents=True)

    for preprocessing, preprocessing_hash in {
        "cropped": CROPPED_HASH,
        "uncropped": UNCROPPED_HASH,
    }.items():
        hash_path = tmp_path / "data" / preprocessing / "preprocessing_hash.txt"
        hash_path.parent.mkdir(parents=True)
        hash_path.write_text(preprocessing_hash + "\n", encoding="utf-8")

    monkeypatch.setattr(api_server, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(api_server, "PULLED_ARTIFACTS_DIR", tmp_path / ".pulled_artifacts")
    monkeypatch.setattr(api_server, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(api_server, "ACTIVE_CHECKPOINT_PATH", active_checkpoint_path)
    return runs_dir, active_checkpoint_path


def _make_run(
    runs_dir: Path,
    run_id: str,
    *,
    decision: str = "KEEP",
    discard_reason: str | None = None,
    preprocessing_hash: str | None = CROPPED_HASH,
    row_preprocessing: str | None = None,
    best: bool = True,
    last: bool = True,
    locked_eval: float = 0.1,
    research_val: float = 0.2,
) -> Path:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    weights_dir = run_dir / "weights"
    weights_dir.mkdir()
    if best:
        (weights_dir / "best.pt").write_bytes(b"best weights")
    if last:
        (weights_dir / "last.pt").write_bytes(b"last weights")
    if preprocessing_hash is not None:
        (weights_dir / "preprocessing_hash.txt").write_text(preprocessing_hash + "\n", encoding="utf-8")

    row = {
        "run_id": run_id,
        "timestamp": f"2026-04-16T00:00:{len(run_id):02d}+00:00",
        "decision": decision,
        "discard_reason": discard_reason,
        "research_val_scored_map50_95": research_val,
        "locked_eval_scored_map50_95": locked_eval,
        "locked_eval_nodule_cyst_recall": 0.5,
        "locked_eval_per_class_precision": {
            class_name: 0.5 for class_name in api_server.SCORED_CLASSES
        },
        "preprocessing_hash": preprocessing_hash or "",
    }
    if row_preprocessing is not None:
        row["preprocessing"] = row_preprocessing
    _write_json(run_dir / "experiment_row.json", row)
    return run_dir


def _write_active_checkpoint(active_checkpoint_path: Path, run_id: str) -> None:
    _write_json(
        active_checkpoint_path,
        {
            "weights_path": f".pulled_artifacts/runs/{run_id}/weights/best.pt",
            "preprocessing": "cropped",
            "iteration_id": run_id,
        },
    )


def test_checkpoint_resolver_prefers_best_weight_for_kept_iteration(tmp_path, monkeypatch):
    runs_dir, _ = _configure_paths(tmp_path, monkeypatch)
    _make_run(runs_dir, "iter_018", decision="KEEP", best=True, last=True)

    checkpoint = api_server._checkpoint_for_iteration("iter_018")

    assert checkpoint.weights_path.name == "best.pt"
    assert checkpoint.preprocessing == "cropped"
    assert checkpoint.iteration_id == "iter_018"


def test_discarded_iteration_with_weights_is_usable(tmp_path, monkeypatch):
    runs_dir, _ = _configure_paths(tmp_path, monkeypatch)
    _make_run(
        runs_dir,
        "iter_019",
        decision="DISCARD",
        discard_reason="comedone_open precision drop",
    )

    option = api_server._checkpoint_option_payload(api_server._resolve_iteration_checkpoint("iter_019"))

    assert option["decision"] == "DISCARD"
    assert option["usable"] is True
    assert option["exploratory_only"] is True
    assert option["discard_reason"] == "comedone_open precision drop"


def test_missing_weights_are_listed_but_disabled(tmp_path, monkeypatch):
    runs_dir, _ = _configure_paths(tmp_path, monkeypatch)
    _make_run(runs_dir, "iter_020", best=False, last=False)

    options = api_server._all_checkpoint_options()
    option = next(item for item in options if item["run_id"] == "iter_020")

    assert option["weights_exists"] is False
    assert option["usable"] is False
    assert option["unusable_reason"] == "missing weights"


def test_unknown_preprocessing_hash_is_rejected(tmp_path, monkeypatch):
    runs_dir, _ = _configure_paths(tmp_path, monkeypatch)
    _make_run(runs_dir, "iter_021", preprocessing_hash="mystery-hash")

    resolution = api_server._resolve_iteration_checkpoint("iter_021")
    assert resolution.usable is False
    assert resolution.unusable_reason == "unknown preprocessing hash: mystery-hash"

    with pytest.raises(HTTPException) as exc:
        api_server._checkpoint_for_iteration("iter_021")
    assert exc.value.status_code == 400
    assert "unknown preprocessing hash" in str(exc.value.detail)


def test_path_traversal_run_ids_are_rejected(tmp_path, monkeypatch):
    _configure_paths(tmp_path, monkeypatch)

    with pytest.raises(HTTPException) as exc:
        api_server._checkpoint_for_iteration("../iter_018")

    assert exc.value.status_code == 400
    assert "invalid iteration_id" in str(exc.value.detail)


def test_inference_checkpoint_endpoint_lists_usable_and_unusable_runs(tmp_path, monkeypatch):
    runs_dir, _ = _configure_paths(tmp_path, monkeypatch)
    _make_run(runs_dir, "iter_018", decision="KEEP")
    _make_run(runs_dir, "iter_022", best=False, last=False)

    client = TestClient(api_server.app)
    response = client.get("/api/inference_checkpoints")

    assert response.status_code == 200
    options = {item["run_id"]: item for item in response.json()}
    assert options["iter_018"]["usable"] is True
    assert options["iter_022"]["usable"] is False
    assert options["iter_022"]["unusable_reason"] == "missing weights"


def test_infer_uses_active_checkpoint_when_iteration_id_is_omitted(tmp_path, monkeypatch):
    runs_dir, active_checkpoint_path = _configure_paths(tmp_path, monkeypatch)
    _make_run(runs_dir, "iter_018", decision="KEEP")
    _write_active_checkpoint(active_checkpoint_path, "iter_018")

    async def fake_read_upload_image(upload):
        await upload.read()
        return object()

    def fake_inference_response(image, checkpoint):
        counts = {class_name: 0 for class_name in api_server.ALL_CLASSES}
        counts["papule"] = 1 if checkpoint.iteration_id == "iter_018" else 0
        return {"detections": [], "counts": counts, "hayashi_badge": api_server.hayashi_badge(counts)}

    monkeypatch.setattr(api_server, "_read_upload_image", fake_read_upload_image)
    monkeypatch.setattr(api_server, "_inference_response", fake_inference_response)

    client = TestClient(api_server.app)
    response = client.post(
        "/api/infer",
        files={"image": ("sample.jpg", b"not-a-real-image", "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["counts"]["papule"] == 1


def test_infer_can_use_explicit_iteration_id(tmp_path, monkeypatch):
    runs_dir, _ = _configure_paths(tmp_path, monkeypatch)
    _make_run(runs_dir, "iter_019", decision="DISCARD")

    async def fake_read_upload_image(upload):
        await upload.read()
        return object()

    def fake_inference_response(image, checkpoint):
        counts = {class_name: 0 for class_name in api_server.ALL_CLASSES}
        counts["papule"] = 2 if checkpoint.iteration_id == "iter_019" else 0
        return {"detections": [], "counts": counts, "hayashi_badge": api_server.hayashi_badge(counts)}

    monkeypatch.setattr(api_server, "_read_upload_image", fake_read_upload_image)
    monkeypatch.setattr(api_server, "_inference_response", fake_inference_response)

    client = TestClient(api_server.app)
    response = client.post(
        "/api/infer",
        data={"iteration_id": "iter_019"},
        files={"image": ("sample.jpg", b"not-a-real-image", "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["counts"]["papule"] == 2


def test_compare_iterations_returns_b_minus_a_count_deltas(tmp_path, monkeypatch):
    runs_dir, _ = _configure_paths(tmp_path, monkeypatch)
    _make_run(runs_dir, "iter_018", decision="KEEP", locked_eval=0.08)
    _make_run(runs_dir, "iter_019", decision="DISCARD", locked_eval=0.12)

    async def fake_read_upload_image(upload):
        await upload.read()
        return object()

    def fake_inference_response(image, checkpoint):
        counts = {class_name: 0 for class_name in api_server.ALL_CLASSES}
        counts["papule"] = 1 if checkpoint.iteration_id == "iter_018" else 3
        return {"detections": [], "counts": counts, "hayashi_badge": api_server.hayashi_badge(counts)}

    monkeypatch.setattr(api_server, "_read_upload_image", fake_read_upload_image)
    monkeypatch.setattr(api_server, "_inference_response", fake_inference_response)

    client = TestClient(api_server.app)
    response = client.post(
        "/api/compare_iterations",
        data={"iteration_a": "iter_018", "iteration_b": "iter_019"},
        files={"image": ("sample.jpg", b"not-a-real-image", "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["iteration_a"]["run_id"] == "iter_018"
    assert payload["iteration_b"]["run_id"] == "iter_019"
    assert payload["deltas"]["counts_diff"]["papule"] == 2
