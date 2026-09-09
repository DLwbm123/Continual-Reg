"""Preserve checkpoint-aligned evidence without editing an interrupted run."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping


def inherit_resume_evidence(checkpoint: Path, run_dir: Path,
                            payload: Mapping[str, object], protocol_migration: bool = False) -> dict:
    checkpoint = checkpoint.resolve(strict=True)
    source_run = checkpoint.parent.parent
    if source_run.resolve() == run_dir.resolve():
        raise ValueError("Resume evidence requires a new run directory")
    source_log = source_run / "train.jsonl"
    raw = source_log.read_bytes()
    lines = raw.splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    step = int(payload["global_step"])
    if len(rows) < step or [row["global_step"] for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("Source log is not a complete, unique checkpoint-aligned history")
    if any(not math.isfinite(v) for row in rows for v in row.values()
           if isinstance(v, (int, float))):
        raise ValueError("Source log contains non-finite evidence")
    prefix = b"".join(lines[:step])
    if prefix and not prefix.endswith(b"\n"):
        raise ValueError("Checkpoint-aligned source prefix has no final newline")
    if (run_dir / "train.jsonl").exists():
        raise FileExistsError("Refusing to overwrite a destination training log")

    prior = source_run / "resume_provenance.json"
    prior_rollbacks = json.loads(prior.read_text()).get("cumulative_rollback_training_seconds", 0.0) if prior.exists() else 0.0
    rollback_seconds = 0.0 if protocol_migration else sum(float(row["step_seconds"]) for row in rows[step:])
    training_seconds = sum(float(row["step_seconds"]) for row in rows[:step])
    evaluation_seconds = sum(float(row["elapsed_seconds"]) for row in payload.get("evaluations", []))
    evidence = {
        "source_run": str(source_run), "source_checkpoint": str(checkpoint),
        "source_checkpoint_bytes": checkpoint.stat().st_size,
        "source_checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
        "checkpoint_global_step": step, "source_last_logged_step": len(rows),
        "inherited_rows": step, "rollback_steps": 0 if protocol_migration else len(rows) - step,
        "protocol_migration": protocol_migration,
        "discarded_old_protocol_suffix_steps": len(rows) - step if protocol_migration else 0,
        "source_log_sha256": hashlib.sha256(raw).hexdigest(),
        "inherited_prefix_sha256": hashlib.sha256(prefix).hexdigest(),
        "inherited_training_seconds": training_seconds,
        "inherited_evaluation_seconds": evaluation_seconds,
        "rollback_training_seconds": rollback_seconds,
        "cumulative_rollback_training_seconds": prior_rollbacks + rollback_seconds,
        "inherited_accounted_seconds": training_seconds + evaluation_seconds + prior_rollbacks + rollback_seconds,
        "runtime_basis": "accounted_active_time_lower_bound; excludes outage and unrecorded prior checkpoint overhead",
        "source_artifacts_unchanged": True,
    }
    with (run_dir / "train.jsonl").open("xb") as handle:
        handle.write(prefix)
    for path in sorted((source_run / "checkpoints").glob("after_*.pt")):
        tasks = ("oasis", "ctct", "nlst", "mrct")
        task = path.stem.removeprefix("after_")
        if "next_task_index" in payload and task in tasks and tasks.index(task) >= int(payload["next_task_index"]):
            continue
        (run_dir / "checkpoints" / path.name).symlink_to(path.resolve(strict=True))
    (run_dir / "checkpoints/resume_origin.pt").symlink_to(checkpoint)
    (run_dir / "checkpoints/latest.pt").symlink_to("resume_origin.pt")
    with (run_dir / "resume_provenance.json").open("x") as handle:
        json.dump(evidence, handle, indent=2)
        handle.write("\n")
    return evidence
