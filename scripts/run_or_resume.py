from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spurbreast_repro.config import config_sha256, load_config, project_path  # noqa: E402
from spurbreast_repro.utils import json_dumps  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_preflight(config: dict[str, Any]) -> dict[str, str]:
    dataset_dir = project_path(config["data"]["dataset_dir"])
    manifest = project_path(config["data"]["manifest_file"])
    audit_path = PROJECT_ROOT / "reports" / "tables" / "data_audit_summary.json"
    checksum_path = PROJECT_ROOT / "data" / "raw" / "CHECKSUMS.json"
    for required in (dataset_dir, manifest, audit_path, checksum_path):
        if not required.exists():
            raise RuntimeError(f"Data preflight missing: {required}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    checksums = json.loads(checksum_path.read_text(encoding="utf-8"))
    if audit.get("status") != "passed":
        raise RuntimeError("Data audit status is not passed")
    actual_manifest = file_sha256(manifest)
    if actual_manifest != audit.get("manifest_sha256"):
        raise RuntimeError("Manifest hash does not match the audited manifest")
    actual_md5 = checksums.get("archive", {}).get("md5")
    expected_md5 = config["download"]["archive_md5"]
    if actual_md5 != expected_md5:
        raise RuntimeError("Archive MD5 provenance does not match the experiment config")
    return {"archive_md5": actual_md5, "manifest_sha256": actual_manifest}


def matching_run_config(result_dir: Path, expected_hash: str) -> bool:
    run_config = result_dir / "config.yaml"
    if not run_config.is_file():
        return False
    loaded, _ = load_config(run_config)
    return config_sha256(loaded) == expected_hash


def choose_action(config: dict[str, Any], force_new: bool = False) -> tuple[str, Path | None]:
    if force_new:
        return "start", None
    name = str(config["experiment"]["name"])
    seed = int(config["experiment"]["seed"])
    expected_hash = config_sha256(config)
    checkpoint_root = project_path(config["paths"]["checkpoints"])
    result_root = project_path(config["paths"]["results"])
    candidates = sorted(
        checkpoint_root.glob(f"{name}-seed{seed}-*"),
        key=lambda path: path.name,
        reverse=True,
    )
    for checkpoint_dir in candidates:
        result_dir = result_root / checkpoint_dir.name
        if not matching_run_config(result_dir, expected_hash):
            continue
        summary_path = result_dir / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("status") == "completed":
                return "complete", checkpoint_dir / "best.pt"
        latest = checkpoint_dir / "latest.pt"
        if latest.is_file():
            return "resume", latest
    return "start", None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely start, resume, or skip a provenance-matched training run"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--force-new", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    provenance = data_preflight(config)
    action, checkpoint = choose_action(config, force_new=args.force_new)
    output = {
        "action": action,
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "config": str(config_path),
        "config_sha256": config_sha256(config),
        **provenance,
    }
    print(json_dumps(output, indent=2))
    if args.dry_run or action == "complete":
        return

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train.py"),
        "--config",
        str(config_path),
        "--device",
        args.device,
    ]
    if action == "resume" and checkpoint is not None:
        command.extend(["--resume", str(checkpoint)])
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
