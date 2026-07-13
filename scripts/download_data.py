from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spurbreast_repro.config import load_config, project_path  # noqa: E402


def file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_resumable(url: str, destination: Path, expected_md5: str | None = None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if expected_md5 and file_hash(destination, "md5") != expected_md5:
            raise RuntimeError(f"Existing file has the wrong MD5: {destination}")
        print(f"Already present: {destination}")
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "SpurBreast-Reproduction/0.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", 200)
        if offset and status != 206:
            print("Server did not honor the resume range; restarting the partial download.")
            offset = 0
        content_range = response.headers.get("Content-Range")
        if content_range and "/" in content_range:
            total = int(content_range.rsplit("/", 1)[1])
        else:
            total = offset + int(response.headers.get("Content-Length", "0"))
        mode = "ab" if offset and status == 206 else "wb"
        with partial.open(mode) as handle, tqdm(
            total=total or None,
            initial=offset,
            unit="B",
            unit_scale=True,
            desc=destination.name,
        ) as progress:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                handle.write(block)
                progress.update(len(block))
    if expected_md5:
        observed = file_hash(partial, "md5")
        if observed != expected_md5:
            raise RuntimeError(f"MD5 mismatch for {partial}: {observed} != {expected_md5}")
    os.replace(partial, destination)
    return destination


def download_parallel_ranges(
    url: str,
    destination: Path,
    total_bytes: int,
    expected_md5: str,
    workers: int = 8,
    segment_bytes: int = 16 * 1024 * 1024,
) -> Path:
    """Resume a verified prefix and fetch the remaining non-overlapping ranges in parallel."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_hash(destination, "md5") != expected_md5:
            raise RuntimeError(f"Existing file has the wrong MD5: {destination}")
        print(f"Already present: {destination}")
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    prefix_bytes = partial.stat().st_size if partial.exists() else 0
    if prefix_bytes > total_bytes:
        raise RuntimeError(f"Partial file is larger than expected: {partial}")
    segment_dir = destination.parent / f".{destination.name}.segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    ranges = [
        (start, min(start + segment_bytes - 1, total_bytes - 1))
        for start in range(prefix_bytes, total_bytes, segment_bytes)
    ]

    def segment_path(start: int, end: int) -> Path:
        return segment_dir / f"{start:012d}-{end:012d}.part"

    completed_bytes = 0
    for start, end in ranges:
        path = segment_path(start, end)
        expected_size = end - start + 1
        if path.exists() and path.stat().st_size == expected_size:
            completed_bytes += expected_size

    progress = tqdm(
        total=total_bytes,
        initial=prefix_bytes + completed_bytes,
        unit="B",
        unit_scale=True,
        desc=f"{destination.name} parallel",
    )

    def fetch(start: int, end: int) -> Path:
        final_segment = segment_path(start, end)
        expected_size = end - start + 1
        if final_segment.exists() and final_segment.stat().st_size == expected_size:
            return final_segment
        temporary = final_segment.with_suffix(".tmp")
        temporary.unlink(missing_ok=True)
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "SpurBreast-Reproduction/0.1",
                "Range": f"bytes={start}-{end}",
            },
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            if getattr(response, "status", None) != 206:
                raise RuntimeError(f"Server did not honor range {start}-{end}")
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {start}-{end}/"):
                raise RuntimeError(f"Unexpected Content-Range: {content_range}")
            with temporary.open("wb") as handle:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    handle.write(block)
        if temporary.stat().st_size != expected_size:
            raise RuntimeError(
                f"Short range {start}-{end}: {temporary.stat().st_size} != {expected_size}"
            )
        os.replace(temporary, final_segment)
        progress.update(expected_size)
        return final_segment

    pending = [
        (start, end)
        for start, end in ranges
        if not segment_path(start, end).exists()
        or segment_path(start, end).stat().st_size != end - start + 1
    ]
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {executor.submit(fetch, start, end): (start, end) for start, end in pending}
            for future in as_completed(futures):
                future.result()
    finally:
        progress.close()

    assembled = destination.with_suffix(destination.suffix + ".assembled")
    assembled.unlink(missing_ok=True)
    with assembled.open("wb") as output:
        if partial.exists():
            with partial.open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
        for start, end in ranges:
            with segment_path(start, end).open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    if assembled.stat().st_size != total_bytes:
        raise RuntimeError(f"Assembled size mismatch: {assembled.stat().st_size} != {total_bytes}")
    observed = file_hash(assembled, "md5")
    if observed != expected_md5:
        raise RuntimeError(f"MD5 mismatch for assembled archive: {observed} != {expected_md5}")
    os.replace(assembled, destination)
    partial.unlink(missing_ok=True)
    shutil.rmtree(segment_dir)
    return destination


def safe_extract_archive(archive: Path, raw_dir: Path) -> Path:
    final_dataset = raw_dir / "field_strength"
    if final_dataset.exists():
        print(f"Already extracted: {final_dataset}")
        return final_dataset
    temporary = raw_dir / ".field_strength_extracting"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    root = temporary.resolve()
    try:
        with zipfile.ZipFile(archive) as bundle:
            bad_member = bundle.testzip()
            if bad_member is not None:
                raise RuntimeError(f"Corrupt ZIP member: {bad_member}")
            for member in bundle.infolist():
                candidate = (temporary / member.filename).resolve()
                if not candidate.is_relative_to(root):
                    raise RuntimeError(f"Unsafe ZIP member path: {member.filename}")
            bundle.extractall(temporary)
        extracted_root = temporary / "field_strength"
        if not extracted_root.is_dir():
            raise RuntimeError("Archive did not contain the expected field_strength root")
        shutil.move(str(extracted_root), str(final_dataset))
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return final_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify official SpurBreast data")
    parser.add_argument("--config", default="configs/reproduction.yaml")
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    config, _ = load_config(args.config)
    download = config["download"]
    archive = project_path(download["archive_file"])
    metadata = project_path(download["metadata_file"])
    archive = download_parallel_ranges(
        download["archive_url"],
        archive,
        total_bytes=int(download["archive_bytes"]),
        expected_md5=download["archive_md5"],
        workers=args.workers,
    )
    metadata = download_resumable(download["metadata_url"], metadata)
    if not args.no_extract:
        safe_extract_archive(archive, archive.parent)
    checksums = {
        "accessed_at": datetime.now(timezone.utc).isoformat(),
        "archive": {
            "path": archive.name,
            "bytes": archive.stat().st_size,
            "md5": file_hash(archive, "md5"),
            "sha256": file_hash(archive, "sha256"),
            "url": download["archive_url"],
        },
        "metadata": {
            "path": metadata.name,
            "bytes": metadata.stat().st_size,
            "sha256": file_hash(metadata, "sha256"),
            "url": download["metadata_url"],
        },
    }
    checksum_path = archive.parent / "CHECKSUMS.json"
    checksum_path.write_text(json.dumps(checksums, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checksums, indent=2))


if __name__ == "__main__":
    main()
