"""Content-addressed caching for pipeline stages."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any


CACHE_FORMAT_VERSION = 1


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_fingerprint(
    stage: str,
    inputs: Iterable[Path],
    code_files: Iterable[Path],
    parameters: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(f"cache-format:{CACHE_FORMAT_VERSION}\nstage:{stage}\n".encode())
    for label, paths in (("input", inputs), ("code", code_files)):
        for path in sorted((Path(item).resolve() for item in paths), key=str):
            if not path.is_file():
                raise FileNotFoundError(f"Cache {label} does not exist: {path}")
            digest.update(f"{label}:{path}:{file_digest(path)}\n".encode())
    encoded_parameters = json.dumps(
        parameters, sort_keys=True, separators=(",", ":"), default=str
    )
    digest.update(encoded_parameters.encode())
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}-", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(payload, output, indent=2, sort_keys=True, default=str)
            output.write("\n")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def run_cached_stage(
    *,
    stage: str,
    output_path: Path,
    manifest_path: Path,
    inputs: Iterable[Path],
    code_files: Iterable[Path],
    parameters: Mapping[str, Any],
    build: Callable[[Path], Mapping[str, Any]],
    force: bool = False,
) -> tuple[bool, dict[str, Any]]:
    """Build a stage atomically, or return its matching cached metadata."""
    fingerprint = stage_fingerprint(stage, inputs, code_files, parameters)
    if not force and output_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("fingerprint") == fingerprint
            and manifest.get("output_digest") == file_digest(output_path)
        ):
            return True, dict(manifest.get("metadata", {}))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.stem}-",
        suffix=output_path.suffix or ".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    temporary_path.unlink()
    try:
        metadata = dict(build(temporary_path))
        if not temporary_path.is_file():
            raise RuntimeError(f"Stage {stage} did not create {temporary_path}")
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    manifest = {
        "cache_format": CACHE_FORMAT_VERSION,
        "fingerprint": fingerprint,
        "metadata": metadata,
        "output": str(output_path.resolve()),
        "output_digest": file_digest(output_path),
        "parameters": dict(parameters),
        "stage": stage,
    }
    _write_json_atomic(manifest_path, manifest)
    return False, metadata
