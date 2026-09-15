from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

LATEST_POINTER = "latest.txt"


def promote(model_path: Path, registry_dir: Path, latest_name: str = LATEST_POINTER) -> Path:
    registry_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    versioned_path = registry_dir / f"model-{stamp}{model_path.suffix}"
    shutil.copy2(model_path, versioned_path)
    (registry_dir / latest_name).write_text(versioned_path.name)
    return versioned_path
