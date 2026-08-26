"""Small shared I/O helpers used across the CLI and export modules."""

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, data: Any, indent: int = 2) -> Path:
    """Write data as indented JSON to path, creating/overwriting the file."""
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
    return path
