from __future__ import annotations

import os
import tempfile
from typing import Any, Dict


def atomic_save(payload: Dict[str, Any], path: str) -> None:
    import torch

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    handle, staging = tempfile.mkstemp(dir=directory, suffix=".staging")
    os.close(handle)
    try:
        torch.save(payload, staging)
        os.replace(staging, path)
    finally:
        if os.path.exists(staging):
            os.remove(staging)


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    import torch

    payload: Dict[str, Any] = torch.load(path, map_location=map_location)
    return payload
