"""Loaders for MATLAB parity reference files (written by ``matlab/export_density_parity_refs.m``)."""

from pathlib import Path

import pytest
from scipy.io import loadmat

PARITY_DIR = Path(__file__).resolve().parents[1] / "data" / "matlab_parity"


def _ref_path(name: str) -> Path:
    filename = name if name.endswith(".mat") else f"{name}.mat"
    return Path(PARITY_DIR) / filename


def ref_available(name: str) -> bool:
    return _ref_path(name).is_file()


def load_ref(name: str) -> dict:
    raw = loadmat(_ref_path(name), squeeze_me=True, struct_as_record=False)
    return {k: v for k, v in raw.items() if not k.startswith("__")}


def require_ref(name: str) -> dict:
    """Load a reference file, skipping the calling test if it has not been exported yet."""
    if not ref_available(name):
        pytest.skip(f"MATLAB parity reference '{name}' not found in {PARITY_DIR}")
    return load_ref(name)
