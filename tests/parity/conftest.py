"""Opt-in MATLAB parity tier: every test here is marked ``parity`` and skips when refs are missing."""

from pathlib import Path

import pytest

from tests.parity._loaders import require_ref

_PARITY_ROOT = Path(__file__).resolve().parent


def pytest_collection_modifyitems(config, items):
    for item in items:
        if _PARITY_ROOT in Path(str(item.fspath)).resolve().parents:
            item.add_marker(pytest.mark.parity)


@pytest.fixture
def parity_ref():
    """Factory fixture: ``parity_ref("refs_distributions")`` loads the ref or skips the test."""
    return require_ref
