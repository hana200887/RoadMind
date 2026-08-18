"""Shared pytest fixtures for the roadmind test suite."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture()
def scratch_dir() -> Iterator[Path]:
    """Independent temporary directory per test.

    pytest's own ``tmp_path`` is unusable on some Windows machines because
    of a broken ACL on the shared ``pytest-of-<user>`` base directory, so
    tests use this fixture instead.
    """
    path = Path(tempfile.mkdtemp(prefix="roadmind-tests-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="session")
def file_symlink_supported() -> bool:
    """Whether the platform allows creating file symlinks."""
    with tempfile.TemporaryDirectory(prefix="roadmind-linkcheck-") as directory:
        base = Path(directory)
        try:
            os.symlink(base / "target", base / "link")
        except (OSError, NotImplementedError):
            return False
        return True
