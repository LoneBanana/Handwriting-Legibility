"""Async tests here run on asyncio only (the app does not depend on trio)."""

from __future__ import annotations

import pytest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
