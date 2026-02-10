from __future__ import annotations

import os

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests that require Dockerized databases and API keys",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: requires running PostgreSQL/Memgraph (and sometimes API keys)",
    )


def _integration_enabled(config: pytest.Config) -> bool:
    if config.getoption("--integration"):
        return True
    return os.getenv("RUN_INTEGRATION") in {"1", "true", "TRUE", "yes", "YES"}


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if _integration_enabled(config):
        return

    skip_marker = pytest.mark.skip(
        reason="integration tests disabled (use --integration or RUN_INTEGRATION=1)"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_marker)
