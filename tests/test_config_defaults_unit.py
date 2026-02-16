from __future__ import annotations

import importlib


def test_enable_seed_rerank_should_default_to_true(monkeypatch) -> None:
    monkeypatch.delenv("ENABLE_SEED_RERANK", raising=False)

    import lib.utils.config as config_module

    config_module = importlib.reload(config_module)
    assert config_module.config.enable_seed_rerank is True
