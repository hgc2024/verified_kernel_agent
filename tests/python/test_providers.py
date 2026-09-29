from __future__ import annotations

import pytest

from vka.agent.providers import ProviderConfig


def test_from_env_reads_all_fields() -> None:
    cfg = ProviderConfig.from_env(
        {
            "VKA_BASE_URL": "http://localhost:8000/v1/",
            "VKA_API_KEY": "secret",
            "VKA_MODEL": "some-model",
        }
    )
    assert cfg.base_url == "http://localhost:8000/v1"
    assert cfg.model == "some-model"
    assert cfg.api_key == "secret"


def test_api_key_is_optional_for_local_servers() -> None:
    cfg = ProviderConfig.from_env({"VKA_BASE_URL": "http://localhost:11434/v1", "VKA_MODEL": "m"})
    assert cfg.api_key == ""


def test_api_key_not_in_repr() -> None:
    cfg = ProviderConfig(base_url="http://x", model="m", api_key="secret")
    assert "secret" not in repr(cfg)


@pytest.mark.parametrize("missing", ["VKA_BASE_URL", "VKA_MODEL"])
def test_missing_required_variable_raises(missing: str) -> None:
    env = {"VKA_BASE_URL": "http://localhost:11434/v1", "VKA_MODEL": "m"}
    env[missing] = ""
    with pytest.raises(ValueError, match=missing):
        ProviderConfig.from_env(env)
