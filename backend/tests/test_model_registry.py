import json
from pathlib import Path

import pytest

from backend.services.model_registry import ModelRegistry


@pytest.fixture
def registry_instance():
    return ModelRegistry()


def test_load_json_file_returns_parsed_json(tmp_path: Path, monkeypatch, registry_instance: ModelRegistry):
    payload = {"ok": True, "items": [1, 2, 3]}
    file_path = tmp_path / "example.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr("backend.services.model_registry.MODELS_DIR", tmp_path)

    result = registry_instance._load_json_file("example.json")

    assert result == payload


def test_load_json_file_returns_empty_when_missing(tmp_path: Path, monkeypatch, registry_instance: ModelRegistry):
    monkeypatch.setattr("backend.services.model_registry.MODELS_DIR", tmp_path)

    result = registry_instance._load_json_file("missing.json")

    assert result == {}


def test_load_json_file_returns_empty_on_invalid_json(tmp_path: Path, monkeypatch, registry_instance: ModelRegistry):
    bad = tmp_path / "bad.json"
    bad.write_text("{invalid", encoding="utf-8")

    monkeypatch.setattr("backend.services.model_registry.MODELS_DIR", tmp_path)

    result = registry_instance._load_json_file("bad.json")

    assert result == {}


def test_get_model_returns_none_for_unknown(registry_instance: ModelRegistry):
    registry_instance._loaded = True

    assert registry_instance.get_model("unknown_model") is None


def test_get_model_returns_existing(registry_instance: ModelRegistry):
    marker = object()
    registry_instance.random_forest = marker
    registry_instance._loaded = True

    assert registry_instance.get_model("random_forest") is marker
