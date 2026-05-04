import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go


def test_headers_include_api_key(frontend_app_module):
    assert frontend_app_module.HEADERS == {"X-API-Key": "test-api-key"}


def test_load_json_reads_valid_json(tmp_path: Path, frontend_app_module):
    payload = {"ok": True, "n": 2}
    f = tmp_path / "payload.json"
    f.write_text(json.dumps(payload), encoding="utf-8")

    data = frontend_app_module.load_json(f)

    assert data == payload


def test_load_json_returns_none_when_missing(tmp_path: Path, frontend_app_module):
    data = frontend_app_module.load_json(tmp_path / "missing.json")

    assert data is None


def test_load_json_handles_parse_error(tmp_path: Path, frontend_app_module, monkeypatch):
    bad = tmp_path / "bad.json"
    bad.write_text("{not-json", encoding="utf-8")

    seen = {}
    monkeypatch.setattr(frontend_app_module.st, "warning", lambda m: seen.setdefault("msg", m))

    data = frontend_app_module.load_json(bad)

    assert data is None
    assert "Failed to read" in seen["msg"]


def test_load_parquet_converts_date_column(tmp_path: Path, frontend_app_module):
    path = tmp_path / "sample.parquet"
    df = pd.DataFrame(
        [{"date": "2024-01-01", "pcode": "SO2501", "value": 1.0}]
    )
    df.to_parquet(path)

    loaded = frontend_app_module.load_parquet(path)

    assert loaded is not None
    assert str(loaded["date"].dtype).startswith("datetime64")


def test_load_parquet_returns_none_when_missing(tmp_path: Path, frontend_app_module):
    loaded = frontend_app_module.load_parquet(tmp_path / "nope.parquet")

    assert loaded is None


def test_load_joblib_round_trip(tmp_path: Path, frontend_app_module):
    model_path = tmp_path / "obj.joblib"
    expected = {"a": 1, "b": [2, 3]}
    joblib.dump(expected, model_path)

    loaded = frontend_app_module.load_joblib(model_path)

    assert loaded == expected


def test_load_joblib_returns_none_when_missing(tmp_path: Path, frontend_app_module):
    loaded = frontend_app_module.load_joblib(tmp_path / "missing.joblib")

    assert loaded is None


def test_theme_updates_layout(frontend_app_module):
    fig = go.Figure()

    themed = frontend_app_module._theme(fig, height=333)

    assert themed.layout.height == 333
    assert themed.layout.plot_bgcolor == frontend_app_module.PLOT_BG
    assert themed.layout.paper_bgcolor == frontend_app_module.PLOT_BG


def test_missing_helper_emits_info(frontend_app_module, monkeypatch):
    seen = {}
    monkeypatch.setattr(frontend_app_module.st, "info", lambda m: seen.setdefault("msg", m))

    frontend_app_module._missing("Dataset", "Do X")

    assert seen["msg"] == "Dataset not available. Do X"


def test_name_to_pcode_is_reverse_map(frontend_app_module):
    for pcode, district in frontend_app_module.DISTRICT_MAP.items():
        assert frontend_app_module.NAME_TO_PCODE[district] == pcode


def test_page_intros_cover_all_navigation_pages(frontend_app_module):
    assert set(frontend_app_module.PAGES).issubset(frontend_app_module.PAGE_INTROS)


def test_page_intro_text_is_meaningful(frontend_app_module):
    overview = frontend_app_module.PAGE_INTROS["📋 Overview"]
    predict = frontend_app_module.PAGE_INTROS["📡 Predict & Narrative"]

    assert "dataset footprint" in overview
    assert "syllabus coverage" in overview
    assert "district-month" in predict
    assert "human-readable brief" in predict


def test_render_page_intro_writes_markdown(frontend_app_module, monkeypatch):
    seen = {}
    monkeypatch.setattr(frontend_app_module.st, "markdown", lambda m, **kwargs: seen.setdefault("msg", m))

    frontend_app_module._render_page_intro("🔗 Association")

    assert "rule-based co-occurrence" in seen["msg"]
