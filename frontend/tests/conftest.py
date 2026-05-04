import importlib
import os
import sys

import pytest
import streamlit as st


os.environ.setdefault("API_KEY", "test-api-key")


@pytest.fixture
def frontend_app_module(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-api-key")
    monkeypatch.setenv("BACKEND_URL", "http://example.test")

    monkeypatch.setattr(st.sidebar, "radio", lambda *args, **kwargs: "📡 Predict & Narrative")
    monkeypatch.setattr(st, "button", lambda *args, **kwargs: False)

    if "frontend.app" in sys.modules:
        del sys.modules["frontend.app"]

    return importlib.import_module("frontend.app")
