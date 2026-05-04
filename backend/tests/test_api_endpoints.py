import asyncio

import pytest

from backend.routers import narrative as narrative_router


def test_health_endpoint_without_ollama(api_client):
    response = api_client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_available"] is True
    assert body["data_available"] is True
    assert body["ollama_available"] is False


def test_protected_routes_require_api_key(api_client):
    response = api_client.post("/predict/mortality", json={"district_pcode": "SO2501"})

    assert response.status_code == 403
    assert "Could not validate credentials" in response.text


def test_predict_mortality_success(api_client, auth_headers):
    payload = {
        "district_pcode": "SO2501",
        "rainfall_anomaly_pct": -30.0,
        "conflict_fatalities": 9,
        "food_price_index": 180.0,
        "ipc_phase4_pct": 20.0,
        "ipc_phase5_pct": 4.0,
    }

    response = api_client.post("/predict/mortality", json=payload, headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["district_pcode"] == "SO2501"
    assert body["risk_level"] == "High"
    assert body["probability"] == pytest.approx(0.82)
    assert body["shap_factors"]["rainfall_anomaly"] == -30.0


def test_predict_mortality_validation_error(api_client, auth_headers):
    payload = {"district_pcode": "SO1", "conflict_fatalities": -2}

    response = api_client.post("/predict/mortality", json=payload, headers=auth_headers)

    assert response.status_code == 422


def test_predict_mortality_missing_district_returns_500(api_client, auth_headers):
    response = api_client.post(
        "/predict/mortality",
        json={"district_pcode": "SO9999"},
        headers=auth_headers,
    )

    assert response.status_code == 500
    assert "No historical data found" in response.text


def test_analyze_rules_returns_all(api_client, auth_headers):
    response = api_client.get("/analyze/rules", headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert "fpgrowth" in body
    assert "apriori" in body


def test_analyze_rules_filters_by_algorithm(api_client, auth_headers):
    response = api_client.get("/analyze/rules?algorithm=fpgrowth", headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == {"fpgrowth": [{"rule": "A->B"}]}


def test_analyze_clusters_returns_profiles(api_client, auth_headers):
    response = api_client.get("/analyze/clusters", headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["pcode"] == "SO2501"
    assert body[0]["features"]["conflict_fatalities"] == 5


def test_anomaly_alerts_returns_registry_alerts(api_client, auth_headers):
    response = api_client.get("/anomaly/alerts", headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == [{"district": "Baidoa", "severity": "CRITICAL"}]


def test_narrative_generate_stream_endpoint(api_client, auth_headers, monkeypatch):
    async def fake_stream(_prediction, _alerts, _rules):
        for chunk in ["Part 1", " and Part 2"]:
            yield chunk

    monkeypatch.setattr(narrative_router, "generate_narrative_stream", fake_stream)

    response = api_client.post(
        "/narrative/generate",
        json={
            "prediction": {"district_pcode": "SO2501"},
            "alerts": [],
            "rules": {},
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.text == "Part 1 and Part 2"


def test_narrative_stream_generator_handles_llm_error(monkeypatch):
    async def failing_stream(_prompt):
        if False:
            yield ""
        raise RuntimeError("llm down")

    monkeypatch.setattr(narrative_router.ollama_client, "stream", failing_stream)

    async def collect():
        chunks = []
        async for c in narrative_router.generate_narrative_stream(
            prediction={"district_pcode": "SO2501"}, alerts=[], rules={}
        ):
            chunks.append(c)
        return chunks

    chunks = asyncio.run(collect())
    assert len(chunks) == 1
    assert "Error: llm down" in chunks[0]
