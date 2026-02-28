from fastapi.testclient import TestClient

from sandbox_service import main as sandbox_main


client = TestClient(sandbox_main.app)


def setup_function() -> None:
    sandbox_main.ROUTE_TRACE_READ_AUTH = False
    sandbox_main.ROUTE_TRACE_API_KEY = ""
    sandbox_main.SANDBOX_API_KEY = ""
    with sandbox_main.TRACE_LOCK:
        sandbox_main.TRACE_STORE.clear()
        sandbox_main.TRACE_REQUEST_INDEX.clear()


def test_trace_endpoints_store_redact_and_render_dashboard() -> None:
    payload = {
        "trace_id": "trace-1",
        "request_id": "req-1",
        "status": "ok",
        "started_at_ts": 1000.0,
        "ended_at_ts": 1001.25,
        "total_latency_ms": 1250.0,
        "meta": {
            "api_key": "super-secret",
            "note": "safe",
        },
        "stages": [
            {
                "stage_key": "llm_json_call",
                "stage_name": "LLM Structured JSON Call",
                "purpose": "Test stage",
                "status": "ok",
                "input_summary": {"format": "json_object", "preview": "{}"},
                "output_summary": {"format": "json_object", "preview": "{}"},
                "processing_summary": "done",
                "details": {
                    "llm": {
                        "messages": [
                            {"role": "system", "content": "You are a test"},
                            {"role": "user", "content": "Bearer abc.def.ghi"},
                        ],
                        "raw_response": "sk-abcdefghijklmnopqrstuvwxyz123",
                    }
                },
            }
        ],
        "final": True,
    }

    upsert = client.post("/v1/traces/upsert", json=payload)
    assert upsert.status_code == 200
    assert upsert.json()["trace_id"] == "trace-1"

    get_resp = client.get("/v1/traces/trace-1")
    assert get_resp.status_code == 200
    trace = get_resp.json()
    assert trace["trace_id"] == "trace-1"
    assert trace["meta"]["api_key"] == "[REDACTED]"
    assert trace["stages"][0]["stage_index"] == 1
    llm_details = trace["stages"][0]["details"]["llm"]
    assert "Bearer [REDACTED]" in llm_details["messages"][1]["content"]
    assert "sk-[REDACTED]" in llm_details["raw_response"]

    latest_resp = client.get("/v1/traces/latest", params={"request_id": "req-1"})
    assert latest_resp.status_code == 200
    assert latest_resp.json()["trace_id"] == "trace-1"

    list_resp = client.get("/v1/traces", params={"limit": 1})
    assert list_resp.status_code == 200
    rows = list_resp.json()
    assert isinstance(rows, list)
    assert rows and rows[0]["trace_id"] == "trace-1"
    assert rows[0]["stage_count"] == 1

    dashboard_resp = client.get("/v1/traces/dashboard")
    assert dashboard_resp.status_code == 200
    assert "Data Route Dashboard" in dashboard_resp.text
