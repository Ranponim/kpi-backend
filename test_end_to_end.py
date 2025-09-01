"""
간단 E2E 테스트 스크립트
- 백엔드 라이브 서버(http://localhost:8000)가 실행 중이어야 합니다.
- 주요 엔드포인트 루트/단건/배치/리포트/프리퍼런스 흐름을 검증합니다.

실행:
  python -m kpi_dashboard.backend.test_end_to_end
"""

import json
import time
import logging
from typing import Dict, Any
import requests

BASE = "http://localhost:8000"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _assert(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def test_root():
    logging.info("[TEST] GET /")
    r = requests.get(f"{BASE}/")
    _assert(r.status_code == 200, f"root status {r.status_code}")
    data = r.json()
    _assert(data.get("message") == "3GPP KPI Management API", "root message mismatch")


def test_statistics_and_trends():
    start = "2025-08-06"
    end = "2025-08-07"
    logging.info("[TEST] GET /api/kpi/statistics")
    r = requests.get(
        f"{BASE}/api/kpi/statistics",
        params={
            "start_date": start,
            "end_date": end,
            "kpi_type": "availability",
            "entity_ids": "LHK078ML1,LHK078MR1",
            "interval_minutes": 60,
        },
    )
    _assert(r.status_code == 200, f"statistics status {r.status_code}")
    rows = r.json().get("data") or []
    _assert(len(rows) > 0, "statistics returned 0 rows")

    logging.info("[TEST] GET /api/kpi/trends")
    r2 = requests.get(
        f"{BASE}/api/kpi/trends",
        params={
            "start_date": start,
            "end_date": end,
            "kpi_type": "availability",
            "entity_id": "LHK078ML1",
            "interval_minutes": 60,
        },
    )
    _assert(r2.status_code == 200, f"trends status {r2.status_code}")
    rows2 = r2.json().get("data") or []
    _assert(len(rows2) > 0, "trends returned 0 rows")


def test_batch_statistics():
    payload: Dict[str, Any] = {
        "start_date": "2025-08-06",
        "end_date": "2025-08-07",
        "kpi_types": ["availability", "rrc", "erab"],
        "entity_ids": "LHK078ML1,LHK078MR1",
        "interval_minutes": 60,
    }
    logging.info("[TEST] POST /api/kpi/statistics/batch")
    r = requests.post(f"{BASE}/api/kpi/statistics/batch", json=payload)
    _assert(r.status_code == 200, f"batch status {r.status_code}")
    data = r.json().get("data") or {}
    _assert(set(data.keys()) == set(payload["kpi_types"]), "batch keys mismatch")
    total = sum(len(v) for v in data.values())
    _assert(total > 0, "batch returned 0 rows")


def test_preferences_and_derived_pegs():
    logging.info("[TEST] POST /api/preferences")
    pref = {
        "name": "E2E Test Pref",
        "description": "for e2e",
        "config": {
            "defaultKPIs": ["availability", "rrc"],
            "defaultNEs": ["nvgnb#10000", "nvgnb#20000"],
            "defaultCellIDs": ["2010", "2011"],
            "availableKPIs": [
                {"value": "availability", "label": "Availability (%)", "threshold": 99.0},
                {"value": "rrc", "label": "RRC Success Rate (%)", "threshold": 98.5},
            ],
        },
    }
    r = requests.post(f"{BASE}/api/preferences", json=pref)
    _assert(r.status_code == 200, f"create pref status {r.status_code}")
    pref_id = r.json().get("id")
    _assert(pref_id is not None, "pref id missing")

    logging.info("[TEST] GET /api/preferences")
    r2 = requests.get(f"{BASE}/api/preferences")
    _assert(r2.status_code == 200, f"get prefs status {r2.status_code}")

    logging.info("[TEST] PUT /api/preferences/{id}/derived-pegs")
    derived = {"telus_RACH_Success": "Random_access_preamble_count/Random_access_response*100"}
    r3 = requests.put(f"{BASE}/api/preferences/{pref_id}/derived-pegs", json={"derived_pegs": derived})
    _assert(r3.status_code == 200, f"put derived status {r3.status_code}")

    logging.info("[TEST] GET /api/preferences/{id}/derived-pegs")
    r4 = requests.get(f"{BASE}/api/preferences/{pref_id}/derived-pegs")
    _assert(r4.status_code == 200, f"get derived status {r4.status_code}")
    _assert("derived_pegs" in r4.json(), "derived_pegs missing")

    # Update preference (name/description/config)
    logging.info("[TEST] PUT /api/preferences/{id}")
    updated = {
        "name": "E2E Test Pref - Updated",
        "description": "updated",
        "config": {
            "defaultKPIs": ["availability"],
            "defaultNEs": ["nvgnb#30000"],
            "defaultCellIDs": ["2012"],
            "availableKPIs": [
                {"value": "availability", "label": "Availability (%)", "threshold": 99.0}
            ],
            "kpiMappings": {"availability": {"peg_like": ["Access_%"]}}
        }
    }
    r5 = requests.put(f"{BASE}/api/preferences/{pref_id}", json=updated)
    _assert(r5.status_code == 200, f"update pref status {r5.status_code}")

    # Verify get returns updated values
    r6 = requests.get(f"{BASE}/api/preferences/{pref_id}")
    _assert(r6.status_code == 200, f"get updated pref status {r6.status_code}")
    body = r6.json()
    _assert(body.get("name") == updated["name"], "updated name mismatch")
    _assert(body.get("description") == updated["description"], "updated description mismatch")
    cfg = body.get("config") or {}
    _assert(cfg.get("kpiMappings", {}).get("availability") is not None, "updated config not applied")

    # Delete preference
    logging.info("[TEST] DELETE /api/preferences/{id}")
    r7 = requests.delete(f"{BASE}/api/preferences/{pref_id}")
    _assert(r7.status_code == 200, f"delete pref status {r7.status_code}")

    # Ensure not found after delete
    r8 = requests.get(f"{BASE}/api/preferences/{pref_id}")
    _assert(r8.status_code == 404, f"expected 404 after delete, got {r8.status_code}")


def test_analysis_result_pipeline_mock():
    """LLM 외부 없이, mock 데이터 기반으로 분석 결과를 저장하는 시나리오 검증."""
    logging.info("[TEST] POST /api/analysis-result (mock payload)")
    payload = {
        "status": "success",
        "n_minus_1": "2025-08-06",
        "n": "2025-08-07",
        "analysis": {"executive_summary": "요약...", "recommended_actions": ["조치1"]},
        "stats": [{"peg_name": "A", "avg_n_minus_1": 1.0, "avg_n": 2.0, "diff": 1.0, "pct_change": 100.0}],
        "chart_overall_base64": None,
        "report_path": "./analysis_output_test/Cell_Analysis_Report_mock.html",
    }
    r = requests.post(f"{BASE}/api/analysis-result", json=payload)
    _assert(r.status_code == 200, f"analysis post status {r.status_code}")
    new_id = r.json().get("id")
    _assert(new_id is not None, "analysis id missing")

    logging.info("[TEST] GET /api/analysis-result/latest")
    r2 = requests.get(f"{BASE}/api/analysis-result/latest")
    _assert(r2.status_code == 200, f"latest status {r2.status_code}")
    _assert(r2.json().get("id") == new_id, "latest id mismatch")


def run_all():
    test_root()
    test_statistics_and_trends()
    test_batch_statistics()
    test_preferences_and_derived_pegs()
    test_analysis_result_pipeline_mock()
    logging.info("[TEST] ALL PASSED")


if __name__ == "__main__":
    run_all()


