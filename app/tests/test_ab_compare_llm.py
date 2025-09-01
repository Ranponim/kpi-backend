import json
import time
import logging

from analysis_llm_old import _analyze_cell_performance_logic as run_old
from analysis_llm import _analyze_cell_performance_logic as run_new


def _build_sample_request():
    return {
        "db": {"host": "localhost", "port": 5432, "user": "postgres", "password": "pass", "dbname": "netperf", "table": "summary"},
        "n_minus_1": "2025-08-01_00:00~2025-08-01_23:59",
        "n": "2025-08-02_00:00~2025-08-02_23:59",
        "columns": [
            {"name": "peg_name"},
            {"name": "avg_value"}
        ],
        "enable_mock": True,
        # 상한 설정(새 로직 테스트를 위해 일부만 지정)
        "max_prompt_tokens": 20000,
        "max_prompt_chars": 60000,
        "max_rows_global": 1000,
        "max_selected_pegs": 50,
    }


def test_ab_compare_llm_paths():
    req = _build_sample_request()

    t0 = time.perf_counter()
    old_res = run_old(dict(req))
    old_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    new_res = run_new(dict(req))
    new_ms = (time.perf_counter() - t1) * 1000.0

    def size_bytes(obj):
        try:
            return len(json.dumps(obj, ensure_ascii=False).encode("utf-8"))
        except Exception:
            return -1

    old_size = size_bytes(old_res)
    new_size = size_bytes(new_res)

    logging.info("A/B 결과: old=%.1fms %dB, new=%.1fms %dB", old_ms, old_size, new_ms, new_size)

    # 기본 sanity check
    assert isinstance(new_res, dict)
    assert "status" in new_res


