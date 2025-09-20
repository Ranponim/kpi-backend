"""
Choi 알고리즘 API 엔드포인트 테스트

이 모듈은 FastAPI를 통해 노출된 Choi 알고리즘 엔드포인트를
실제 HTTP 요청으로 테스트합니다.

Author: Choi Algorithm API Test Team
Created: 2025-09-20
"""

import pytest
import json
from datetime import datetime
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.main import app

# TestClient 생성
client = TestClient(app)


class TestChoiAlgorithmAPI:
    """
    Choi 알고리즘 API 엔드포인트 테스트 클래스
    
    FastAPI TestClient를 사용하여 실제 HTTP 요청을 시뮬레이션하고
    응답의 구조와 내용을 검증합니다.
    """
    
    def test_choi_analysis_endpoint_exists(self):
        """Choi 분석 엔드포인트 존재 확인"""
        # API 정보 엔드포인트로 Choi 분석 엔드포인트 확인
        response = client.get("/api/kpi/info")
        assert response.status_code == 200
        
        info_data = response.json()
        assert "choi_analysis" in info_data["endpoints"]
        assert "choi_algorithm" in info_data
        
        print("✅ Choi 분석 엔드포인트 존재 확인")
    
    def test_choi_analysis_basic_request(self):
        """기본 Choi 분석 요청 테스트"""
        # 기본 요청 데이터
        request_data = {
            "input_data": {
                "ems_ip": "192.168.1.100",
                "ne_list": ["NE001", "NE002"]
            },
            "cell_ids": ["cell_001", "cell_002"],
            "time_range": {
                "pre_start": "2025-09-20T10:00:00",
                "pre_end": "2025-09-20T11:00:00",
                "post_start": "2025-09-20T14:00:00",
                "post_end": "2025-09-20T15:00:00"
            },
            "compare_mode": True
        }
        
        # API 요청 실행
        response = client.post("/api/kpi/choi-analysis", json=request_data)
        
        # HTTP 상태 코드 검증
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        # 응답 JSON 파싱
        response_data = response.json()
        
        # 필수 필드 존재 확인
        required_fields = [
            "timestamp", "processing_time_ms", "algorithm_version",
            "filtering", "abnormal_detection", "kpi_judgement",
            "total_cells_analyzed", "total_pegs_analyzed"
        ]
        
        for field in required_fields:
            assert field in response_data, f"Required field '{field}' missing in response"
        
        # 기본 값 검증
        assert response_data["total_cells_analyzed"] == len(request_data["cell_ids"])
        assert response_data["algorithm_version"] == "1.0.0"
        assert isinstance(response_data["processing_time_ms"], (int, float))
        assert response_data["processing_time_ms"] > 0
        
        print(f"✅ 기본 Choi 분석 요청 성공: {response_data['processing_time_ms']:.2f}ms")
    
    def test_choi_analysis_filtering_results(self):
        """필터링 결과 구조 검증"""
        request_data = {
            "input_data": {"ems_ip": "192.168.1.101"},
            "cell_ids": ["cell_test"],
            "time_range": {"start": "2025-09-20T10:00:00"},
            "compare_mode": True
        }
        
        response = client.post("/api/kpi/choi-analysis", json=request_data)
        assert response.status_code == 200
        
        response_data = response.json()
        filtering = response_data["filtering"]
        
        # 필터링 결과 구조 검증
        assert "filter_ratio" in filtering
        assert "valid_time_slots" in filtering
        assert "median_values" in filtering
        assert "preprocessing_stats" in filtering
        
        # 필터링 비율 범위 검증
        assert 0.0 <= filtering["filter_ratio"] <= 1.0
        
        print(f"✅ 필터링 결과 검증: 비율 {filtering['filter_ratio']:.1%}")
    
    def test_choi_analysis_abnormal_detection_results(self):
        """이상 탐지 결과 구조 검증"""
        request_data = {
            "input_data": {"ems_ip": "192.168.1.102"},
            "cell_ids": ["cell_test_1", "cell_test_2"],
            "time_range": {"start": "2025-09-20T10:00:00"},
            "compare_mode": True
        }
        
        response = client.post("/api/kpi/choi-analysis", json=request_data)
        assert response.status_code == 200
        
        response_data = response.json()
        abnormal = response_data["abnormal_detection"]
        
        # 이상 탐지 결과 구조 검증
        required_anomaly_fields = [
            "range_violations", "new_statistics", "nd_anomalies", 
            "zero_anomalies", "high_delta_anomalies", "display_results"
        ]
        
        for field in required_anomaly_fields:
            assert field in abnormal, f"Abnormal detection field '{field}' missing"
        
        # α0 규칙 결과 검증
        display_results = abnormal["display_results"]
        expected_anomaly_types = ["Range", "ND", "Zero", "New", "High Delta"]
        
        for anomaly_type in expected_anomaly_types:
            assert anomaly_type in display_results
            assert isinstance(display_results[anomaly_type], bool)
        
        print(f"✅ 이상 탐지 결과 검증: {len(display_results)}개 탐지기")
    
    def test_choi_analysis_error_handling(self):
        """API 오류 처리 테스트"""
        # 필수 필드 누락 테스트
        invalid_request = {
            "input_data": {"ems_ip": "test"},
            # cell_ids 누락
            "time_range": {"start": "2025-09-20T10:00:00"}
        }
        
        response = client.post("/api/kpi/choi-analysis", json=invalid_request)
        assert response.status_code == 400
        
        error_data = response.json()
        assert "detail" in error_data
        assert "cell_ids" in error_data["detail"]
        
        print("✅ 필수 필드 누락 오류 처리 검증")
        
        # 잘못된 시간 형식 테스트
        invalid_time_request = {
            "input_data": {"ems_ip": "test"},
            "cell_ids": ["cell_001"],
            "time_range": {
                "pre_start": "invalid-time-format"
            }
        }
        
        response = client.post("/api/kpi/choi-analysis", json=invalid_time_request)
        assert response.status_code == 400
        
        error_data = response.json()
        assert "잘못된 시간 형식" in error_data["detail"]
        
        print("✅ 잘못된 시간 형식 오류 처리 검증")
    
    def test_choi_analysis_performance_via_api(self):
        """API를 통한 성능 테스트"""
        # 중간 규모 데이터로 성능 테스트
        request_data = {
            "input_data": {"ems_ip": "192.168.1.200"},
            "cell_ids": [f"cell_{i:03d}" for i in range(5)],  # 5개 셀
            "time_range": {"start": "2025-09-20T10:00:00"},
            "compare_mode": True
        }
        
        start_time = datetime.now()
        
        response = client.post("/api/kpi/choi-analysis", json=request_data)
        
        api_response_time = (datetime.now() - start_time).total_seconds() * 1000
        
        assert response.status_code == 200
        
        response_data = response.json()
        algorithm_processing_time = response_data["processing_time_ms"]
        
        # 성능 기준 검증
        assert api_response_time < 1000, f"API response time {api_response_time:.2f}ms exceeds 1s"
        assert algorithm_processing_time < 500, f"Algorithm time {algorithm_processing_time:.2f}ms exceeds 500ms"
        
        print(f"✅ API 성능 테스트: API {api_response_time:.2f}ms, 알고리즘 {algorithm_processing_time:.2f}ms")
    
    def test_choi_analysis_response_schema_validation(self):
        """응답 스키마 상세 검증"""
        request_data = {
            "input_data": {"ems_ip": "192.168.1.103"},
            "cell_ids": ["cell_schema_test"],
            "time_range": {"start": "2025-09-20T10:00:00"}
        }
        
        response = client.post("/api/kpi/choi-analysis", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # 메타데이터 검증
        assert isinstance(data["timestamp"], str)
        assert isinstance(data["processing_time_ms"], (int, float))
        assert isinstance(data["algorithm_version"], str)
        assert isinstance(data["total_cells_analyzed"], int)
        assert isinstance(data["total_pegs_analyzed"], int)
        assert isinstance(data["processing_warnings"], list)
        
        # 필터링 결과 상세 검증
        filtering = data["filtering"]
        assert isinstance(filtering["filter_ratio"], (int, float))
        assert isinstance(filtering["valid_time_slots"], dict)
        assert isinstance(filtering["median_values"], dict)
        assert isinstance(filtering["preprocessing_stats"], dict)
        
        # 이상 탐지 결과 상세 검증
        abnormal = data["abnormal_detection"]
        assert isinstance(abnormal["range_violations"], dict)
        assert isinstance(abnormal["nd_anomalies"], dict)
        assert isinstance(abnormal["zero_anomalies"], dict)
        assert isinstance(abnormal["high_delta_anomalies"], dict)
        assert isinstance(abnormal["display_results"], dict)
        
        # KPI 판정 결과 검증
        assert isinstance(data["kpi_judgement"], dict)
        assert isinstance(data["ui_summary"], dict)
        assert isinstance(data["config_used"], dict)
        
        print("✅ 응답 스키마 상세 검증 완료")


# =============================================================================
# API 성능 및 부하 테스트
# =============================================================================

class TestChoiAPIPerformance:
    """Choi 알고리즘 API 성능 테스트"""
    
    def test_concurrent_requests_simulation(self):
        """동시 요청 시뮬레이션 테스트"""
        import threading
        import time
        
        results = []
        
        def make_request(thread_id):
            """스레드별 요청 실행"""
            request_data = {
                "input_data": {"ems_ip": f"192.168.1.{200 + thread_id}"},
                "cell_ids": [f"cell_{thread_id:03d}"],
                "time_range": {"start": "2025-09-20T10:00:00"}
            }
            
            start = time.time()
            response = client.post("/api/kpi/choi-analysis", json=request_data)
            duration = (time.time() - start) * 1000
            
            results.append({
                "thread_id": thread_id,
                "status_code": response.status_code,
                "duration_ms": duration,
                "success": response.status_code == 200
            })
        
        # 5개 동시 요청 실행
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
        
        # 모든 스레드 시작
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # 모든 스레드 완료 대기
        for thread in threads:
            thread.join()
        
        total_time = (time.time() - start_time) * 1000
        
        # 결과 검증
        success_count = sum(1 for r in results if r["success"])
        avg_duration = sum(r["duration_ms"] for r in results) / len(results)
        
        assert success_count == 5, f"Expected 5 successful requests, got {success_count}"
        assert avg_duration < 1000, f"Average response time {avg_duration:.2f}ms exceeds 1s"
        assert total_time < 2000, f"Total concurrent time {total_time:.2f}ms exceeds 2s"
        
        print(f"✅ 동시 요청 테스트: {success_count}/5 성공, 평균 {avg_duration:.2f}ms")
    
    def test_large_cell_list_api_performance(self):
        """대용량 셀 리스트 API 성능 테스트"""
        # 20개 셀로 대용량 테스트
        request_data = {
            "input_data": {"ems_ip": "192.168.1.300"},
            "cell_ids": [f"cell_{i:03d}" for i in range(20)],
            "time_range": {"start": "2025-09-20T10:00:00"}
        }
        
        start_time = datetime.now()
        
        response = client.post("/api/kpi/choi-analysis", json=request_data)
        
        api_time = (datetime.now() - start_time).total_seconds() * 1000
        
        assert response.status_code == 200
        
        data = response.json()
        algorithm_time = data["processing_time_ms"]
        
        # 성능 기준: 20셀 × 100ms = 2초 이하
        assert api_time < 2000, f"API time {api_time:.2f}ms exceeds 2s for 20 cells"
        assert algorithm_time < 1500, f"Algorithm time {algorithm_time:.2f}ms exceeds 1.5s"
        
        # 결과 무결성 검증
        assert data["total_cells_analyzed"] == 20
        assert data["total_pegs_analyzed"] > 0
        
        print(f"✅ 대용량 API 성능: API {api_time:.2f}ms, 알고리즘 {algorithm_time:.2f}ms")


# =============================================================================
# 실제 시나리오 기반 API 테스트
# =============================================================================

class TestChoiAPIScenarios:
    """실제 시나리오 기반 API 테스트"""
    
    def test_normal_scenario_via_api(self):
        """정상 시나리오 API 테스트"""
        request_data = {
            "input_data": {
                "ems_ip": "192.168.1.100",
                "ne_list": ["NE001", "NE002"]
            },
            "cell_ids": ["cell_001", "cell_002"],
            "time_range": {
                "pre_start": "2025-09-20T10:00:00",
                "pre_end": "2025-09-20T11:00:00",
                "post_start": "2025-09-20T14:00:00",
                "post_end": "2025-09-20T15:00:00"
            }
        }
        
        response = client.post("/api/kpi/choi-analysis", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # 6장 필터링 결과 검증
        filtering = data["filtering"]
        assert 0.0 <= filtering["filter_ratio"] <= 1.0
        assert isinstance(filtering["valid_time_slots"], dict)
        
        # 4장 이상 탐지 결과 검증
        abnormal = data["abnormal_detection"]
        assert isinstance(abnormal["display_results"], dict)
        assert len(abnormal["display_results"]) == 5  # 5개 탐지기
        
        print("✅ 정상 시나리오 API 테스트 성공")
    
    def test_fifty_percent_rule_via_api(self):
        """50% 규칙 트리거 API 테스트"""
        # 극심한 변동 데이터로 50% 규칙 트리거 시도
        request_data = {
            "input_data": {"ems_ip": "192.168.1.104"},
            "cell_ids": ["cell_extreme_variation"],
            "time_range": {"start": "2025-09-20T10:00:00"}
        }
        
        response = client.post("/api/kpi/choi-analysis", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # 필터링 결과에서 경고 메시지 확인 (50% 규칙 트리거 시)
        filtering = data["filtering"]
        
        # 필터링 비율이 낮거나 경고 메시지가 있을 수 있음
        if filtering["filter_ratio"] <= 0.5:
            assert filtering["warning_message"] is not None
            print(f"✅ 50% 규칙 트리거 확인: 비율 {filtering['filter_ratio']:.1%}")
        else:
            print(f"✅ 정상 필터링: 비율 {filtering['filter_ratio']:.1%}")


# =============================================================================
# 직접 실행 (pytest 없이)
# =============================================================================

def run_api_tests_directly():
    """API 테스트 직접 실행"""
    print("🌐 Choi 알고리즘 API 테스트 직접 실행")
    print("=" * 50)
    
    try:
        # 기본 테스트 클래스 인스턴스 생성
        basic_tests = TestChoiAlgorithmAPI()
        performance_tests = TestChoiAPIPerformance()
        scenario_tests = TestChoiAPIScenarios()
        
        # 기본 테스트들
        print("1. 기본 API 테스트:")
        basic_tests.test_choi_analysis_endpoint_exists()
        basic_tests.test_choi_analysis_basic_request()
        basic_tests.test_choi_analysis_filtering_results()
        basic_tests.test_choi_analysis_abnormal_detection_results()
        basic_tests.test_choi_analysis_error_handling()
        
        # 성능 테스트들
        print("\n2. 성능 테스트:")
        performance_tests.test_large_cell_list_api_performance()
        
        # 시나리오 테스트들
        print("\n3. 시나리오 테스트:")
        scenario_tests.test_normal_scenario_via_api()
        scenario_tests.test_fifty_percent_rule_via_api()
        
        print("\n🎉 모든 API 테스트 성공!")
        print("🏆 Choi 알고리즘 API 엔드포인트 완전 검증!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ API 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_api_tests_directly()
    if not success:
        sys.exit(1)
