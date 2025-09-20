"""
API 엔드포인트 통합 테스트

이 모듈은 FastAPI 엔드포인트들의 정확성과 안정성을
보장하기 위한 포괄적인 통합 테스트를 제공합니다.
"""

import pytest
import json
from typing import Dict, Any
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from app.main import app
from app.models.mahalanobis import (
    KpiDataInput, AnalysisOptionsInput, StatisticalTestInput
)
import logging

# 로거 설정
logger = logging.getLogger(__name__)


@pytest.fixture
def client():
    """테스트용 FastAPI 클라이언트"""
    return TestClient(app)


@pytest.fixture
def sample_kpi_data():
    """테스트용 KPI 데이터"""
    return {
        "kpiData": {
            "RACH Success Rate": [98.5, 97.2, 99.1, 96.8, 98.3],
            "RLC DL Throughput": [45.2, 42.1, 46.8, 43.5, 44.9],
            "Normal KPI": [100.0, 99.8, 100.2, 99.9, 100.1]
        },
        "timestamps": [
            "2024-01-01T10:00:00Z",
            "2024-01-01T11:00:00Z",
            "2024-01-01T12:00:00Z",
            "2024-01-01T13:00:00Z",
            "2024-01-01T14:00:00Z"
        ],
        "periodLabels": ["N-1", "N-2", "N-3", "N-4", "N-5"]
    }


@pytest.fixture
def analysis_options():
    """테스트용 분석 옵션"""
    return {
        "threshold": 0.1,
        "sampleSize": 10,
        "significanceLevel": 0.05
    }


class TestMahalanobisAPI:
    """마할라노비스 분석 API 테스트"""

    def test_mahalanobis_analysis_success(self, client, sample_kpi_data, analysis_options):
        """마할라노비스 분석 성공 테스트"""
        request_data = {
            "kpiData": sample_kpi_data["kpiData"],
            "timestamps": sample_kpi_data["timestamps"],
            "periodLabels": sample_kpi_data["periodLabels"],
            "threshold": analysis_options["threshold"],
            "sampleSize": analysis_options["sampleSize"],
            "significanceLevel": analysis_options["significanceLevel"]
        }

        response = client.post("/api/analysis/mahalanobis", json=request_data)

        assert response.status_code == 200

        response_data = response.json()
        assert response_data["success"] is True
        assert "data" in response_data
        assert "processing_time" in response_data
        assert response_data["processing_time"] > 0

        # 결과 구조 검증
        data = response_data["data"]
        assert "totalKpis" in data
        assert "abnormalKpis" in data
        assert "abnormalScore" in data
        assert "alarmLevel" in data
        assert "analysis" in data

    def test_mahalanobis_analysis_invalid_data(self, client):
        """잘못된 데이터로 마할라노비스 분석 테스트"""
        # 빈 KPI 데이터
        request_data = {
            "kpiData": {},
            "threshold": 0.1,
            "sampleSize": 10,
            "significanceLevel": 0.05
        }

        response = client.post("/api/analysis/mahalanobis", json=request_data)

        assert response.status_code == 200  # 현재는 200으로 실패 응답 반환

        response_data = response.json()
        assert response_data["success"] is False
        assert "비어있습니다" in response_data["message"]

    def test_mahalanobis_quick_analysis(self, client, sample_kpi_data):
        """빠른 마할라노비스 분석 테스트"""
        request_data = {
            "kpiData": sample_kpi_data["kpiData"],
            "threshold": 0.05,  # 더 엄격한 임계값
            "sampleSize": 5,    # 더 작은 샘플 크기
            "significanceLevel": 0.01  # 더 엄격한 유의 수준
        }

        response = client.post("/api/analysis/mahalanobis", json=request_data)

        assert response.status_code == 200

        response_data = response.json()
        assert response_data["success"] is True
        assert response_data["processing_time"] > 0

    def test_mahalanobis_analysis_with_abnormal_data(self, client):
        """이상 데이터로 마할라노비스 분석 테스트"""
        abnormal_kpi_data = {
            "kpiData": {
                "RACH Success Rate": [98.5, 85.2],  # 큰 변화
                "RLC DL Throughput": [45.2, 25.1], # 큰 변화
                "Normal KPI": [100.0, 101.0]       # 정상 범위
            },
            "threshold": 0.05,
            "sampleSize": 10,
            "significanceLevel": 0.05
        }

        response = client.post("/api/analysis/mahalanobis", json=abnormal_kpi_data)

        assert response.status_code == 200

        response_data = response.json()
        assert response_data["success"] is True

        # 이상 KPI가 감지되었는지 확인
        data = response_data["data"]
        if len(data["abnormalKpis"]) > 0:
            assert data["alarmLevel"] in ["caution", "warning", "critical"]

    def test_mahalanobis_info_endpoint(self, client):
        """마할라노비스 분석 정보 엔드포인트 테스트"""
        response = client.get("/api/analysis/mahalanobis/info")

        assert response.status_code == 200

        response_data = response.json()
        assert "service" in response_data
        assert "version" in response_data
        assert "description" in response_data

    def test_mahalanobis_health_endpoint(self, client):
        """마할라노비스 분석 상태 엔드포인트 테스트"""
        response = client.get("/api/analysis/mahalanobis/health")

        assert response.status_code == 200

        response_data = response.json()
        assert "status" in response_data
        assert response_data["status"] in ["healthy", "unhealthy"]


class TestStatisticalTestsAPI:
    """통계 테스트 API 테스트"""

    @pytest.fixture
    def sample_statistical_test_data(self):
        """테스트용 통계 테스트 데이터"""
        return {
            "groupA": [98.5, 97.8, 99.2, 96.7, 98.1],
            "groupB": [85.2, 87.3, 84.9, 86.1, 85.8],
            "significanceLevel": 0.05
        }

    def test_mann_whitney_u_test_success(self, client, sample_statistical_test_data):
        """Mann-Whitney U Test 성공 테스트"""
        response = client.post("/api/statistical-tests/mann-whitney-u", json=sample_statistical_test_data)

        assert response.status_code == 200

        response_data = response.json()
        assert response_data["success"] is True
        assert "result" in response_data
        assert "processing_time" in response_data

        result = response_data["result"]
        assert result["test_name"] == "Mann-Whitney U"
        assert isinstance(result["significant"], bool)
        assert isinstance(result["p_value"], float)
        assert 0 <= result["p_value"] <= 1

    def test_kolmogorov_smirnov_test_success(self, client, sample_statistical_test_data):
        """Kolmogorov-Smirnov Test 성공 테스트"""
        response = client.post("/api/statistical-tests/kolmogorov-smirnov", json=sample_statistical_test_data)

        assert response.status_code == 200

        response_data = response.json()
        assert response_data["success"] is True
        assert "result" in response_data
        assert "processing_time" in response_data

        result = response_data["result"]
        assert result["test_name"] == "Kolmogorov-Smirnov"
        assert isinstance(result["significant"], bool)
        assert isinstance(result["p_value"], float)
        assert 0 <= result["p_value"] <= 1

    def test_batch_mann_whitney_u_test(self, client, sample_statistical_test_data):
        """배치 Mann-Whitney U Test 테스트"""
        batch_data = {
            "test1": sample_statistical_test_data,
            "test2": {
                "groupA": [100.0, 99.5, 100.2, 99.8, 100.1],
                "groupB": [98.5, 99.2, 98.8, 99.1, 98.7],
                "significanceLevel": 0.05
            }
        }

        response = client.post("/api/statistical-tests/batch/mann-whitney-u", json=batch_data)

        assert response.status_code == 200

        response_data = response.json()
        assert response_data["success"] is True
        assert "batch_summary" in response_data
        assert "results" in response_data

        batch_summary = response_data["batch_summary"]
        assert batch_summary["total_cases"] == 2
        assert batch_summary["successful_cases"] <= 2
        assert batch_summary["failed_cases"] >= 0

    def test_compare_statistical_tests(self, client, sample_statistical_test_data):
        """통계 테스트 비교 API 테스트"""
        response = client.post("/api/statistical-tests/compare-tests", json=sample_statistical_test_data)

        assert response.status_code == 200

        response_data = response.json()
        assert response_data["success"] is True
        assert "comparison_summary" in response_data
        assert "mann_whitney_u" in response_data
        assert "kolmogorov_smirnov" in response_data

        comparison_summary = response_data["comparison_summary"]
        assert "consistency" in comparison_summary
        assert "confidence" in comparison_summary
        assert "recommendation" in comparison_summary

    def test_statistical_tests_info_endpoint(self, client):
        """통계 테스트 정보 엔드포인트 테스트"""
        response = client.get("/api/statistical-tests/info")

        assert response.status_code == 200

        response_data = response.json()
        assert "service_name" in response_data
        assert "version" in response_data
        assert "supported_tests" in response_data

    def test_statistical_tests_health_endpoint(self, client):
        """통계 테스트 상태 엔드포인트 테스트"""
        response = client.get("/api/statistical-tests/health")

        assert response.status_code == 200

        response_data = response.json()
        assert "status" in response_data
        assert response_data["status"] in ["healthy", "degraded", "unhealthy"]

    def test_statistical_tests_with_identical_data(self, client):
        """동일한 데이터로 통계 테스트"""
        identical_data = {
            "groupA": [98.5, 97.8, 99.2, 96.7, 98.1],
            "groupB": [98.5, 97.8, 99.2, 96.7, 98.1],  # 동일한 데이터
            "significanceLevel": 0.05
        }

        # Mann-Whitney U Test
        mw_response = client.post("/api/statistical-tests/mann-whitney-u", json=identical_data)
        assert mw_response.status_code == 200
        mw_data = mw_response.json()
        assert mw_data["success"] is True
        assert mw_data["result"]["significant"] is False  # 유의한 차이가 없어야 함

        # Kolmogorov-Smirnov Test
        ks_response = client.post("/api/statistical-tests/kolmogorov-smirnov", json=identical_data)
        assert ks_response.status_code == 200
        ks_data = ks_response.json()
        assert ks_data["success"] is True
        assert ks_data["result"]["significant"] is False  # 유의한 차이가 없어야 함

    def test_statistical_tests_error_handling(self, client):
        """통계 테스트 에러 처리 테스트"""
        # 잘못된 데이터
        invalid_data = {
            "groupA": [],  # 빈 그룹
            "groupB": [1.0, 2.0, 3.0],
            "significanceLevel": 0.05
        }

        response = client.post("/api/statistical-tests/mann-whitney-u", json=invalid_data)

        # 현재 구현에서는 200으로 실패 응답을 반환하지만,
        # 실제로는 422(Unprocessable Entity)를 반환하는 것이 더 적절할 수 있음
        assert response.status_code in [200, 422]

        if response.status_code == 200:
            response_data = response.json()
            assert response_data["success"] is False


class TestAPIIntegration:
    """API 통합 테스트"""

    def test_full_workflow_mahalanobis_and_statistical_tests(self, client, sample_kpi_data):
        """마할라노비스 분석과 통계 테스트의 전체 워크플로우 테스트"""
        # 1. 마할라노비스 분석 수행
        mahalanobis_request = {
            "kpiData": sample_kpi_data["kpiData"],
            "threshold": 0.1,
            "sampleSize": 10,
            "significanceLevel": 0.05
        }

        mahalanobis_response = client.post("/api/analysis/mahalanobis", json=mahalanobis_request)
        assert mahalanobis_response.status_code == 200

        mahalanobis_data = mahalanobis_response.json()
        assert mahalanobis_data["success"] is True

        # 2. 얻은 데이터로 독립적인 통계 테스트 수행
        statistical_test_request = {
            "groupA": sample_kpi_data["kpiData"]["RACH Success Rate"],
            "groupB": sample_kpi_data["kpiData"]["RLC DL Throughput"],
            "significanceLevel": 0.05
        }

        # Mann-Whitney U Test
        mw_response = client.post("/api/statistical-tests/mann-whitney-u", json=statistical_test_request)
        assert mw_response.status_code == 200

        # Kolmogorov-Smirnov Test
        ks_response = client.post("/api/statistical-tests/kolmogorov-smirnov", json=statistical_test_request)
        assert ks_response.status_code == 200

        # 3. 두 통계 테스트 비교
        compare_response = client.post("/api/statistical-tests/compare-tests", json=statistical_test_request)
        assert compare_response.status_code == 200

        # 모든 응답 검증
        mw_data = mw_response.json()
        ks_data = ks_response.json()
        compare_data = compare_response.json()

        assert mw_data["success"] is True
        assert ks_data["success"] is True
        assert compare_data["success"] is True

        # 비교 결과 검증
        assert "mann_whitney_u" in compare_data
        assert "kolmogorov_smirnov" in compare_data
        assert "comparison_summary" in compare_data

    def test_api_performance(self, client, sample_kpi_data):
        """API 성능 테스트"""
        import time

        mahalanobis_request = {
            "kpiData": sample_kpi_data["kpiData"],
            "threshold": 0.1,
            "sampleSize": 10,
            "significanceLevel": 0.05
        }

        # 5회 반복 테스트
        response_times = []
        for _ in range(5):
            start_time = time.time()
            response = client.post("/api/analysis/mahalanobis", json=mahalanobis_request)
            end_time = time.time()

            assert response.status_code == 200
            response_times.append(end_time - start_time)

        # 평균 응답 시간 검증 (2초 이내)
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 2.0

        logger.info(f"API 평균 응답 시간: {avg_response_time:.3f}초")

    def test_api_error_responses(self, client):
        """API 에러 응답 테스트"""
        # 잘못된 JSON
        response = client.post("/api/analysis/mahalanobis", data="invalid json")
        assert response.status_code == 400

        # 잘못된 엔드포인트
        response = client.get("/api/nonexistent/endpoint")
        assert response.status_code == 404

        # 잘못된 HTTP 메소드
        response = client.put("/api/analysis/mahalanobis", json={})
        assert response.status_code == 405


class TestAPICaching:
    """API 캐싱 테스트"""

    def test_mahalanobis_caching(self, client, sample_kpi_data, analysis_options):
        """마할라노비스 분석 캐싱 테스트"""
        import time

        request_data = {
            "kpiData": sample_kpi_data["kpiData"],
            "timestamps": sample_kpi_data["timestamps"],
            "periodLabels": sample_kpi_data["periodLabels"],
            "threshold": analysis_options["threshold"],
            "sampleSize": analysis_options["sampleSize"],
            "significanceLevel": analysis_options["significanceLevel"]
        }

        # 첫 번째 요청
        start_time = time.time()
        response1 = client.post("/api/analysis/mahalanobis", json=request_data)
        time1 = time.time() - start_time

        assert response1.status_code == 200
        data1 = response1.json()

        # 두 번째 요청 (캐시 히트)
        start_time = time.time()
        response2 = client.post("/api/analysis/mahalanobis", json=request_data)
        time2 = time.time() - start_time

        assert response2.status_code == 200
        data2 = response2.json()

        # 결과가 동일해야 함
        assert data1["success"] == data2["success"]
        assert data1["data"]["totalKpis"] == data2["data"]["totalKpis"]

        # 두 번째 요청이 더 빠르거나 비슷해야 함
        assert time2 <= time1 * 1.5  # 50% 이내 오차 허용

        logger.info(f"캐시 테스트 - 첫 번째: {time1:.4f}초, 두 번째: {time2:.4f}초")

    def test_statistical_tests_caching(self, client):
        """통계 테스트 캐싱 테스트"""
        import time

        request_data = {
            "groupA": [98.5, 97.8, 99.2, 96.7, 98.1],
            "groupB": [85.2, 87.3, 84.9, 86.1, 85.8],
            "significanceLevel": 0.05
        }

        # Mann-Whitney U Test 캐싱 테스트
        start_time = time.time()
        response1 = client.post("/api/statistical-tests/mann-whitney-u", json=request_data)
        time1 = time.time() - start_time

        start_time = time.time()
        response2 = client.post("/api/statistical-tests/mann-whitney-u", json=request_data)
        time2 = time.time() - start_time

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        # 결과가 동일해야 함
        assert data1["result"]["significant"] == data2["result"]["significant"]
        assert abs(data1["result"]["p_value"] - data2["result"]["p_value"]) < 1e-10

        logger.info(f"Mann-Whitney 캐시 테스트 - 첫 번째: {time1:.4f}초, 두 번째: {time2:.4f}초")


if __name__ == "__main__":
    # 테스트 실행
    pytest.main([__file__, "-v", "--tb=short"])


