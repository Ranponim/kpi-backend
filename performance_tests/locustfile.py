"""
Locust 부하 테스트 스크립트

마할라노비스 분석 API의 성능과 확장성을 테스트합니다.
"""

import json
import random
from locust import HttpUser, task, between, constant
from typing import Dict, List, Any


class MahalanobisAnalysisUser(HttpUser):
    """마할라노비스 분석 API 사용자 시뮬레이션"""

    # 요청 간격 설정 (1-3초 사이)
    wait_time = between(1, 3)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_data_cache = {}

    def on_start(self):
        """사용자 세션 시작 시 초기화"""
        self.test_data_cache = self.generate_test_data()

    def generate_test_data(self) -> Dict[str, Any]:
        """테스트용 KPI 데이터 생성"""
        # 다양한 크기의 KPI 데이터 생성
        kpi_count = random.randint(3, 10)
        data_points = random.randint(5, 20)

        kpi_data = {}
        for i in range(kpi_count):
            kpi_name = f"KPI_{i+1}"
            # 랜덤한 KPI 값 생성 (정상 범위 + 약간의 변동성)
            base_value = random.uniform(80, 120)
            values = []

            for _ in range(data_points):
                # 약간의 노이즈 추가
                noise = random.uniform(-5, 5)
                value = max(0, min(200, base_value + noise))
                values.append(round(value, 2))

            kpi_data[kpi_name] = values

        return {
            "kpiData": kpi_data,
            "threshold": round(random.uniform(0.05, 0.2), 3),
            "sampleSize": random.randint(5, 20),
            "significanceLevel": round(random.uniform(0.01, 0.1), 3)
        }

    @task(3)  # 30% 확률
    def mahalanobis_analysis(self):
        """마할라노비스 분석 API 호출"""
        test_data = self.test_data_cache

        with self.client.post(
            "/api/analysis/mahalanobis",
            json=test_data,
            catch_response=True,
            name="마할라노비스 분석"
        ) as response:
            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("success") is True:
                    # 성공적인 응답
                    processing_time = response_data.get("processing_time", 0)
                    response.success()

                    # 처리 시간 로깅
                    self.environment.events.request.fire(
                        request_type="POST",
                        name="마할라노비스 분석",
                        response_time=processing_time * 1000,  # 초를 밀리초로 변환
                        response_length=len(json.dumps(response_data)),
                        exception=None,
                        context={},
                        url="/api/analysis/mahalanobis"
                    )
                else:
                    # 비즈니스 로직 실패
                    response.failure(f"분석 실패: {response_data.get('message', 'Unknown error')}")
            else:
                # HTTP 에러
                response.failure(f"HTTP {response.status_code}: {response.text}")

    @task(2)  # 20% 확률
    def quick_mahalanobis_analysis(self):
        """빠른 마할라노비스 분석 API 호출"""
        # 간단한 데이터로 빠른 분석 테스트
        quick_data = {
            "kpiData": {
                "Quick KPI": [100.0, 99.8, 100.2, 99.9, 100.1]
            }
        }

        with self.client.post(
            "/api/analysis/mahalanobis/quick",
            json=quick_data,
            catch_response=True,
            name="빠른 마할라노비스 분석"
        ) as response:
            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("success") is True:
                    response.success()
                else:
                    response.failure(f"빠른 분석 실패: {response_data.get('message')}")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

    @task(1)  # 10% 확률
    def statistical_tests(self):
        """통계 테스트 API 호출"""
        # Mann-Whitney U Test 데이터
        test_data = {
            "groupA": [random.uniform(90, 110) for _ in range(10)],
            "groupB": [random.uniform(85, 105) for _ in range(10)],
            "significanceLevel": 0.05
        }

        # Mann-Whitney U Test
        with self.client.post(
            "/api/statistical-tests/mann-whitney-u",
            json=test_data,
            catch_response=True,
            name="Mann-Whitney U Test"
        ) as response:
            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("success") is True:
                    response.success()
                else:
                    response.failure(f"Mann-Whitney 테스트 실패")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

        # Kolmogorov-Smirnov Test (동일한 데이터로)
        with self.client.post(
            "/api/statistical-tests/kolmogorov-smirnov",
            json=test_data,
            catch_response=True,
            name="Kolmogorov-Smirnov Test"
        ) as response:
            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("success") is True:
                    response.success()
                else:
                    response.failure(f"Kolmogorov-Smirnov 테스트 실패")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

    @task(1)  # 10% 확률
    def cache_operations(self):
        """캐시 관련 API 호출"""
        # 캐시 통계 조회
        with self.client.get(
            "/api/analysis/cache/stats",
            catch_response=True,
            name="캐시 통계 조회"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

    @task(1)  # 10% 확률
    def health_checks(self):
        """건강 상태 확인"""
        # 마할라노비스 분석 건강 상태
        with self.client.get(
            "/api/analysis/mahalanobis/health",
            catch_response=True,
            name="마할라노비스 건강 상태"
        ) as response:
            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure(f"서비스 상태: {response_data.get('status')}")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

        # 통계 테스트 건강 상태
        with self.client.get(
            "/api/statistical-tests/health",
            catch_response=True,
            name="통계 테스트 건강 상태"
        ) as response:
            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("status") in ["healthy", "degraded"]:
                    response.success()
                else:
                    response.failure(f"서비스 상태: {response_data.get('status')}")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

    @task(1)  # 10% 확률
    def error_scenarios(self):
        """에러 시나리오 테스트"""
        # 잘못된 데이터로 요청 (에러 처리 테스트)
        invalid_data = {
            "kpiData": {},  # 빈 데이터
            "threshold": 0.1,
            "sampleSize": 10,
            "significanceLevel": 0.05
        }

        with self.client.post(
            "/api/analysis/mahalanobis",
            json=invalid_data,
            catch_response=True,
            name="에러 시나리오 테스트"
        ) as response:
            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("success") is False:
                    # 예상된 에러 응답
                    response.success()
                else:
                    response.failure("예상치 못한 성공 응답")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")


class HighLoadUser(HttpUser):
    """고부하 시나리오용 사용자"""

    # 더 빈번한 요청 (0.5-1초 사이)
    wait_time = between(0.5, 1)

    @task
    def stress_test(self):
        """스트레스 테스트용 간단한 분석"""
        simple_data = {
            "kpiData": {
                "Stress KPI": [100.0, 99.8, 100.2]
            },
            "threshold": 0.1,
            "sampleSize": 5,
            "significanceLevel": 0.05
        }

        self.client.post("/api/analysis/mahalanobis", json=simple_data)


class CachePerformanceUser(HttpUser):
    """캐시 성능 테스트용 사용자"""

    wait_time = constant(0.1)  # 매우 빠른 요청

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 동일한 데이터로 캐시 효과 테스트
        self.test_data = {
            "kpiData": {
                "Cache KPI": [100.0, 99.8, 100.2, 99.9, 100.1]
            },
            "threshold": 0.1,
            "sampleSize": 10,
            "significanceLevel": 0.05
        }

    @task
    def cache_hit_test(self):
        """캐시 히트 테스트"""
        self.client.post("/api/analysis/mahalanobis", json=self.test_data)


