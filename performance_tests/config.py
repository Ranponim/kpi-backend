"""
부하 테스트 설정 파일

Locust 테스트의 다양한 시나리오와 설정을 정의합니다.
"""

import os
from typing import Dict, List, Any

# 테스트 환경 설정
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
TEST_DURATION = int(os.getenv("TEST_DURATION", "300"))  # 5분
SPAWN_RATE = int(os.getenv("SPAWN_RATE", "10"))  # 초당 사용자 증가 수
MAX_USERS = int(os.getenv("MAX_USERS", "100"))  # 최대 사용자 수

# 테스트 시나리오 설정
TEST_SCENARIOS = {
    "normal_load": {
        "user_class": "MahalanobisAnalysisUser",
        "users": 20,
        "spawn_rate": 5,
        "duration": 180,  # 3분
        "description": "일반적인 부하 테스트"
    },

    "high_load": {
        "user_class": "HighLoadUser",
        "users": 100,
        "spawn_rate": 20,
        "duration": 300,  # 5분
        "description": "고부하 스트레스 테스트"
    },

    "cache_performance": {
        "user_class": "CachePerformanceUser",
        "users": 50,
        "spawn_rate": 25,
        "duration": 120,  # 2분
        "description": "캐시 성능 테스트"
    },

    "spike_test": {
        "user_class": "HighLoadUser",
        "users": 200,
        "spawn_rate": 50,
        "duration": 60,  # 1분
        "description": "급격한 트래픽 증가 테스트"
    }
}

# 성능 기준치
PERFORMANCE_THRESHOLDS = {
    "response_time_95p": 2000,  # 95% 응답 시간 2초 이내
    "response_time_99p": 5000,  # 99% 응답 시간 5초 이내
    "error_rate": 0.05,        # 에러율 5% 이하
    "requests_per_second": 50, # 초당 50개 요청 처리
    "cpu_usage": 80,           # CPU 사용률 80% 이하
    "memory_usage": 85         # 메모리 사용률 85% 이하
}

# 모니터링 메트릭
MONITORING_METRICS = {
    "api_endpoints": [
        "/api/analysis/mahalanobis",
        "/api/analysis/mahalanobis/quick",
        "/api/statistical-tests/mann-whitney-u",
        "/api/statistical-tests/kolmogorov-smirnov"
    ],

    "system_metrics": [
        "cpu_percent",
        "memory_percent",
        "disk_usage",
        "network_io"
    ],

    "cache_metrics": [
        "cache_hit_rate",
        "cache_size",
        "cache_entries",
        "cache_memory_usage"
    ]
}

# 테스트 데이터 설정
TEST_DATA_CONFIG = {
    "kpi_ranges": {
        "normal": (90, 110),
        "warning": (80, 120),
        "critical": (70, 130)
    },

    "data_sizes": {
        "small": (3, 5),      # 3-5개 KPI, 각 5개 데이터 포인트
        "medium": (5, 10),    # 5-10개 KPI, 각 10개 데이터 포인트
        "large": (10, 20)     # 10-20개 KPI, 각 20개 데이터 포인트
    },

    "analysis_options": {
        "conservative": {
            "threshold": 0.05,
            "sample_size": 5,
            "significance_level": 0.01
        },
        "normal": {
            "threshold": 0.1,
            "sample_size": 10,
            "significance_level": 0.05
        },
        "aggressive": {
            "threshold": 0.2,
            "sample_size": 20,
            "significance_level": 0.1
        }
    }
}

# 보고서 설정
REPORT_CONFIG = {
    "output_dir": "performance_reports",
    "formats": ["html", "json", "csv"],
    "include_charts": True,
    "include_raw_data": False,

    "charts": {
        "response_times": {
            "title": "응답 시간 분포",
            "metrics": ["avg", "95p", "99p", "min", "max"]
        },
        "requests_per_second": {
            "title": "초당 요청 수",
            "time_window": "1s"
        },
        "error_rates": {
            "title": "에러율 추이",
            "time_window": "10s"
        },
        "cache_performance": {
            "title": "캐시 성능",
            "metrics": ["hit_rate", "size", "entries"]
        }
    }
}

def get_scenario_config(scenario_name: str) -> Dict[str, Any]:
    """테스트 시나리오 설정 반환"""
    return TEST_SCENARIOS.get(scenario_name, TEST_SCENARIOS["normal_load"])

def get_performance_thresholds() -> Dict[str, float]:
    """성능 기준치 반환"""
    return PERFORMANCE_THRESHOLDS.copy()

def generate_test_data(size: str = "medium", kpi_range: str = "normal") -> Dict[str, Any]:
    """테스트 데이터 생성"""
    import random

    kpi_min, kpi_max = TEST_DATA_CONFIG["kpi_ranges"][kpi_range]
    min_kpis, max_kpis = TEST_DATA_CONFIG["data_sizes"][size]

    kpi_count = random.randint(min_kpis, max_kpis)
    data_points = random.randint(5, 20)

    kpi_data = {}
    for i in range(kpi_count):
        kpi_name = f"Test_KPI_{i+1}"
        values = [round(random.uniform(kpi_min, kpi_max), 2) for _ in range(data_points)]
        kpi_data[kpi_name] = values

    return {
        "kpiData": kpi_data,
        "threshold": 0.1,
        "sampleSize": 10,
        "significanceLevel": 0.05
    }


