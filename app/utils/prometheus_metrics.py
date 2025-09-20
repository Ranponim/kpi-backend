"""
Prometheus 메트릭 수집 모듈

마할라노비스 분석 API의 성능 메트릭을 수집하고 Prometheus에 노출합니다.
"""

import time
import logging
from typing import Dict, Any, Optional
from prometheus_client import (
    Counter, Histogram, Gauge, CollectorRegistry,
    generate_latest, CONTENT_TYPE_LATEST
)
from fastapi import Response
import psutil
import os

logger = logging.getLogger(__name__)

# 사용자 정의 레지스트리 생성 (기본 레지스트리와 충돌 방지)
registry = CollectorRegistry()

# API 요청 메트릭
REQUEST_COUNT = Counter(
    'mahalanobis_api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

REQUEST_LATENCY = Histogram(
    'mahalanobis_api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint'],
    registry=registry,
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
)

# 마할라노비스 분석 메트릭
ANALYSIS_COUNT = Counter(
    'mahalanobis_analysis_total',
    'Total number of mahalanobis analysis operations',
    ['status'],
    registry=registry
)

ANALYSIS_DURATION = Histogram(
    'mahalanobis_analysis_duration_seconds',
    'Mahalanobis analysis duration in seconds',
    registry=registry,
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
)

CACHE_HITS = Counter(
    'mahalanobis_cache_hits_total',
    'Total number of cache hits',
    registry=registry
)

CACHE_MISSES = Counter(
    'mahalanobis_cache_misses_total',
    'Total number of cache misses',
    registry=registry
)

# KPI 메트릭
KPI_COUNT = Histogram(
    'mahalanobis_kpi_count',
    'Number of KPIs processed',
    registry=registry,
    buckets=(1, 5, 10, 25, 50, 100, 200)
)

ABNORMAL_KPI_COUNT = Histogram(
    'mahalanobis_abnormal_kpi_count',
    'Number of abnormal KPIs detected',
    registry=registry,
    buckets=(0, 1, 5, 10, 25, 50, 100)
)

# 시스템 메트릭
SYSTEM_CPU_USAGE = Gauge(
    'mahalanobis_system_cpu_usage_percent',
    'System CPU usage percentage',
    registry=registry
)

SYSTEM_MEMORY_USAGE = Gauge(
    'mahalanobis_system_memory_usage_percent',
    'System memory usage percentage',
    registry=registry
)

SYSTEM_DISK_USAGE = Gauge(
    'mahalanobis_system_disk_usage_percent',
    'System disk usage percentage',
    registry=registry
)

# 캐시 메트릭
CACHE_SIZE = Gauge(
    'mahalanobis_cache_size_bytes',
    'Cache size in bytes',
    registry=registry
)

CACHE_ENTRIES = Gauge(
    'mahalanobis_cache_entries_total',
    'Total number of cache entries',
    registry=registry
)

CACHE_HIT_RATE = Gauge(
    'mahalanobis_cache_hit_rate_percent',
    'Cache hit rate percentage',
    registry=registry
)


class PrometheusMetricsCollector:
    """Prometheus 메트릭 수집기"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._last_collection_time = 0
        self._collection_interval = 15  # 15초마다 수집

    def collect_system_metrics(self):
        """시스템 메트릭 수집"""
        try:
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=1)
            SYSTEM_CPU_USAGE.set(cpu_percent)

            # 메모리 사용률
            memory = psutil.virtual_memory()
            SYSTEM_MEMORY_USAGE.set(memory.percent)

            # 디스크 사용률
            disk = psutil.disk_usage('/')
            SYSTEM_DISK_USAGE.set(disk.percent)

            self.logger.debug(f"시스템 메트릭 수집 완료 - CPU: {cpu_percent}%, 메모리: {memory.percent}%, 디스크: {disk.percent}%")

        except Exception as e:
            self.logger.error(f"시스템 메트릭 수집 실패: {e}")

    def collect_cache_metrics(self, cache_manager=None):
        """캐시 메트릭 수집"""
        try:
            if cache_manager:
                # 캐시 통계 수집
                stats = cache_manager.get_stats_sync()
                if stats:
                    CACHE_SIZE.set(stats.get('size', 0))
                    CACHE_ENTRIES.set(stats.get('entries', 0))
                    CACHE_HIT_RATE.set(stats.get('hit_rate', 0))

                    self.logger.debug(f"캐시 메트릭 수집 완료 - 크기: {stats.get('size', 0)} bytes, 항목: {stats.get('entries', 0)}개")
            else:
                # 캐시 관리자가 없는 경우 기본값 설정
                CACHE_SIZE.set(0)
                CACHE_ENTRIES.set(0)
                CACHE_HIT_RATE.set(0)

        except Exception as e:
            self.logger.error(f"캐시 메트릭 수집 실패: {e}")

    def record_api_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """API 요청 메트릭 기록"""
        try:
            # 요청 카운트 증가
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status=str(status_code)
            ).inc()

            # 요청 지연 시간 기록
            REQUEST_LATENCY.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)

            self.logger.debug(f"API 요청 메트릭 기록 - {method} {endpoint} {status_code} ({duration:.3f}s)")

        except Exception as e:
            self.logger.error(f"API 요청 메트릭 기록 실패: {e}")

    def record_analysis(self, status: str, duration: float, kpi_count: int, abnormal_kpi_count: int, cache_hit: bool = False):
        """마할라노비스 분석 메트릭 기록"""
        try:
            # 분석 카운트 증가
            ANALYSIS_COUNT.labels(status=status).inc()

            # 분석 시간 기록
            ANALYSIS_DURATION.observe(duration)

            # KPI 수 기록
            KPI_COUNT.observe(kpi_count)

            # 이상 KPI 수 기록
            ABNORMAL_KPI_COUNT.observe(abnormal_kpi_count)

            # 캐시 히트/미스 기록
            if cache_hit:
                CACHE_HITS.inc()
            else:
                CACHE_MISSES.inc()

            self.logger.debug(f"분석 메트릭 기록 - 상태: {status}, 시간: {duration:.3f}s, KPI: {kpi_count}개, 이상 KPI: {abnormal_kpi_count}개, 캐시 히트: {cache_hit}")

        except Exception as e:
            self.logger.error(f"분석 메트릭 기록 실패: {e}")

    def get_metrics_response(self) -> Response:
        """Prometheus 메트릭 응답 생성"""
        try:
            # 주기적으로 시스템 메트릭 수집
            current_time = time.time()
            if current_time - self._last_collection_time >= self._collection_interval:
                self.collect_system_metrics()
                self._last_collection_time = current_time

            # Prometheus 형식으로 메트릭 생성
            metrics_data = generate_latest(registry)

            return Response(
                content=metrics_data,
                media_type=CONTENT_TYPE_LATEST
            )

        except Exception as e:
            self.logger.error(f"메트릭 응답 생성 실패: {e}")
            return Response(
                content="# 메트릭 생성 실패\n",
                media_type=CONTENT_TYPE_LATEST,
                status_code=500
            )


# 전역 메트릭 수집기 인스턴스
metrics_collector = PrometheusMetricsCollector()


def get_metrics_collector() -> PrometheusMetricsCollector:
    """메트릭 수집기 인스턴스 반환"""
    return metrics_collector


def record_api_request(method: str, endpoint: str, status_code: int, duration: float):
    """API 요청 메트릭 기록 헬퍼 함수"""
    metrics_collector.record_api_request(method, endpoint, status_code, duration)


def record_analysis(status: str, duration: float, kpi_count: int, abnormal_kpi_count: int, cache_hit: bool = False):
    """분석 메트릭 기록 헬퍼 함수"""
    metrics_collector.record_analysis(status, duration, kpi_count, abnormal_kpi_count, cache_hit)


def update_cache_metrics(cache_manager=None):
    """캐시 메트릭 업데이트 헬퍼 함수"""
    metrics_collector.collect_cache_metrics(cache_manager)


