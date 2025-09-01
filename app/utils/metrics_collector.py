"""
시스템 메트릭 수집 및 분석

API 성능, 에러율, 시스템 리소스, 비즈니스 메트릭을 실시간으로 수집하고 분석합니다.
"""

import time
import logging
import asyncio
import psutil
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
import threading
from contextlib import asynccontextmanager

from ..middleware.request_tracing import create_request_context_logger

logger = logging.getLogger("app.metrics_collector")

# 메트릭 보관 기간 (메모리)
METRICS_RETENTION_MINUTES = 60
MAX_SAMPLES_PER_METRIC = 3600  # 1시간 * 60분 * 1샘플/분


@dataclass
class APIMetric:
    """API 메트릭 데이터"""
    endpoint: str
    method: str
    status_code: int
    duration_ms: float
    timestamp: datetime
    request_size: int = 0
    response_size: int = 0
    user_agent: str = ""
    remote_addr: str = ""


@dataclass
class SystemMetric:
    """시스템 메트릭 데이터"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    
    
@dataclass
class DatabaseMetric:
    """데이터베이스 메트릭 데이터"""
    timestamp: datetime
    connection_count: int
    active_queries: int
    slow_queries: int
    cache_hit_ratio: float
    index_usage: float
    storage_size_mb: float
    
    
@dataclass
class BusinessMetric:
    """비즈니스 메트릭 데이터"""
    timestamp: datetime
    analysis_results_created: int = 0
    analysis_results_viewed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    error_count: int = 0
    unique_users: int = 0


class MetricsCollector:
    """통합 메트릭 수집기"""
    
    def __init__(self):
        self.api_metrics: deque = deque(maxlen=MAX_SAMPLES_PER_METRIC)
        self.system_metrics: deque = deque(maxlen=MAX_SAMPLES_PER_METRIC)
        self.database_metrics: deque = deque(maxlen=MAX_SAMPLES_PER_METRIC)
        self.business_metrics: deque = deque(maxlen=MAX_SAMPLES_PER_METRIC)
        
        # 실시간 카운터
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.endpoint_stats = defaultdict(lambda: {"count": 0, "total_time": 0, "errors": 0})
        
        # 수집 스레드
        self._collection_thread = None
        self._stop_collection = False
        self._lock = threading.Lock()
        
        logger.info("메트릭 수집기 초기화 완료")
    
    def start_collection(self):
        """메트릭 수집 시작"""
        if self._collection_thread is not None:
            return
            
        self._stop_collection = False
        self._collection_thread = threading.Thread(target=self._collect_system_metrics_loop, daemon=True)
        self._collection_thread.start()
        logger.info("메트릭 수집 시작")
    
    def stop_collection(self):
        """메트릭 수집 중지"""
        self._stop_collection = True
        if self._collection_thread:
            self._collection_thread.join(timeout=5)
            self._collection_thread = None
        logger.info("메트릭 수집 중지")
    
    def _collect_system_metrics_loop(self):
        """시스템 메트릭 수집 루프 (백그라운드 스레드)"""
        while not self._stop_collection:
            try:
                metric = self._collect_system_metric()
                with self._lock:
                    self.system_metrics.append(metric)
                
                # 비즈니스 메트릭도 함께 수집
                business_metric = self._collect_business_metric()
                with self._lock:
                    self.business_metrics.append(business_metric)
                    
            except Exception as e:
                logger.warning(f"시스템 메트릭 수집 실패: {e}")
            
            # 60초 대기 (1분 간격 수집)
            for _ in range(60):
                if self._stop_collection:
                    break
                time.sleep(1)
    
    def _collect_system_metric(self) -> SystemMetric:
        """현재 시스템 메트릭 수집"""
        try:
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 메모리 정보
            memory = psutil.virtual_memory()
            
            # 디스크 정보
            disk = psutil.disk_usage('/')
            
            # 네트워크 정보
            try:
                network = psutil.net_io_counters()
                bytes_sent = network.bytes_sent
                bytes_recv = network.bytes_recv
            except:
                bytes_sent = bytes_recv = 0
            
            return SystemMetric(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / 1024 / 1024,
                memory_available_mb=memory.available / 1024 / 1024,
                disk_usage_percent=disk.percent,
                disk_free_gb=disk.free / 1024 / 1024 / 1024,
                network_bytes_sent=bytes_sent,
                network_bytes_recv=bytes_recv
            )
            
        except Exception as e:
            logger.warning(f"시스템 메트릭 수집 실패: {e}")
            return SystemMetric(
                timestamp=datetime.now(),
                cpu_percent=0,
                memory_percent=0,
                memory_used_mb=0,
                memory_available_mb=0,
                disk_usage_percent=0,
                disk_free_gb=0
            )
    
    def _collect_business_metric(self) -> BusinessMetric:
        """비즈니스 메트릭 수집"""
        with self._lock:
            # 현재 분의 카운터 집계
            current_minute = datetime.now().replace(second=0, microsecond=0)
            
            analysis_created = self.request_counts.get(f"analysis_created_{current_minute}", 0)
            analysis_viewed = self.request_counts.get(f"analysis_viewed_{current_minute}", 0)
            cache_hits = self.request_counts.get(f"cache_hits_{current_minute}", 0)
            cache_misses = self.request_counts.get(f"cache_misses_{current_minute}", 0)
            error_count = sum(self.error_counts.values())
            
            return BusinessMetric(
                timestamp=current_minute,
                analysis_results_created=analysis_created,
                analysis_results_viewed=analysis_viewed,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                error_count=error_count,
                unique_users=len(set())  # IP별 사용자 수 (구현 필요)
            )
    
    def record_api_call(self, endpoint: str, method: str, status_code: int, 
                       duration_ms: float, request_size: int = 0, 
                       response_size: int = 0, user_agent: str = "",
                       remote_addr: str = ""):
        """API 호출 메트릭 기록"""
        metric = APIMetric(
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            duration_ms=duration_ms,
            timestamp=datetime.now(),
            request_size=request_size,
            response_size=response_size,
            user_agent=user_agent,
            remote_addr=remote_addr
        )
        
        with self._lock:
            self.api_metrics.append(metric)
            
            # 실시간 통계 업데이트
            endpoint_key = f"{method}:{endpoint}"
            self.endpoint_stats[endpoint_key]["count"] += 1
            self.endpoint_stats[endpoint_key]["total_time"] += duration_ms
            
            if status_code >= 400:
                self.endpoint_stats[endpoint_key]["errors"] += 1
                self.error_counts[f"error_{status_code}"] += 1
    
    def record_business_event(self, event_type: str, details: Dict[str, Any] = None):
        """비즈니스 이벤트 기록"""
        current_minute = datetime.now().replace(second=0, microsecond=0)
        key = f"{event_type}_{current_minute}"
        
        with self._lock:
            self.request_counts[key] += 1
    
    def get_api_metrics_summary(self, minutes: int = 15) -> Dict[str, Any]:
        """API 메트릭 요약 (최근 N분)"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        
        with self._lock:
            recent_metrics = [m for m in self.api_metrics if m.timestamp >= cutoff]
        
        if not recent_metrics:
            return {"message": "최근 API 호출이 없습니다.", "period_minutes": minutes}
        
        # 통계 계산
        total_requests = len(recent_metrics)
        successful_requests = len([m for m in recent_metrics if m.status_code < 400])
        error_requests = total_requests - successful_requests
        
        durations = [m.duration_ms for m in recent_metrics]
        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)
        min_duration = min(durations)
        
        # 엔드포인트별 통계
        endpoint_stats = defaultdict(lambda: {"count": 0, "avg_duration": 0, "errors": 0})
        for metric in recent_metrics:
            key = f"{metric.method} {metric.endpoint}"
            endpoint_stats[key]["count"] += 1
            endpoint_stats[key]["avg_duration"] += metric.duration_ms
            if metric.status_code >= 400:
                endpoint_stats[key]["errors"] += 1
        
        # 평균 계산
        for key in endpoint_stats:
            count = endpoint_stats[key]["count"]
            endpoint_stats[key]["avg_duration"] /= count
        
        return {
            "period_minutes": minutes,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "error_requests": error_requests,
            "success_rate": successful_requests / total_requests * 100,
            "error_rate": error_requests / total_requests * 100,
            "performance": {
                "avg_duration_ms": round(avg_duration, 2),
                "max_duration_ms": round(max_duration, 2),
                "min_duration_ms": round(min_duration, 2)
            },
            "endpoints": dict(endpoint_stats),
            "period": {
                "start": cutoff.isoformat(),
                "end": datetime.now().isoformat()
            }
        }
    
    def get_system_metrics_summary(self, minutes: int = 15) -> Dict[str, Any]:
        """시스템 메트릭 요약"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        
        with self._lock:
            recent_metrics = [m for m in self.system_metrics if m.timestamp >= cutoff]
        
        if not recent_metrics:
            return {"message": "시스템 메트릭이 없습니다.", "period_minutes": minutes}
        
        # 평균값 계산
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_disk = sum(m.disk_usage_percent for m in recent_metrics) / len(recent_metrics)
        
        # 최신 값
        latest = recent_metrics[-1]
        
        return {
            "period_minutes": minutes,
            "samples_count": len(recent_metrics),
            "current": {
                "cpu_percent": latest.cpu_percent,
                "memory_percent": latest.memory_percent,
                "memory_used_mb": round(latest.memory_used_mb, 1),
                "memory_available_mb": round(latest.memory_available_mb, 1),
                "disk_usage_percent": latest.disk_usage_percent,
                "disk_free_gb": round(latest.disk_free_gb, 1)
            },
            "averages": {
                "cpu_percent": round(avg_cpu, 1),
                "memory_percent": round(avg_memory, 1),
                "disk_usage_percent": round(avg_disk, 1)
            },
            "alerts": self._generate_system_alerts(latest),
            "timestamp": latest.timestamp.isoformat()
        }
    
    def _generate_system_alerts(self, metric: SystemMetric) -> List[Dict[str, Any]]:
        """시스템 알림 생성"""
        alerts = []
        
        if metric.cpu_percent > 80:
            alerts.append({
                "type": "warning" if metric.cpu_percent < 90 else "critical",
                "message": f"CPU 사용률이 높습니다: {metric.cpu_percent:.1f}%",
                "value": metric.cpu_percent,
                "threshold": 80
            })
        
        if metric.memory_percent > 85:
            alerts.append({
                "type": "warning" if metric.memory_percent < 95 else "critical",
                "message": f"메모리 사용률이 높습니다: {metric.memory_percent:.1f}%",
                "value": metric.memory_percent,
                "threshold": 85
            })
        
        if metric.disk_usage_percent > 90:
            alerts.append({
                "type": "warning" if metric.disk_usage_percent < 95 else "critical",
                "message": f"디스크 사용률이 높습니다: {metric.disk_usage_percent:.1f}%",
                "value": metric.disk_usage_percent,
                "threshold": 90
            })
        
        return alerts
    
    def get_performance_alerts(self, minutes: int = 5) -> List[Dict[str, Any]]:
        """성능 알림 생성"""
        alerts = []
        api_summary = self.get_api_metrics_summary(minutes)
        
        # API 성능 알림
        if api_summary.get("error_rate", 0) > 5:
            alerts.append({
                "type": "warning" if api_summary["error_rate"] < 10 else "critical",
                "message": f"API 에러율이 높습니다: {api_summary['error_rate']:.1f}%",
                "value": api_summary["error_rate"],
                "threshold": 5,
                "category": "api_errors"
            })
        
        avg_duration = api_summary.get("performance", {}).get("avg_duration_ms", 0)
        if avg_duration > 5000:  # 5초
            alerts.append({
                "type": "warning" if avg_duration < 10000 else "critical",
                "message": f"API 응답 시간이 느립니다: {avg_duration:.0f}ms",
                "value": avg_duration,
                "threshold": 5000,
                "category": "api_performance"
            })
        
        # 시스템 알림 추가
        system_summary = self.get_system_metrics_summary(minutes)
        alerts.extend(system_summary.get("alerts", []))
        
        return alerts
    
    def get_comprehensive_health_report(self) -> Dict[str, Any]:
        """종합 건강 상태 보고서"""
        try:
            api_metrics = self.get_api_metrics_summary(15)
            system_metrics = self.get_system_metrics_summary(15)
            alerts = self.get_performance_alerts(5)
            
            # 전체 건강 상태 판단
            critical_alerts = [a for a in alerts if a.get("type") == "critical"]
            warning_alerts = [a for a in alerts if a.get("type") == "warning"]
            
            if critical_alerts:
                overall_status = "critical"
            elif warning_alerts:
                overall_status = "warning"
            elif api_metrics.get("error_rate", 0) > 1:
                overall_status = "degraded"
            else:
                overall_status = "healthy"
            
            return {
                "overall_status": overall_status,
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "critical_alerts": len(critical_alerts),
                    "warning_alerts": len(warning_alerts),
                    "api_error_rate": api_metrics.get("error_rate", 0),
                    "avg_response_time": api_metrics.get("performance", {}).get("avg_duration_ms", 0),
                    "cpu_usage": system_metrics.get("current", {}).get("cpu_percent", 0),
                    "memory_usage": system_metrics.get("current", {}).get("memory_percent", 0)
                },
                "details": {
                    "api_metrics": api_metrics,
                    "system_metrics": system_metrics,
                    "alerts": alerts
                },
                "recommendations": self._generate_recommendations(alerts, api_metrics, system_metrics)
            }
            
        except Exception as e:
            logger.error(f"건강 상태 보고서 생성 실패: {e}")
            return {
                "overall_status": "unknown",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_recommendations(self, alerts: List[Dict], api_metrics: Dict, 
                                system_metrics: Dict) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []
        
        # 시스템 리소스 권장사항
        cpu_usage = system_metrics.get("current", {}).get("cpu_percent", 0)
        memory_usage = system_metrics.get("current", {}).get("memory_percent", 0)
        
        if cpu_usage > 80:
            recommendations.append("CPU 사용률이 높습니다. 프로세스 최적화나 스케일 아웃을 고려하세요.")
        
        if memory_usage > 85:
            recommendations.append("메모리 사용률이 높습니다. 메모리 리크 확인이나 캐시 최적화를 검토하세요.")
        
        # API 성능 권장사항
        error_rate = api_metrics.get("error_rate", 0)
        avg_duration = api_metrics.get("performance", {}).get("avg_duration_ms", 0)
        
        if error_rate > 5:
            recommendations.append("API 에러율이 높습니다. 로그를 확인하고 에러 원인을 분석하세요.")
        
        if avg_duration > 3000:
            recommendations.append("API 응답 시간이 느립니다. 데이터베이스 쿼리 최적화나 캐싱을 강화하세요.")
        
        # 알림 기반 권장사항
        if any(a.get("category") == "api_performance" for a in alerts):
            recommendations.append("Redis 캐시 상태를 확인하고 인덱스 최적화를 검토하세요.")
        
        return recommendations


# 전역 메트릭 수집기
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """메트릭 수집기 인스턴스 반환"""
    global _metrics_collector
    
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
        _metrics_collector.start_collection()
    
    return _metrics_collector


def stop_metrics_collection():
    """메트릭 수집 중지"""
    global _metrics_collector
    
    if _metrics_collector:
        _metrics_collector.stop_collection()
        _metrics_collector = None
