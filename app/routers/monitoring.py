"""
시스템 모니터링 API

성능 메트릭, 건강 상태, 알림 등 시스템 모니터링 관련 엔드포인트를 제공합니다.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse

from ..utils.metrics_collector import get_metrics_collector
from ..utils.cache_manager import get_cache_manager
from ..utils.data_optimization import get_optimization_stats
from ..middleware.request_tracing import create_request_context_logger, get_current_request_id
from ..db import get_database

logger = logging.getLogger("app.monitoring")

# 라우터 생성
router = APIRouter(
    prefix="/api/monitoring",
    tags=["Monitoring"],
    responses={
        500: {"description": "Internal server error"},
        503: {"description": "Service unavailable"}
    }
)


@router.get("/health/comprehensive", summary="종합 건강 상태", tags=["Health"])
async def get_comprehensive_health():
    """
    시스템 전체의 종합적인 건강 상태를 반환합니다.
    
    API 성능, 시스템 리소스, 데이터베이스, 캐시 등 모든 구성 요소의 상태를 포함합니다.
    """
    request_id = get_current_request_id()
    req_logger = create_request_context_logger("app.monitoring.comprehensive_health")
    
    try:
        req_logger.info("종합 건강 상태 확인 시작")
        
        # 메트릭 수집기에서 종합 보고서 가져오기
        metrics_collector = get_metrics_collector()
        health_report = metrics_collector.get_comprehensive_health_report()
        
        # 추가 서비스 상태 확인
        cache_manager = await get_cache_manager()
        cache_stats = await cache_manager.get_cache_stats()
        
        # 데이터 최적화 통계
        optimization_stats = await get_optimization_stats()
        
        # 전체 응답 구성
        comprehensive_health = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "overall_status": health_report.get("overall_status", "unknown"),
            "components": {
                "api": {
                    "status": "healthy" if health_report.get("summary", {}).get("api_error_rate", 0) < 5 else "degraded",
                    "metrics": health_report.get("details", {}).get("api_metrics", {})
                },
                "system": {
                    "status": _determine_system_status(health_report.get("details", {}).get("system_metrics", {})),
                    "metrics": health_report.get("details", {}).get("system_metrics", {})
                },
                "cache": {
                    "status": "healthy" if cache_stats.get("redis_available", False) else "degraded",
                    "metrics": cache_stats
                },
                "database": {
                    "status": await _check_database_health(),
                    "optimization": optimization_stats
                }
            },
            "alerts": health_report.get("details", {}).get("alerts", []),
            "recommendations": health_report.get("recommendations", []),
            "summary": health_report.get("summary", {})
        }
        
        # 상태에 따른 HTTP 상태 코드 결정
        overall_status = comprehensive_health["overall_status"]
        if overall_status == "critical":
            status_code = 503
        elif overall_status in ["warning", "degraded"]:
            status_code = 200  # 여전히 서비스 가능
        else:
            status_code = 200
        
        req_logger.info("종합 건강 상태 확인 완료", extra={
            "overall_status": overall_status,
            "alert_count": len(comprehensive_health["alerts"]),
            "recommendation_count": len(comprehensive_health["recommendations"])
        })
        
        return JSONResponse(content=comprehensive_health, status_code=status_code)
        
    except Exception as e:
        req_logger.error(f"종합 건강 상태 확인 실패: {e}", extra={
            "error_type": type(e).__name__
        }, exc_info=True)
        
        return JSONResponse(
            status_code=503,
            content={
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "overall_status": "critical",
                "error": {
                    "message": "건강 상태 확인 실패",
                    "details": str(e)
                }
            }
        )


@router.get("/metrics/api", summary="API 성능 메트릭", tags=["Metrics"])
async def get_api_metrics(
    minutes: int = Query(15, ge=1, le=1440, description="조회할 시간 범위 (분)")
):
    """
    API 성능 메트릭을 조회합니다.
    
    응답 시간, 에러율, 엔드포인트별 통계 등을 제공합니다.
    """
    request_id = get_current_request_id()
    req_logger = create_request_context_logger("app.monitoring.api_metrics")
    
    try:
        req_logger.info("API 메트릭 조회 시작", extra={"period_minutes": minutes})
        
        metrics_collector = get_metrics_collector()
        api_metrics = metrics_collector.get_api_metrics_summary(minutes)
        
        req_logger.info("API 메트릭 조회 완료", extra={
            "period_minutes": minutes,
            "total_requests": api_metrics.get("total_requests", 0),
            "error_rate": api_metrics.get("error_rate", 0)
        })
        
        return {
            "status": "success",
            "request_id": request_id,
            "data": api_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        req_logger.error(f"API 메트릭 조회 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"API 메트릭 조회 실패: {str(e)}"
        )


@router.get("/metrics/system", summary="시스템 리소스 메트릭", tags=["Metrics"])
async def get_system_metrics(
    minutes: int = Query(15, ge=1, le=1440, description="조회할 시간 범위 (분)")
):
    """
    시스템 리소스 메트릭을 조회합니다.
    
    CPU, 메모리, 디스크 사용률 등을 제공합니다.
    """
    request_id = get_current_request_id()
    req_logger = create_request_context_logger("app.monitoring.system_metrics")
    
    try:
        req_logger.info("시스템 메트릭 조회 시작", extra={"period_minutes": minutes})
        
        metrics_collector = get_metrics_collector()
        system_metrics = metrics_collector.get_system_metrics_summary(minutes)
        
        req_logger.info("시스템 메트릭 조회 완료", extra={
            "period_minutes": minutes,
            "cpu_usage": system_metrics.get("current", {}).get("cpu_percent", 0),
            "memory_usage": system_metrics.get("current", {}).get("memory_percent", 0)
        })
        
        return {
            "status": "success",
            "request_id": request_id,
            "data": system_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        req_logger.error(f"시스템 메트릭 조회 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"시스템 메트릭 조회 실패: {str(e)}"
        )


@router.get("/alerts", summary="시스템 알림", tags=["Alerts"])
async def get_alerts(
    minutes: int = Query(5, ge=1, le=60, description="알림 확인 시간 범위 (분)"),
    severity: Optional[str] = Query(None, description="알림 심각도 필터 (warning, critical)")
):
    """
    시스템 알림을 조회합니다.
    
    성능, 에러율, 리소스 사용량 등에 대한 알림을 제공합니다.
    """
    request_id = get_current_request_id()
    req_logger = create_request_context_logger("app.monitoring.alerts")
    
    try:
        req_logger.info("알림 조회 시작", extra={
            "period_minutes": minutes,
            "severity_filter": severity
        })
        
        metrics_collector = get_metrics_collector()
        alerts = metrics_collector.get_performance_alerts(minutes)
        
        # 심각도 필터 적용
        if severity:
            alerts = [alert for alert in alerts if alert.get("type") == severity]
        
        # 알림 분류
        critical_alerts = [a for a in alerts if a.get("type") == "critical"]
        warning_alerts = [a for a in alerts if a.get("type") == "warning"]
        
        req_logger.info("알림 조회 완료", extra={
            "total_alerts": len(alerts),
            "critical_count": len(critical_alerts),
            "warning_count": len(warning_alerts)
        })
        
        return {
            "status": "success",
            "request_id": request_id,
            "data": {
                "alerts": alerts,
                "summary": {
                    "total": len(alerts),
                    "critical": len(critical_alerts),
                    "warning": len(warning_alerts)
                },
                "period_minutes": minutes
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        req_logger.error(f"알림 조회 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"알림 조회 실패: {str(e)}"
        )


@router.get("/performance/summary", summary="성능 요약", tags=["Performance"])
async def get_performance_summary():
    """
    시스템 전체 성능 요약을 제공합니다.
    
    핵심 성능 지표들의 요약 정보를 제공합니다.
    """
    request_id = get_current_request_id()
    req_logger = create_request_context_logger("app.monitoring.performance_summary")
    
    try:
        req_logger.info("성능 요약 조회 시작")
        
        metrics_collector = get_metrics_collector()
        
        # 다양한 시간 범위의 메트릭 수집
        api_metrics_5min = metrics_collector.get_api_metrics_summary(5)
        api_metrics_1hour = metrics_collector.get_api_metrics_summary(60)
        system_metrics = metrics_collector.get_system_metrics_summary(15)
        alerts = metrics_collector.get_performance_alerts(5)
        
        # 캐시 성능
        cache_manager = await get_cache_manager()
        cache_stats = await cache_manager.get_cache_stats()
        
        # 성능 요약 구성
        performance_summary = {
            "overall_health": _calculate_overall_health_score(
                api_metrics_5min, system_metrics, alerts
            ),
            "api_performance": {
                "current_5min": {
                    "avg_response_time": api_metrics_5min.get("performance", {}).get("avg_duration_ms", 0),
                    "error_rate": api_metrics_5min.get("error_rate", 0),
                    "requests_per_minute": api_metrics_5min.get("total_requests", 0)
                },
                "last_hour": {
                    "avg_response_time": api_metrics_1hour.get("performance", {}).get("avg_duration_ms", 0),
                    "error_rate": api_metrics_1hour.get("error_rate", 0),
                    "total_requests": api_metrics_1hour.get("total_requests", 0)
                }
            },
            "system_resources": {
                "cpu_usage": system_metrics.get("current", {}).get("cpu_percent", 0),
                "memory_usage": system_metrics.get("current", {}).get("memory_percent", 0),
                "disk_usage": system_metrics.get("current", {}).get("disk_usage_percent", 0)
            },
            "cache_performance": {
                "redis_available": cache_stats.get("redis_available", False),
                "memory_cache_usage": cache_stats.get("memory_cache", {}).get("usage_percentage", 0),
                "hit_rate": cache_stats.get("redis", {}).get("hit_rate", 0) if cache_stats.get("redis_available") else None
            },
            "alerts_summary": {
                "total_alerts": len(alerts),
                "critical_alerts": len([a for a in alerts if a.get("type") == "critical"]),
                "warning_alerts": len([a for a in alerts if a.get("type") == "warning"])
            }
        }
        
        req_logger.info("성능 요약 조회 완료", extra={
            "health_score": performance_summary["overall_health"]["score"],
            "alert_count": performance_summary["alerts_summary"]["total_alerts"]
        })
        
        return {
            "status": "success",
            "request_id": request_id,
            "data": performance_summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        req_logger.error(f"성능 요약 조회 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"성능 요약 조회 실패: {str(e)}"
        )


# 헬퍼 함수들

def _determine_system_status(system_metrics: Dict[str, Any]) -> str:
    """시스템 메트릭을 기반으로 상태 결정"""
    current = system_metrics.get("current", {})
    cpu_usage = current.get("cpu_percent", 0)
    memory_usage = current.get("memory_percent", 0)
    disk_usage = current.get("disk_usage_percent", 0)
    
    if cpu_usage > 90 or memory_usage > 95 or disk_usage > 95:
        return "critical"
    elif cpu_usage > 80 or memory_usage > 85 or disk_usage > 90:
        return "warning"
    else:
        return "healthy"


async def _check_database_health() -> str:
    """데이터베이스 건강 상태 확인"""
    try:
        db = await get_database()
        # 간단한 ping 명령으로 연결 상태 확인
        await db.command("ping")
        return "healthy"
    except Exception:
        return "unhealthy"


def _calculate_overall_health_score(api_metrics: Dict, system_metrics: Dict, 
                                  alerts: List[Dict]) -> Dict[str, Any]:
    """전체 건강 점수 계산 (0-100)"""
    score = 100
    
    # API 성능 점수 (40점 만점)
    error_rate = api_metrics.get("error_rate", 0)
    avg_duration = api_metrics.get("performance", {}).get("avg_duration_ms", 0)
    
    # 에러율 기반 점수 차감
    if error_rate > 10:
        score -= 40
    elif error_rate > 5:
        score -= 20
    elif error_rate > 1:
        score -= 10
    
    # 응답 시간 기반 점수 차감
    if avg_duration > 10000:  # 10초
        score -= 30
    elif avg_duration > 5000:  # 5초
        score -= 15
    elif avg_duration > 2000:  # 2초
        score -= 5
    
    # 시스템 리소스 점수 (30점 만점)
    current = system_metrics.get("current", {})
    cpu_usage = current.get("cpu_percent", 0)
    memory_usage = current.get("memory_percent", 0)
    
    if cpu_usage > 90 or memory_usage > 95:
        score -= 30
    elif cpu_usage > 80 or memory_usage > 85:
        score -= 15
    elif cpu_usage > 70 or memory_usage > 75:
        score -= 5
    
    # 알림 기반 점수 차감 (30점 만점)
    critical_alerts = len([a for a in alerts if a.get("type") == "critical"])
    warning_alerts = len([a for a in alerts if a.get("type") == "warning"])
    
    score -= critical_alerts * 15  # 치명적 알림당 15점 차감
    score -= warning_alerts * 5   # 경고 알림당 5점 차감
    
    # 최소 0점 보장
    score = max(0, score)
    
    # 상태 결정
    if score >= 90:
        status = "excellent"
    elif score >= 75:
        status = "good"
    elif score >= 50:
        status = "fair"
    elif score >= 25:
        status = "poor"
    else:
        status = "critical"
    
    return {
        "score": score,
        "status": status,
        "factors": {
            "api_performance": 100 - min(40, error_rate * 4 + (avg_duration / 250)),
            "system_resources": 100 - min(30, max(cpu_usage - 50, 0) + max(memory_usage - 50, 0)),
            "alerts_impact": 100 - min(30, critical_alerts * 15 + warning_alerts * 5)
        }
    }
