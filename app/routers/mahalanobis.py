"""
마할라노비스 거리 분석 API 라우터

이 모듈은 마할라노비스 거리 분석을 위한 FastAPI 엔드포인트를 제공합니다.
"""

import logging
import hashlib
import json
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import JSONResponse

from ..models.mahalanobis import (
    KpiDataInput,
    AnalysisOptionsInput,
    MahalanobisAnalysisResult,
    StatisticalTestInput,
    StatisticalTestResponse
)
from ..services.mahalanobis_service import mahalanobis_service
from ..utils.cache_manager import get_cache_manager
from ..utils.prometheus_metrics import record_analysis

# 로거 설정
logger = logging.getLogger(__name__)

# 캐시 버전 관리 (캐시 무효화를 위한 버전)
CACHE_VERSION = "1.0"

# 캐시 키 생성 함수
def generate_cache_key(kpi_data: KpiDataInput, options: AnalysisOptionsInput, version: str = None) -> str:
    """
    마할라노비스 분석을 위한 캐시 키 생성

    Args:
        kpi_data: KPI 데이터 입력
        options: 분석 옵션
        version: 캐시 버전 (옵션, 기본값: CACHE_VERSION)

    Returns:
        str: 캐시 키
    """
    if version is None:
        version = CACHE_VERSION

    # 캐시 키 데이터 구성
    cache_data = {
        "kpi_data": kpi_data.kpi_data,
        "timestamps": kpi_data.timestamps,
        "period_labels": kpi_data.period_labels,
        "threshold": options.threshold,
        "sample_size": options.sample_size,
        "significance_level": options.significance_level,
        "version": version  # 캐시 무효화를 위한 버전
    }

    # 데이터를 정렬하여 일관된 해시 생성
    cache_string = json.dumps(cache_data, sort_keys=True, default=str)

    # SHA-256 해시 생성
    cache_key = hashlib.sha256(cache_string.encode()).hexdigest()

    return f"mahalanobis:{version}:{cache_key}"

# 캐시 TTL 설정 (30분)
MAHALANOBIS_CACHE_TTL = 1800

# 라우터 생성
router = APIRouter(
    prefix="/api/analysis",
    tags=["analysis"],
    responses={
        500: {"description": "서버 내부 오류"},
        400: {"description": "잘못된 요청"},
        422: {"description": "데이터 검증 오류"}
    }
)


@router.post(
    "/mahalanobis",
    response_model=MahalanobisAnalysisResult,
    summary="마할라노비스 거리 분석 수행",
    description="""
    KPI 데이터를 입력받아 마할라노비스 거리 기반으로 이상 감지 분석을 수행합니다.

    **분석 과정:**
    1. **1차 스크리닝**: 각 KPI의 변화율을 계산하여 임계치 기반 이상 감지
    2. **이상 KPI 선별**: 변화율이 큰 KPI들을 추출
    3. **2차 심층 분석**: 선별된 KPI들에 대해 통계적 검정 수행
    4. **종합 판정**: 분석 결과를 종합하여 최종 알람 레벨 결정

    **통계 테스트:**
    - Mann-Whitney U Test: 두 그룹의 중앙값 차이 검정
    - Kolmogorov-Smirnov Test: 두 그룹의 분포 차이 검정

    **알람 레벨:**
    - `normal`: 정상 범위
    - `caution`: 주의 필요 (약한 이상 감지)
    - `warning`: 경고 (중간 정도 이상 감지)
    - `critical`: 심각 (강한 이상 감지)
    """,
    responses={
        200: {
            "description": "마할라노비스 분석 성공",
            "content": {
                "application/json": {
                    "example": {
                        "data": {
                            "totalKpis": 3,
                            "abnormalKpis": [
                                {
                                    "kpiName": "RACH Success Rate",
                                    "n1Value": 98.5,
                                    "nValue": 85.2,
                                    "changeRate": 0.135,
                                    "severity": "caution"
                                }
                            ],
                            "abnormalScore": 0.333,
                            "alarmLevel": "caution",
                            "analysis": {
                                "screening": {
                                    "status": "caution",
                                    "score": 0.333,
                                    "threshold": 0.1,
                                    "description": "비정상 패턴 감지됨"
                                },
                                "drilldown": {
                                    "statisticalAnalysis": [
                                        {
                                            "kpiName": "RACH Success Rate",
                                            "changeRate": 0.135,
                                            "severity": "caution",
                                            "statisticalTests": {
                                                "mannWhitney": {
                                                    "U": 25.0,
                                                    "pValue": 0.0079,
                                                    "significant": True,
                                                    "interpretation": "통계적으로 유의한 차이 (p=0.0079)"
                                                },
                                                "kolmogorovSmirnov": {
                                                    "D": 1.0,
                                                    "pValue": 0.0079,
                                                    "significant": True,
                                                    "interpretation": "분포에 유의한 차이 (D=1.0000, p=0.0079)"
                                                }
                                            }
                                        }
                                    ],
                                    "summary": {
                                        "totalAnalyzed": 1,
                                        "statisticallySignificant": 1,
                                        "highConfidenceFindings": 1
                                    }
                                }
                            }
                        },
                        "processingTime": 0.0035
                    }
                }
            }
        },
        422: {
            "description": "데이터 검증 실패",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "KPI 데이터가 비어있습니다"
                    }
                }
            }
        }
    }
)
async def perform_mahalanobis_analysis(
    kpi_data: KpiDataInput,
    options: AnalysisOptionsInput,
    cache_manager = Depends(get_cache_manager)
) -> MahalanobisAnalysisResult:
    """
    마할라노비스 거리 분석 API 엔드포인트 (캐싱 지원)

    Args:
        kpi_data: KPI 데이터 입력
        options: 분석 옵션
        cache_manager: 캐시 관리자

    Returns:
        MahalanobisAnalysisResult: 분석 결과

    Raises:
        HTTPException: 요청 오류 또는 서버 오류
    """
    start_time = datetime.utcnow()
    cache_hit = False

    try:
        # 캐시 키 생성
        cache_key = generate_cache_key(kpi_data, options)
        logger.info(f"마할라노비스 거리 분석 요청 시작 - 캐시 키: {cache_key[:16]}...")

        # 캐시 조회 (폴백 메커니즘 적용)
        cache_available = True
        cached_result = None

        try:
            cached_result = await cache_manager.get(cache_key)
        except Exception as cache_error:
            logger.warning(f"캐시 조회 실패: {cache_error}, 캐시 없이 분석 진행")
            cache_available = False

        if cached_result is not None and cache_available:
            cache_hit = True
            logger.info("⚡ 캐시 히트: 저장된 마할라노비스 분석 결과 반환")

            # 캐시된 데이터를 Pydantic 모델로 변환
            if isinstance(cached_result, dict):
                result = MahalanobisAnalysisResult(**cached_result)
            else:
                # JSON 문자열인 경우 파싱
                import json
                try:
                    cached_data = json.loads(cached_result)
                    result = MahalanobisAnalysisResult(**cached_data)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"캐시 데이터 파싱 실패: {e}, 재계산 수행")
                    cached_result = None
                    cache_hit = False

        if not cache_hit:
            logger.info(f"📊 캐시 미스: 마할라노비스 분석 계산 시작 - KPI 수: {len(kpi_data.kpi_data)}, 임계값: {options.threshold}")

            # 마할라노비스 분석 수행
            result = mahalanobis_service.calculate_mahalanobis_distance(kpi_data, options)

            # 캐시에 저장 (캐시가 사용 가능한 경우에만)
            if cache_available:
                try:
                    result_dict = result.model_dump()
                    await cache_manager.set(cache_key, result_dict, ttl=MAHALANOBIS_CACHE_TTL)
                    logger.info(f"💾 분석 결과 캐시 저장 완료 (TTL: {MAHALANOBIS_CACHE_TTL}초)")
                except Exception as cache_error:
                    logger.warning(f"캐시 저장 실패 (폴백 모드로 계속 진행): {cache_error}")
            else:
                logger.info("💾 캐시 저장 생략 (캐시 시스템 unavailable)")

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        # 처리 시간 정보 추가
        result_dict = result.model_dump()
        result_dict['apiProcessingTime'] = processing_time
        result_dict['cacheHit'] = cache_hit
        result_dict['cacheAvailable'] = cache_available

        # 캐시 성능 메트릭 수집
        try:
            if cache_available:
                cache_stats = await cache_manager.get_stats()
                result_dict['cacheStats'] = {
                    'hitRate': cache_stats.get('hit_rate', 0),
                    'size': cache_stats.get('size', 0),
                    'type': cache_stats.get('type', 'unknown')
                }
        except Exception as metrics_error:
            logger.debug(f"캐시 메트릭 수집 실패: {metrics_error}")

        # Prometheus 메트릭 기록
        try:
            kpi_count = len(kpi_data.kpi_data) if kpi_data.kpi_data else 0
            abnormal_kpi_count = len(result_dict.get('data', {}).get('abnormalKpis', []))
            analysis_status = "success" if result.success else "failed"

            record_analysis(
                status=analysis_status,
                duration=processing_time,
                kpi_count=kpi_count,
                abnormal_kpi_count=abnormal_kpi_count,
                cache_hit=cache_hit
            )
        except Exception as metrics_error:
            logger.debug(f"분석 메트릭 기록 실패: {metrics_error}")

        logger.info(f"마할라노비스 거리 분석 완료 - 처리 시간: {processing_time:.3f}초, 캐시 히트: {cache_hit}, 캐시 사용 가능: {cache_available}")

        return MahalanobisAnalysisResult(**result_dict)

    except ValueError as e:
        logger.warning(f"마할라노비스 분석 검증 오류: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"데이터 검증 오류: {str(e)}"
        )

    except RuntimeError as e:
        logger.error(f"마할라노비스 분석 런타임 오류: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"분석 처리 오류: {str(e)}"
        )

    except Exception as e:
        logger.error(f"마할라노비스 분석 서버 오류: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"서버 내부 오류: {str(e)}"
        )


@router.post(
    "/mahalanobis/quick",
    response_model=MahalanobisAnalysisResult,
    summary="빠른 마할라노비스 분석 (기본 옵션)",
    description="""
    기본 분석 옵션으로 빠르게 마할라노비스 거리 분석을 수행합니다.

    **기본 옵션:**
    - 임계값: 0.1 (10%)
    - 샘플 크기: 20
    - 유의 수준: 0.05

    **사용 사례:**
    - 빠른 이상 감지
    - 실시간 모니터링
    - 기본 설정으로 충분한 경우
    """,
    responses={
        200: {
            "description": "빠른 마할라노비스 분석 성공"
        }
    }
)
async def perform_quick_mahalanobis_analysis(
    kpi_data: KpiDataInput,
    cache_manager = Depends(get_cache_manager)
) -> MahalanobisAnalysisResult:
    """
    빠른 마할라노비스 분석 API 엔드포인트 (캐싱 지원)

    기본 옵션으로 빠르게 분석을 수행합니다.

    Args:
        kpi_data: KPI 데이터 입력
        cache_manager: 캐시 관리자

    Returns:
        MahalanobisAnalysisResult: 분석 결과
    """
    # 기본 옵션 설정
    default_options = AnalysisOptionsInput(
        threshold=0.1,
        sample_size=20,
        significance_level=0.05
    )

    # 일반 분석 엔드포인트 호출 (캐시 매니저 전달)
    return await perform_mahalanobis_analysis(kpi_data, default_options, cache_manager)


@router.get(
    "/mahalanobis/info",
    summary="마할라노비스 분석 서비스 정보",
    description="마할라노비스 분석 서비스의 버전과 지원 기능 정보를 반환합니다.",
    response_model=Dict[str, Any]
)
async def get_mahalanobis_info() -> Dict[str, Any]:
    """
    마할라노비스 분석 서비스 정보 엔드포인트

    Returns:
        Dict[str, Any]: 서비스 정보
    """
    try:
        service_info = mahalanobis_service.get_performance_stats()

        return {
            "service": service_info,
            "endpoints": [
                {
                    "path": "/api/analysis/mahalanobis",
                    "method": "POST",
                    "description": "마할라노비스 거리 분석 수행"
                },
                {
                    "path": "/api/analysis/mahalanobis/quick",
                    "method": "POST",
                    "description": "빠른 마할라노비스 분석 (기본 옵션)"
                }
            ],
            "supportedAlgorithms": [
                "Mahalanobis Distance",
                "Mann-Whitney U Test",
                "Kolmogorov-Smirnov Test"
            ],
            "analysisTypes": [
                "1차 스크리닝 (변화율 기반)",
                "2차 심층 분석 (통계 검정)",
                "종합 이상 점수 계산",
                "알람 레벨 판정"
            ],
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    except Exception as e:
        logger.error(f"서비스 정보 조회 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="서비스 정보 조회 실패"
        )


@router.get(
    "/mahalanobis/health",
    summary="마할라노비스 분석 API 상태 확인",
    description="마할라노비스 분석 API의 상태를 확인합니다.",
    response_model=Dict[str, Any]
)
async def mahalanobis_health_check(
    cache_manager = Depends(get_cache_manager)
) -> Dict[str, Any]:
    """
    마할라노비스 분석 API 상태 확인 엔드포인트

    Args:
        cache_manager: 캐시 관리자

    Returns:
        Dict[str, Any]: 상태 정보
    """
    try:
        start_time = datetime.utcnow()

        # 간단한 테스트 수행으로 서비스 상태 확인
        test_kpi_data = KpiDataInput(
            kpiData={
                'test': [100.0, 101.0]
            }
        )
        test_options = AnalysisOptionsInput()

        result = mahalanobis_service.calculate_mahalanobis_distance(test_kpi_data, test_options)

        # 캐시 상태 확인
        cache_stats = await cache_manager.get_stats()

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "service": "MahalanobisAnalysisService",
            "version": "1.0.0",
            "test_status": "passed" if result.success else "failed",
            "response_time": f"{processing_time:.4f}s",
            "cache": {
                "status": "available",
                "type": cache_stats.get("type", "unknown"),
                "size": cache_stats.get("size", 0),
                "hit_rate": cache_stats.get("hit_rate", 0),
                "mahalanobis_cache_ttl": MAHALANOBIS_CACHE_TTL
            }
        }

    except Exception as e:
        logger.error(f"마할라노비스 분석 API 상태 확인 실패: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "service": "MahalanobisAnalysisService",
            "error": str(e),
            "cache": {
                "status": "unavailable"
            }
        }


@router.get(
    "/cache/stats",
    summary="마할라노비스 캐시 통계 조회",
    description="마할라노비스 분석 캐시의 통계 정보를 조회합니다.",
    response_model=Dict[str, Any]
)
async def get_cache_stats(
    cache_manager = Depends(get_cache_manager)
) -> Dict[str, Any]:
    """
    캐시 통계 조회 엔드포인트

    Args:
        cache_manager: 캐시 관리자

    Returns:
        Dict[str, Any]: 캐시 통계 정보
    """
    try:
        stats = await cache_manager.get_stats()

        # 마할라노비스 관련 캐시 항목 수 계산
        mahalanobis_keys = 0
        total_keys = 0
        mahalanobis_memory_usage = 0

        try:
            all_keys = await cache_manager.get_all_keys()
            total_keys = len(all_keys)
            mahalanobis_keys = len([key for key in all_keys if key.startswith("mahalanobis:")])

            # 마할라노비스 캐시 메모리 사용량 계산
            for key in all_keys:
                if key.startswith("mahalanobis:"):
                    try:
                        value = await cache_manager.get(key)
                        if value:
                            # 대략적인 메모리 사용량 계산 (키 + 값)
                            key_size = len(key.encode('utf-8'))
                            value_size = len(str(value).encode('utf-8')) if isinstance(value, (dict, list)) else len(str(value).encode('utf-8'))
                            mahalanobis_memory_usage += key_size + value_size
                    except Exception as e:
                        logger.debug(f"캐시 항목 메모리 계산 실패: {key} - {e}")

        except Exception as e:
            logger.warning(f"캐시 키 조회 실패: {e}")

        # 캐시 성능 메트릭 계산
        hit_count = stats.get("hit_count", 0)
        miss_count = stats.get("miss_count", 0)
        total_requests = hit_count + miss_count
        calculated_hit_rate = (hit_count / total_requests * 100) if total_requests > 0 else 0

        # 평균 응답 시간 추정 (캐시 히트 vs 미스)
        avg_hit_time = 0.001  # 1ms (캐시 히트)
        avg_miss_time = 0.100  # 100ms (캐시 미스 + 계산)
        avg_response_time = (hit_count * avg_hit_time + miss_count * avg_miss_time) / total_requests if total_requests > 0 else 0

        return {
            "cache_type": stats.get("type", "unknown"),
            "size": stats.get("size", 0),
            "max_size": stats.get("max_size", 0),
            "hit_rate": stats.get("hit_rate", calculated_hit_rate),
            "hit_count": hit_count,
            "miss_count": miss_count,
            "total_requests": total_requests,
            "avg_response_time": round(avg_response_time, 4),
            "mahalanobis_entries": mahalanobis_keys,
            "total_entries": total_keys,
            "mahalanobis_memory_usage": mahalanobis_memory_usage,
            "mahalanobis_cache_ttl": MAHALANOBIS_CACHE_TTL,
            "cache_version": CACHE_VERSION,
            "performance_metrics": {
                "cache_efficiency": calculated_hit_rate,
                "time_saved": round(miss_count * (avg_miss_time - avg_hit_time), 4),
                "mahalanobis_ratio": (mahalanobis_keys / total_keys * 100) if total_keys > 0 else 0
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    except Exception as e:
        logger.error(f"캐시 통계 조회 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"캐시 통계 조회 실패: {str(e)}"
        )


@router.post(
    "/cache/clear",
    summary="마할라노비스 캐시 정리",
    description="마할라노비스 분석 캐시를 정리합니다.",
    response_model=Dict[str, Any]
)
async def clear_cache(
    pattern: str = None,
    cache_manager = Depends(get_cache_manager)
) -> Dict[str, Any]:
    """
    캐시 정리 엔드포인트

    Args:
        pattern: 정리할 패턴 (옵션, 기본값: "mahalanobis:")
        cache_manager: 캐시 관리자

    Returns:
        Dict[str, Any]: 정리 결과
    """
    try:
        if pattern is None:
            pattern = "mahalanobis:"

        logger.info(f"캐시 정리 요청: 패턴 '{pattern}'")

        # 캐시 정리 수행
        cleared_count = await cache_manager.clear_pattern(pattern)

        logger.info(f"캐시 정리 완료: {cleared_count}개 항목 제거")

        return {
            "success": True,
            "cleared_count": cleared_count,
            "pattern": pattern,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    except Exception as e:
        logger.error(f"캐시 정리 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"캐시 정리 실패: {str(e)}"
        )


@router.post(
    "/cache/invalidate",
    summary="특정 마할라노비스 캐시 무효화",
    description="특정 캐시 키를 무효화합니다.",
    response_model=Dict[str, Any]
)
async def invalidate_cache(
    cache_key: str,
    cache_manager = Depends(get_cache_manager)
) -> Dict[str, Any]:
    """
    캐시 무효화 엔드포인트

    Args:
        cache_key: 무효화할 캐시 키
        cache_manager: 캐시 관리자

    Returns:
        Dict[str, Any]: 무효화 결과
    """
    try:
        logger.info(f"캐시 무효화 요청: {cache_key}")

        # 캐시 키 존재 확인
        exists = await cache_manager.exists(cache_key)

        if exists:
            # 캐시 삭제
            await cache_manager.delete(cache_key)
            logger.info(f"캐시 무효화 완료: {cache_key}")

            return {
                "success": True,
                "cache_key": cache_key,
                "invalidated": True,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        else:
            logger.info(f"캐시 키가 존재하지 않음: {cache_key}")

            return {
                "success": True,
                "cache_key": cache_key,
                "invalidated": False,
                "message": "캐시 키가 존재하지 않습니다",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

    except Exception as e:
        logger.error(f"캐시 무효화 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"캐시 무효화 실패: {str(e)}"
        )


@router.post(
    "/cache/invalidate-version",
    summary="마할라노비스 캐시 버전 무효화",
    description="특정 버전의 모든 마할라노비스 캐시를 무효화합니다.",
    response_model=Dict[str, Any]
)
async def invalidate_cache_version(
    version: str = None,
    cache_manager = Depends(get_cache_manager)
) -> Dict[str, Any]:
    """
    캐시 버전 무효화 엔드포인트

    Args:
        version: 무효화할 버전 (옵션, 기본값: 현재 버전)
        cache_manager: 캐시 관리자

    Returns:
        Dict[str, Any]: 무효화 결과
    """
    try:
        if version is None:
            version = CACHE_VERSION

        pattern = f"mahalanobis:{version}:*"
        logger.info(f"캐시 버전 무효화 요청: 버전 {version}")

        # 해당 버전의 모든 캐시 정리
        cleared_count = await cache_manager.clear_pattern(pattern)

        logger.info(f"캐시 버전 무효화 완료: 버전 {version}, {cleared_count}개 항목 제거")

        return {
            "success": True,
            "version": version,
            "cleared_count": cleared_count,
            "pattern": pattern,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    except Exception as e:
        logger.error(f"캐시 버전 무효화 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"캐시 버전 무효화 실패: {str(e)}"
        )


@router.post(
    "/cache/update-version",
    summary="마할라노비스 캐시 버전 업데이트",
    description="캐시 버전을 업데이트하여 이전 버전 캐시를 무효화합니다.",
    response_model=Dict[str, Any]
)
async def update_cache_version(
    new_version: str,
    cache_manager = Depends(get_cache_manager)
) -> Dict[str, Any]:
    """
    캐시 버전 업데이트 엔드포인트

    Args:
        new_version: 새로운 버전
        cache_manager: 캐시 관리자

    Returns:
        Dict[str, Any]: 업데이트 결과
    """
    try:
        global CACHE_VERSION
        old_version = CACHE_VERSION

        logger.info(f"캐시 버전 업데이트 요청: {old_version} → {new_version}")

        # 이전 버전 캐시 정리
        old_pattern = f"mahalanobis:{old_version}:*"
        cleared_count = await cache_manager.clear_pattern(old_pattern)

        # 버전 업데이트
        CACHE_VERSION = new_version

        logger.info(f"캐시 버전 업데이트 완료: {old_version} → {new_version}, {cleared_count}개 이전 버전 캐시 제거")

        return {
            "success": True,
            "old_version": old_version,
            "new_version": new_version,
            "cleared_count": cleared_count,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    except Exception as e:
        logger.error(f"캐시 버전 업데이트 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"캐시 버전 업데이트 실패: {str(e)}"
        )


@router.get(
    "/cache/version",
    summary="현재 마할라노비스 캐시 버전 조회",
    description="현재 사용 중인 캐시 버전을 조회합니다.",
    response_model=Dict[str, Any]
)
async def get_cache_version() -> Dict[str, Any]:
    """
    캐시 버전 조회 엔드포인트

    Returns:
        Dict[str, Any]: 버전 정보
    """
    return {
        "version": CACHE_VERSION,
        "ttl": MAHALANOBIS_CACHE_TTL,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
