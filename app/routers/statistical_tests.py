"""
독립적인 통계 테스트 API 라우터

이 모듈은 마할라노비스 분석과 별개로 사용할 수 있는
독립적인 통계 테스트 엔드포인트를 제공합니다.
"""

import logging
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from ..models.mahalanobis import StatisticalTestInput, StatisticalTestResponse, StatisticalTestResult
from ..services.statistical_tests_service import statistical_tests_service

# 로거 설정
logger = logging.getLogger(__name__)

# 라우터 생성
router = APIRouter(
    prefix="/api/statistical-tests",
    tags=["statistical-tests"],
    responses={
        500: {"description": "서버 내부 오류"},
        400: {"description": "잘못된 요청"},
        422: {"description": "데이터 검증 오류"}
    }
)


@router.post(
    "/mann-whitney-u",
    response_model=StatisticalTestResponse,
    summary="Mann-Whitney U Test 수행",
    description="""
    두 그룹의 데이터를 비교하여 Mann-Whitney U 통계 검정을 수행합니다.

    **특징:**
    - 비모수적 검정 (데이터의 정규성 가정 불필요)
    - 두 독립적인 그룹의 중앙값 차이 검정
    - 이상치에 강건한 분석 제공

    **사용 사례:**
    - 두 기간의 KPI 성능 비교
    - 서로 다른 그룹 간 차이 분석
    - 순위 기반 통계적 유의성 검정
    """,
    responses={
        200: {
            "description": "Mann-Whitney U Test 성공",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "testName": "Mann-Whitney U",
                        "result": {
                            "testName": "Mann-Whitney U",
                            "statistic": 25.0,
                            "pValue": 0.0234,
                            "significant": True,
                            "interpretation": "통계적으로 유의한 차이 (p=0.0234)"
                        },
                        "processingTime": 0.0012,
                        "timestamp": "2024-01-01T12:00:00Z"
                    }
                }
            }
        }
    }
)
async def perform_mann_whitney_u_test(
    request: StatisticalTestInput
) -> StatisticalTestResponse:
    """
    Mann-Whitney U Test API 엔드포인트

    Args:
        request: 테스트 요청 데이터

    Returns:
        StatisticalTestResponse: 테스트 결과

    Raises:
        HTTPException: 요청 오류 또는 서버 오류
    """
    start_time = datetime.utcnow()

    try:
        logger.info("Mann-Whitney U Test 요청 시작")
        logger.info(f"그룹 A 크기: {len(request.group_a)}, 그룹 B 크기: {len(request.group_b)}")

        # 통계 테스트 수행
        result = statistical_tests_service.mann_whitney_u_test(
            request.group_a,
            request.group_b,
            request.significance_level
        )

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        response = StatisticalTestResponse(
            success=True,
            test_name=result.test_name,
            result=result,
            processing_time=processing_time,
            message="Mann-Whitney U Test completed successfully"
        )

        logger.info(f"Mann-Whitney U Test 완료 - 처리 시간: {processing_time:.3f}초")
        return response

    except ValueError as e:
        logger.warning(f"Mann-Whitney U Test 검증 오류: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"데이터 검증 오류: {str(e)}"
        )

    except Exception as e:
        logger.error(f"Mann-Whitney U Test 서버 오류: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"서버 내부 오류: {str(e)}"
        )


@router.post(
    "/kolmogorov-smirnov",
    response_model=StatisticalTestResponse,
    summary="Kolmogorov-Smirnov Test 수행",
    description="""
    두 그룹의 데이터 분포를 비교하여 Kolmogorov-Smirnov 통계 검정을 수행합니다.

    **특징:**
    - 두 샘플의 분포 차이 검정
    - 누적 분포 함수(CDF) 기반 비교
    - 정규성 가정 불필요

    **사용 사례:**
    - 두 기간 데이터의 분포 변화 분석
    - 서로 다른 그룹의 분포 비교
    - 이상 감지 및 분포 변화 탐지
    """,
    responses={
        200: {
            "description": "Kolmogorov-Smirnov Test 성공",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "testName": "Kolmogorov-Smirnov",
                        "result": {
                            "testName": "Kolmogorov-Smirnov",
                            "statistic": 0.15,
                            "pValue": 0.034,
                            "significant": True,
                            "interpretation": "분포에 유의한 차이 (D=0.1500, p=0.0340)"
                        },
                        "processingTime": 0.0015,
                        "timestamp": "2024-01-01T12:00:00Z"
                    }
                }
            }
        }
    }
)
async def perform_kolmogorov_smirnov_test(
    request: StatisticalTestInput
) -> StatisticalTestResponse:
    """
    Kolmogorov-Smirnov Test API 엔드포인트

    Args:
        request: 테스트 요청 데이터

    Returns:
        StatisticalTestResponse: 테스트 결과

    Raises:
        HTTPException: 요청 오류 또는 서버 오류
    """
    start_time = datetime.utcnow()

    try:
        logger.info("Kolmogorov-Smirnov Test 요청 시작")
        logger.info(f"그룹 A 크기: {len(request.group_a)}, 그룹 B 크기: {len(request.group_b)}")

        # 통계 테스트 수행
        result = statistical_tests_service.kolmogorov_smirnov_test(
            request.group_a,
            request.group_b,
            request.significance_level
        )

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        response = StatisticalTestResponse(
            success=True,
            test_name=result.test_name,
            result=result,
            processing_time=processing_time,
            message="Kolmogorov-Smirnov Test completed successfully"
        )

        logger.info(f"Kolmogorov-Smirnov Test 완료 - 처리 시간: {processing_time:.3f}초")
        return response

    except ValueError as e:
        logger.warning(f"Kolmogorov-Smirnov Test 검증 오류: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"데이터 검증 오류: {str(e)}"
        )

    except Exception as e:
        logger.error(f"Kolmogorov-Smirnov Test 서버 오류: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"서버 내부 오류: {str(e)}"
        )


@router.get(
    "/info",
    summary="통계 테스트 서비스 정보",
    description="통계 테스트 서비스의 버전과 지원 기능 정보를 반환합니다.",
    response_model=Dict[str, Any]
)
async def get_statistical_tests_info() -> Dict[str, Any]:
    """
    통계 테스트 서비스 정보 엔드포인트

    Returns:
        Dict[str, Any]: 서비스 정보
    """
    try:
        service_info = statistical_tests_service.get_service_info()

        return {
            "service": service_info,
            "endpoints": [
                {
                    "path": "/api/statistical-tests/mann-whitney-u",
                    "method": "POST",
                    "description": "Mann-Whitney U Test 수행"
                },
                {
                    "path": "/api/statistical-tests/kolmogorov-smirnov",
                    "method": "POST",
                    "description": "Kolmogorov-Smirnov Test 수행"
                }
            ],
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    except Exception as e:
        logger.error(f"서비스 정보 조회 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="서비스 정보 조회 실패"
        )


@router.post(
    "/batch/mann-whitney-u",
    summary="배치 Mann-Whitney U Test 수행",
    description="""
    여러 데이터셋에 대해 Mann-Whitney U Test를 배치로 수행합니다.

    **특징:**
    - 여러 테스트 케이스를 한 번에 처리
    - 각 테스트별 독립적인 결과 반환
    - 배치 처리로 효율성 향상

    **입력 형식:**
    - 각 테스트 케이스는 독립적인 groupA와 groupB를 가짐
    - 모든 테스트에 동일한 significanceLevel 적용
    """,
    response_model=Dict[str, Any]
)
async def perform_batch_mann_whitney_u_test(
    test_cases: Dict[str, StatisticalTestInput]
) -> Dict[str, Any]:
    """
    배치 Mann-Whitney U Test API 엔드포인트

    Args:
        test_cases: 테스트 케이스 딕셔너리 (키: 테스트 식별자, 값: 테스트 입력)

    Returns:
        Dict[str, Any]: 배치 테스트 결과
    """
    start_time = datetime.utcnow()

    try:
        logger.info(f"배치 Mann-Whitney U Test 시작 - 케이스 수: {len(test_cases)}")

        results = {}
        total_processing_time = 0.0

        for test_id, test_input in test_cases.items():
            case_start_time = datetime.utcnow()

            try:
                result = statistical_tests_service.mann_whitney_u_test(
                    test_input.group_a,
                    test_input.group_b,
                    test_input.significance_level
                )

                case_processing_time = (datetime.utcnow() - case_start_time).total_seconds()
                total_processing_time += case_processing_time

                results[test_id] = {
                    "success": True,
                    "result": result,
                    "processing_time": case_processing_time
                }

            except Exception as e:
                case_processing_time = (datetime.utcnow() - case_start_time).total_seconds()
                total_processing_time += case_processing_time

                logger.warning(f"배치 테스트 케이스 실패 - {test_id}: {e}")
                results[test_id] = {
                    "success": False,
                    "error": str(e),
                    "processing_time": case_processing_time
                }

        total_time = (datetime.utcnow() - start_time).total_seconds()

        response = {
            "success": True,
            "batch_summary": {
                "total_cases": len(test_cases),
                "successful_cases": sum(1 for r in results.values() if r["success"]),
                "failed_cases": sum(1 for r in results.values() if not r["success"]),
                "total_processing_time": total_time,
                "average_case_time": total_time / len(test_cases) if test_cases else 0
            },
            "results": results,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        logger.info(f"배치 Mann-Whitney U Test 완료 - 성공: {response['batch_summary']['successful_cases']}, 실패: {response['batch_summary']['failed_cases']}")
        return response

    except Exception as e:
        logger.error(f"배치 Mann-Whitney U Test 서버 오류: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"배치 테스트 처리 실패: {str(e)}"
        )


@router.post(
    "/compare-tests",
    summary="두 통계 테스트 비교",
    description="""
    동일한 데이터에 대해 Mann-Whitney U Test와 Kolmogorov-Smirnov Test를
    동시에 수행하여 결과를 비교합니다.

    **특징:**
    - 두 검정 방법을 동시에 적용
    - 결과 일관성 검증
    - 상호 보완적 분석 제공

    **사용 사례:**
    - 검정 방법 간 결과 비교
    - 분석 신뢰성 향상
    - 다양한 관점에서의 데이터 해석
    """,
    response_model=Dict[str, Any]
)
async def compare_statistical_tests(
    test_input: StatisticalTestInput
) -> Dict[str, Any]:
    """
    통계 테스트 비교 API 엔드포인트

    Args:
        test_input: 테스트 입력 데이터

    Returns:
        Dict[str, Any]: 두 검정 방법의 비교 결과
    """
    start_time = datetime.utcnow()

    try:
        logger.info("통계 테스트 비교 시작")

        # 두 검정 방법 동시에 수행
        mw_result = statistical_tests_service.mann_whitney_u_test(
            test_input.group_a,
            test_input.group_b,
            test_input.significance_level
        )

        ks_result = statistical_tests_service.kolmogorov_smirnov_test(
            test_input.group_a,
            test_input.group_b,
            test_input.significance_level
        )

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        # 결과 비교 및 해석
        both_significant = mw_result.significant and ks_result.significant
        either_significant = mw_result.significant or ks_result.significant
        conflicting_results = mw_result.significant != ks_result.significant

        interpretation = {
            "consistency": "일관됨" if not conflicting_results else "불일치",
            "confidence": "높음" if both_significant else ("중간" if either_significant else "낮음"),
            "recommendation": _generate_comparison_recommendation(
                mw_result, ks_result, conflicting_results
            )
        }

        response = {
            "success": True,
            "comparison_summary": {
                "data_size": {
                    "group_a": len(test_input.group_a),
                    "group_b": len(test_input.group_b)
                },
                "significance_level": test_input.significance_level,
                "both_significant": both_significant,
                "either_significant": either_significant,
                "conflicting_results": conflicting_results,
                "interpretation": interpretation
            },
            "mann_whitney_u": mw_result,
            "kolmogorov_smirnov": ks_result,
            "processing_time": processing_time,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        logger.info(f"통계 테스트 비교 완료 - 일관성: {interpretation['consistency']}, 신뢰도: {interpretation['confidence']}")
        return response

    except Exception as e:
        logger.error(f"통계 테스트 비교 서버 오류: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"테스트 비교 처리 실패: {str(e)}"
        )


def _generate_comparison_recommendation(
    mw_result: StatisticalTestResult,
    ks_result: StatisticalTestResult,
    conflicting: bool
) -> str:
    """비교 결과에 따른 추천 사항 생성"""
    if not conflicting:
        if mw_result.significant and ks_result.significant:
            return "두 검정 방법 모두에서 유의한 차이가 발견되었습니다. 결과가 매우 신뢰할 수 있습니다."
        else:
            return "두 검정 방법 모두에서 유의한 차이가 발견되지 않았습니다. 그룹 간 차이가 미미합니다."
    else:
        if mw_result.significant:
            return "Mann-Whitney U Test에서만 유의한 차이가 발견되었습니다. 중앙값 차이에 초점을 맞춰 분석하세요."
        else:
            return "Kolmogorov-Smirnov Test에서만 유의한 차이가 발견되었습니다. 분포 형태의 차이에 초점을 맞춰 분석하세요."


@router.get(
    "/health",
    summary="통계 테스트 API 상태 확인",
    description="통계 테스트 API의 상태를 확인합니다.",
    response_model=Dict[str, Any]
)
async def statistical_tests_health_check() -> Dict[str, Any]:
    """
    통계 테스트 API 상태 확인 엔드포인트

    Returns:
        Dict[str, Any]: 상태 정보
    """
    try:
        # 여러 테스트로 종합 상태 확인
        test_cases = [
            ([1.0, 2.0, 3.0], [1.1, 2.1, 3.1]),  # 유의한 차이 없음
            ([1.0, 2.0, 3.0], [5.0, 6.0, 7.0]),  # 유의한 차이 있음
        ]

        successful_tests = 0
        for group_a, group_b in test_cases:
            try:
                result = statistical_tests_service.mann_whitney_u_test(group_a, group_b, 0.05)
                if result.significant is not None:
                    successful_tests += 1
            except:
                pass

        health_status = "healthy" if successful_tests == len(test_cases) else "degraded"

        return {
            "status": health_status,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "service": "StatisticalTestsService",
            "version": "1.0.0",
            "tests_passed": successful_tests,
            "tests_total": len(test_cases),
            "uptime": "active"
        }

    except Exception as e:
        logger.error(f"통계 테스트 API 상태 확인 실패: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "service": "StatisticalTestsService",
            "error": str(e)
        }
