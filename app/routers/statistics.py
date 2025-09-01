"""
Statistics 비교 분석 API 라우터

이 모듈은 두 개의 날짜 구간에 대한 KPI 데이터 비교 분석을 위한
FastAPI 엔드포인트를 제공합니다.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from fastapi.responses import JSONResponse
import logging

from ..db import get_database
from ..models.statistics import (
    StatisticsCompareRequest, StatisticsCompareResponse, StatisticsCompareError,
    PegComparisonResult
)
from ..utils.statistics_db import StatisticsDataBase
from ..utils.statistics_analyzer import StatisticsAnalyzer, validate_data_consistency
from ..exceptions import DatabaseConnectionException, InvalidAnalysisDataException

# 로거 설정
logger = logging.getLogger(__name__)

# 라우터 생성
router = APIRouter(
    prefix="/api/statistics",
    tags=["statistics"],
    responses={
        500: {"model": StatisticsCompareError, "description": "서버 내부 오류"},
        400: {"model": StatisticsCompareError, "description": "잘못된 요청"},
        404: {"model": StatisticsCompareError, "description": "데이터 없음"}
    }
)

async def get_statistics_db(db=Depends(get_database)) -> StatisticsDataBase:
    """Statistics 데이터베이스 의존성 주입"""
    return StatisticsDataBase(db)

@router.post(
    "/compare",
    response_model=StatisticsCompareResponse,
    summary="두 기간 KPI 데이터 비교 분석",
    description="""
    두 개의 날짜 구간에 대한 KPI 데이터를 비교 분석하여 다음 지표들을 계산합니다:
    
    **분석 지표:**
    - 평균값 (Mean)
    - 델타 (Delta): period2 - period1
    - 델타 백분율 (Delta Percentage)
    - RSD (Relative Standard Deviation): 상대표준편차
    - 통계적 유의성 (t-검정)
    - 개선/악화 상태 판정
    
    **필터 옵션:**
    - NE 필터링
    - Cell ID 필터링  
    - 이상치 포함/제외
    - 소수점 자릿수 설정
    """,
    responses={
        200: {
            "description": "비교 분석 성공",
            "content": {
                "application/json": {
                    "example": {
                        "request_summary": {
                            "period1": "2025-08-01 to 2025-08-07",
                            "period2": "2025-08-08 to 2025-08-14",
                            "peg_count": 3
                        },
                        "analysis_results": [
                            {
                                "peg_name": "availability",
                                "delta": 0.2,
                                "delta_percentage": 0.201,
                                "improvement_status": "improved"
                            }
                        ],
                        "summary": {
                            "total_pegs_analyzed": 3,
                            "improved_count": 2,
                            "degraded_count": 1
                        }
                    }
                }
            }
        }
    }
)
async def compare_statistics(
    request: StatisticsCompareRequest,
    background_tasks: BackgroundTasks,
    stats_db: StatisticsDataBase = Depends(get_statistics_db)
) -> StatisticsCompareResponse:
    """
    두 기간의 KPI 데이터 비교 분석 API
    
    Args:
        request: 비교 분석 요청 데이터
        background_tasks: 백그라운드 작업 (로깅, 캐싱 등)
        stats_db: Statistics 데이터베이스 인스턴스
        
    Returns:
        비교 분석 결과
        
    Raises:
        HTTPException: 요청 오류, 데이터 부족, 서버 오류 등
    """
    start_time = datetime.utcnow()
    
    try:
        logger.info("Statistics 비교 분석 요청 시작")
        logger.info(f"요청 파라미터: period1={request.period1.start_date}~{request.period1.end_date}, "
                   f"period2={request.period2.start_date}~{request.period2.end_date}, "
                   f"PEGs={request.peg_names}")
        
        # 1. 데이터 가용성 검증
        await validate_request_data_availability(request, stats_db)
        
        # 2. 두 기간 데이터 조회
        period1_data, period2_data = await fetch_comparison_data(request, stats_db)
        
        # 3. 데이터 일관성 검증  
        validation_result = validate_data_consistency(period1_data, period2_data, request.peg_names)
        
        if not validation_result["is_consistent"]:
            logger.warning(f"데이터 일관성 문제: {validation_result}")
            
            # 사용 가능한 PEG만으로 분석 진행
            available_pegs = validation_result["available_pegs"]
            if not available_pegs:
                raise InvalidAnalysisDataException(
                    "분석할 수 있는 공통 PEG 데이터가 없습니다",
                    validation_errors=[validation_result]
                )
            
            logger.info(f"사용 가능한 PEG로 분석 진행: {available_pegs}")
            request.peg_names = available_pegs
        
        # 4. 통계 분석 수행
        analyzer = StatisticsAnalyzer(decimal_places=request.decimal_places)
        comparison_results = analyzer.calculate_comparison_statistics(
            period1_data, period2_data, request.peg_names
        )
        
        if not comparison_results:
            raise InvalidAnalysisDataException(
                "분석 결과를 생성할 수 없습니다",
                validation_errors=["insufficient_data", validation_result]
            )
        
        # 5. 요약 통계 계산
        summary_stats = analyzer.calculate_summary_statistics(comparison_results)
        
        # 6. 응답 생성
        response = create_comparison_response(
            request, comparison_results, summary_stats, 
            analyzer, start_time, validation_result
        )
        
        # 7. 백그라운드 작업 등록 (로깅, 메트릭 등)
        background_tasks.add_task(
            log_analysis_metrics, 
            request, len(comparison_results), start_time
        )
        
        logger.info(f"Statistics 비교 분석 완료 - PEG 수: {len(comparison_results)}")
        return response
        
    except InvalidAnalysisDataException as e:
        logger.error(f"요청 검증 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=StatisticsCompareError(
                error_code="VALIDATION_ERROR",
                error_message=str(e),
                details=getattr(e, 'details', None),
                suggestions=[
                    "날짜 범위를 확인해주세요",
                    "PEG 이름을 확인해주세요", 
                    "필터 조건을 완화해보세요"
                ]
            ).dict()
        )
        
    except DatabaseConnectionException as e:
        logger.error(f"데이터베이스 오류: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=StatisticsCompareError(
                error_code="DATABASE_ERROR",
                error_message="데이터베이스 연결 오류가 발생했습니다",
                details={"original_error": str(e)},
                suggestions=["잠시 후 다시 시도해주세요"]
            ).dict()
        )
        
    except Exception as e:
        logger.error(f"Statistics 비교 분석 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=StatisticsCompareError(
                error_code="INTERNAL_ERROR",
                error_message="서버 내부 오류가 발생했습니다",
                details={"error_type": type(e).__name__},
                suggestions=["잠시 후 다시 시도해주세요"]
            ).dict()
        )

async def validate_request_data_availability(
    request: StatisticsCompareRequest,
    stats_db: StatisticsDataBase
) -> None:
    """
    요청된 데이터의 가용성을 사전 검증
    
    Args:
        request: 분석 요청
        stats_db: 데이터베이스 인스턴스
        
    Raises:
        InvalidAnalysisDataException: 데이터 부족 시
    """
    logger.info("데이터 가용성 사전 검증 시작")
    
    # 각 기간별 데이터 가용성 확인
    period1_availability = await stats_db.validate_data_availability(
        request.period1.start_date, request.period1.end_date, request.peg_names,
        request.ne_filter, request.cell_id_filter
    )
    
    period2_availability = await stats_db.validate_data_availability(
        request.period2.start_date, request.period2.end_date, request.peg_names,
        request.ne_filter, request.cell_id_filter
    )
    
    # 최소 데이터 요구사항 확인
    if period1_availability["total_data_points"] == 0:
        raise InvalidAnalysisDataException(
            "첫 번째 기간에 분석할 데이터가 없습니다",
            validation_errors=[period1_availability, request.peg_names]
        )
    
    if period2_availability["total_data_points"] == 0:
        raise InvalidAnalysisDataException(
            "두 번째 기간에 분석할 데이터가 없습니다",
            validation_errors=[period2_availability, request.peg_names]
        )
    
    # 공통 PEG 확인
    common_pegs = set(period1_availability["available_pegs"]) & set(period2_availability["available_pegs"])
    if not common_pegs:
        raise InvalidAnalysisDataException(
            "두 기간 모두에 존재하는 공통 PEG가 없습니다",
            validation_errors=[
                period1_availability["available_pegs"],
                period2_availability["available_pegs"],
                request.peg_names
            ]
        )
    
    logger.info(f"데이터 가용성 검증 완료 - 공통 PEG: {len(common_pegs)}개")

async def fetch_comparison_data(
    request: StatisticsCompareRequest,
    stats_db: StatisticsDataBase
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    비교 분석을 위한 두 기간 데이터 조회
    
    Args:
        request: 분석 요청
        stats_db: 데이터베이스 인스턴스
        
    Returns:
        (period1_data, period2_data) 튜플
    """
    logger.info("비교 데이터 조회 시작")
    
    # 병렬로 두 기간 데이터 조회
    period1_data = await stats_db.get_period_data(
        request.period1.start_date, request.period1.end_date,
        request.peg_names, request.ne_filter, request.cell_id_filter,
        request.include_outliers
    )
    
    period2_data = await stats_db.get_period_data(
        request.period2.start_date, request.period2.end_date,
        request.peg_names, request.ne_filter, request.cell_id_filter,
        request.include_outliers
    )
    
    logger.info(f"데이터 조회 완료 - Period1: {len(period1_data)}개, Period2: {len(period2_data)}개")
    return period1_data, period2_data

def create_comparison_response(
    request: StatisticsCompareRequest,
    comparison_results: List[PegComparisonResult],
    summary_stats: Dict[str, Any],
    analyzer: StatisticsAnalyzer,
    start_time: datetime,
    validation_result: Dict[str, Any]
) -> StatisticsCompareResponse:
    """
    비교 분석 응답 생성
    
    Args:
        request: 원본 요청
        comparison_results: 비교 분석 결과
        summary_stats: 요약 통계
        analyzer: 분석기 인스턴스
        start_time: 분석 시작 시간
        validation_result: 데이터 검증 결과
        
    Returns:
        완성된 응답 객체
    """
    logger.info("비교 분석 응답 생성")
    
    # 요청 요약 생성
    request_summary = {
        "period1": f"{request.period1.start_date.strftime('%Y-%m-%d')} to {request.period1.end_date.strftime('%Y-%m-%d')}",
        "period2": f"{request.period2.start_date.strftime('%Y-%m-%d')} to {request.period2.end_date.strftime('%Y-%m-%d')}",
        "peg_count": len(request.peg_names),
        "requested_pegs": request.peg_names,
        "filter_applied": bool(request.ne_filter or request.cell_id_filter),
        "ne_filter": request.ne_filter,
        "cell_id_filter": request.cell_id_filter,
        "include_outliers": request.include_outliers,
        "decimal_places": request.decimal_places
    }
    
    # 필터 정보
    filters_applied = {
        "ne_filter": request.ne_filter,
        "cell_id_filter": request.cell_id_filter,
        "include_outliers": request.include_outliers
    }
    
    # 메타데이터 생성
    metadata = analyzer.generate_analysis_metadata(
        start_time, 
        validation_result["period1_data_count"],
        validation_result["period2_data_count"],
        request.peg_names,
        filters_applied
    )
    
    # 응답 객체 생성
    response = StatisticsCompareResponse(
        request_summary=request_summary,
        analysis_results=comparison_results,
        summary=summary_stats,
        metadata=metadata
    )
    
    return response

async def log_analysis_metrics(
    request: StatisticsCompareRequest,
    results_count: int,
    start_time: datetime
) -> None:
    """
    분석 메트릭 로깅 (백그라운드 작업)
    
    Args:
        request: 분석 요청
        results_count: 결과 수
        start_time: 시작 시간
    """
    try:
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        logger.info(f"📊 Statistics 분석 메트릭:")
        logger.info(f"  - 처리 시간: {processing_time:.3f}초")
        logger.info(f"  - 분석된 PEG: {results_count}개")
        logger.info(f"  - 요청된 PEG: {len(request.peg_names)}개")
        logger.info(f"  - 필터 적용: {bool(request.ne_filter or request.cell_id_filter)}")
        logger.info(f"  - 이상치 포함: {request.include_outliers}")
        
        # 향후 메트릭 저장소(예: InfluxDB, Prometheus)에 전송 가능
        
    except Exception as e:
        logger.error(f"메트릭 로깅 실패: {e}")

@router.get(
    "/health",
    summary="Statistics API 상태 확인",
    description="Statistics 비교 분석 API의 상태와 데이터베이스 연결을 확인합니다.",
    response_model=Dict[str, Any]
)
async def statistics_health_check(
    stats_db: StatisticsDataBase = Depends(get_statistics_db)
) -> Dict[str, Any]:
    """
    Statistics API 상태 확인 엔드포인트
    
    Returns:
        API 상태 정보
    """
    try:
        logger.info("Statistics API 상태 확인 시작")
        
        # 데이터베이스 연결 테스트
        test_start = datetime(2025, 8, 1)
        test_end = datetime(2025, 8, 2)
        availability = await stats_db.validate_data_availability(
            test_start, test_end, ['availability']
        )
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "database_connection": "ok",
            "available_data_points": availability.get("total_data_points", 0),
            "api_version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"Statistics API 상태 확인 실패: {e}")
        return {
            "status": "unhealthy", 
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "database_connection": "failed",
            "error": str(e),
            "api_version": "1.0.0"
        }

# Mock 데이터 생성 엔드포인트 (개발/테스트용)
@router.post(
    "/mock-data",
    summary="Mock 데이터 생성 (개발/테스트용)",
    description="Statistics 분석 테스트를 위한 Mock 데이터를 생성합니다.",
    response_model=Dict[str, Any]
)
async def generate_mock_data(
    count: int = 1000,
    stats_db: StatisticsDataBase = Depends(get_statistics_db)
) -> Dict[str, Any]:
    """
    Mock 데이터 생성 엔드포인트 (개발/테스트용)
    
    Args:
        count: 생성할 데이터 포인트 수
        stats_db: 데이터베이스 인스턴스
        
    Returns:
        생성 결과
    """
    try:
        logger.info(f"Mock 데이터 생성 시작: {count}개")
        
        from ..utils.statistics_db import create_sample_data
        await create_sample_data(stats_db.db, count)
        
        return {
            "status": "success",
            "message": f"Mock 데이터 {count}개 생성 완료",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        logger.error(f"Mock 데이터 생성 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Mock 데이터 생성 실패: {str(e)}"
        )
