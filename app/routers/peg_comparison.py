"""
PEG 비교분석 API 라우터

이 모듈은 PEG(Performance Engineering Guidelines) 비교분석 기능을 위한 
API 엔드포인트들을 정의합니다.
"""

import logging
from fastapi import APIRouter, HTTPException, status, Query, Depends, BackgroundTasks
from typing import Optional, Dict, Any
from datetime import datetime
import asyncio

from ..models.peg_comparison import (
    PEGComparisonRequest,
    PEGComparisonResponse,
    AsyncTaskResponse,
    PEGComparisonAnalysisModel
)
from ..exceptions.peg_comparison_exceptions import (
    PEGComparisonException,
    MCPConnectionError,
    DataValidationError,
    AnalysisDataNotFoundError,
    AsyncTaskNotFoundError,
    RateLimitExceededError,
    PermissionDeniedError
)
from ..services.mcp_client_service import get_mcp_client
from ..services.peg_comparison_cache_service import get_peg_comparison_cache_service
from ..services.async_task_service import get_async_task_service, TaskStatus
from ..db import get_analysis_collection
from ..middleware.request_tracing import create_request_context_logger, log_business_event

# 로깅 설정
logger = logging.getLogger("app.routers.peg_comparison")

# 라우터 생성
router = APIRouter(
    prefix="/api/analysis/results",
    tags=["PEG Comparison Analysis"],
    responses={
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not Found"},
        429: {"description": "Too Many Requests"},
        500: {"description": "Internal Server Error"},
        503: {"description": "Service Unavailable"}
    }
)


async def _get_analysis_data(analysis_id: str) -> Dict[str, Any]:
    """
    분석 데이터 조회
    
    Args:
        analysis_id: 분석 ID
        
    Returns:
        Dict[str, Any]: 분석 데이터
        
    Raises:
        AnalysisDataNotFoundError: 분석 데이터를 찾을 수 없는 경우
    """
    try:
        collection = get_analysis_collection()
        
        # 분석 결과 조회
        analysis_result = await collection.find_one({"_id": analysis_id})
        
        if not analysis_result:
            raise AnalysisDataNotFoundError(analysis_id)
        
        # MCP 서버에 전달할 데이터 구성
        raw_data = {
            "stats": analysis_result.get("stats", {}),
            "peg_definitions": analysis_result.get("request_params", {}).get("peg_definitions", {}),
            "period_info": {
                "n1_start": analysis_result.get("request_params", {}).get("n1_start"),
                "n1_end": analysis_result.get("request_params", {}).get("n1_end"),
                "n_start": analysis_result.get("request_params", {}).get("n_start"),
                "n_end": analysis_result.get("request_params", {}).get("n_end"),
            }
        }
        
        return raw_data
        
    except AnalysisDataNotFoundError:
        raise
    
    except Exception as e:
        logger.error(f"분석 데이터 조회 실패: {analysis_id}", extra={
            "analysis_id": analysis_id,
            "error_type": type(e).__name__,
            "error_message": str(e)
        }, exc_info=True)
        
        raise AnalysisDataNotFoundError(
            analysis_id,
            message=f"분석 데이터 조회 중 오류 발생: {str(e)}"
        )


async def _process_peg_comparison_async(
    task_id: str,
    analysis_id: str,
    request_data: PEGComparisonRequest
) -> None:
    """
    비동기 PEG 비교분석 처리
    
    Args:
        task_id: 작업 ID
        analysis_id: 분석 ID
        request_data: 요청 데이터
    """
    try:
        async_task_service = await get_async_task_service()
        mcp_client = await get_mcp_client()
        cache_service = await get_peg_comparison_cache_service()
        
        # 작업 진행률 업데이트
        await async_task_service.update_task_progress(
            task_id, 10, TaskStatus.PROGRESS, "분석 데이터 조회 중..."
        )
        
        # 분석 데이터 조회
        raw_data = await _get_analysis_data(analysis_id)
        
        await async_task_service.update_task_progress(
            task_id, 30, TaskStatus.PROGRESS, "MCP 서버에 분석 요청 중..."
        )
        
        # MCP 서버 호출
        mcp_response = await mcp_client.call_peg_comparison(
            analysis_id=analysis_id,
            raw_data=raw_data,
            options={
                "include_metadata": request_data.include_metadata,
                "cache_ttl": request_data.cache_ttl,
                "algorithm_version": request_data.algorithm_version
            }
        )
        
        await async_task_service.update_task_progress(
            task_id, 70, TaskStatus.PROGRESS, "결과 처리 중..."
        )
        
        # 결과를 PEGComparisonAnalysisModel로 변환
        result = PEGComparisonAnalysisModel(**mcp_response.data)
        
        await async_task_service.update_task_progress(
            task_id, 90, TaskStatus.PROGRESS, "결과 캐싱 중..."
        )
        
        # 결과 캐싱
        await cache_service.cache_peg_comparison_result(analysis_id, result)
        
        # 작업 완료
        await async_task_service.complete_task(
            task_id,
            result_data=result.model_dump()
        )
        
        logger.info(f"비동기 PEG 비교분석 완료: {analysis_id}", extra={
            "task_id": task_id,
            "analysis_id": analysis_id,
            "processing_time": mcp_response.processing_time
        })
        
    except Exception as e:
        logger.error(f"비동기 PEG 비교분석 실패: {analysis_id}", extra={
            "task_id": task_id,
            "analysis_id": analysis_id,
            "error_type": type(e).__name__,
            "error_message": str(e)
        }, exc_info=True)
        
        # 작업 실패 처리
        async_task_service = await get_async_task_service()
        await async_task_service.fail_task(
            task_id,
            f"PEG 비교분석 실패: {str(e)}",
            {"error_type": type(e).__name__, "analysis_id": analysis_id}
        )


@router.get(
    "/{analysis_id}/peg-comparison",
    response_model=PEGComparisonResponse,
    summary="PEG 비교분석 결과 조회",
    description="특정 분석 ID에 대한 PEG 비교분석 결과를 조회합니다."
)
async def get_peg_comparison(
    analysis_id: str,
    include_metadata: bool = Query(True, description="메타데이터 포함 여부"),
    cache_ttl: int = Query(3600, ge=60, le=86400, description="캐시 TTL (초)"),
    async_processing: bool = Query(False, description="비동기 처리 여부"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    PEG 비교분석 결과 조회
    
    - **analysis_id**: 분석 ID
    - **include_metadata**: 메타데이터 포함 여부
    - **cache_ttl**: 캐시 TTL (초)
    - **async_processing**: 비동기 처리 여부
    """
    try:
        req_logger = create_request_context_logger("app.peg_comparison.get")
        
        req_logger.info("PEG 비교분석 요청 시작", extra={
            "analysis_id": analysis_id,
            "include_metadata": include_metadata,
            "cache_ttl": cache_ttl,
            "async_processing": async_processing
        })
        
        # 비즈니스 이벤트 로깅
        log_business_event("peg_comparison_request_started", {
            "analysis_id": analysis_id,
            "async_processing": async_processing
        })
        
        start_time = datetime.utcnow()
        
        # 캐시 서비스 조회
        cache_service = await get_peg_comparison_cache_service()
        
        # 캐시된 결과 확인
        cached_result = await cache_service.get_cached_peg_comparison_result(analysis_id)
        
        if cached_result:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            req_logger.info("캐시된 PEG 비교분석 결과 반환", extra={
                "analysis_id": analysis_id,
                "processing_time": processing_time,
                "cached": True
            })
            
            return PEGComparisonResponse(
                success=True,
                data=cached_result,
                processing_time=processing_time,
                cached=True,
                mcp_version=cached_result.mcp_version
            )
        
        # 비동기 처리 요청인 경우
        if async_processing:
            async_task_service = await get_async_task_service()
            
            # 새 작업 생성
            task_id = await async_task_service.create_task(
                analysis_id=analysis_id,
                task_type="peg_comparison"
            )
            
            # 백그라운드 작업으로 처리 시작
            request_data = PEGComparisonRequest(
                analysis_id=analysis_id,
                include_metadata=include_metadata,
                cache_ttl=cache_ttl,
                async_processing=True
            )
            
            background_tasks.add_task(
                _process_peg_comparison_async,
                task_id,
                analysis_id,
                request_data
            )
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            req_logger.info("비동기 PEG 비교분석 작업 시작", extra={
                "analysis_id": analysis_id,
                "task_id": task_id,
                "processing_time": processing_time
            })
            
            return PEGComparisonResponse(
                success=True,
                data=None,
                processing_time=processing_time,
                cached=False,
                mcp_version="v2.1.0",
                message=f"비동기 작업이 시작되었습니다. 작업 ID: {task_id}"
            )
        
        # 동기 처리
        req_logger.info("동기 PEG 비교분석 처리 시작", extra={
            "analysis_id": analysis_id
        })
        
        # 분석 데이터 조회
        raw_data = await _get_analysis_data(analysis_id)
        
        # MCP 서버 호출
        mcp_client = await get_mcp_client()
        mcp_response = await mcp_client.call_peg_comparison(
            analysis_id=analysis_id,
            raw_data=raw_data,
            options={
                "include_metadata": include_metadata,
                "cache_ttl": cache_ttl,
                "algorithm_version": "v2.1.0"
            }
        )
        
        # 결과를 PEGComparisonAnalysisModel로 변환
        result = PEGComparisonAnalysisModel(**mcp_response.data)
        
        # 결과 캐싱
        await cache_service.cache_peg_comparison_result(analysis_id, result)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        req_logger.info("PEG 비교분석 완료", extra={
            "analysis_id": analysis_id,
            "processing_time": processing_time,
            "mcp_processing_time": mcp_response.processing_time,
            "cached": False
        })
        
        # 데이터 이벤트 로깅
        log_business_event("peg_comparison_completed", {
            "analysis_id": analysis_id,
            "processing_time": processing_time,
            "mcp_processing_time": mcp_response.processing_time
        })
        
        return PEGComparisonResponse(
            success=True,
            data=result,
            processing_time=processing_time,
            cached=False,
            mcp_version=mcp_response.algorithm_version
        )
        
    except (AnalysisDataNotFoundError, DataValidationError, MCPConnectionError) as e:
        # 이미 정의된 예외는 그대로 전파
        raise e
        
    except Exception as e:
        req_logger.error("PEG 비교분석 중 예상치 못한 오류", extra={
            "analysis_id": analysis_id,
            "error_type": type(e).__name__,
            "error_message": str(e)
        }, exc_info=True)
        
        raise PEGComparisonException(
            message=f"PEG 비교분석 실패: {str(e)}",
            details={"analysis_id": analysis_id, "error_type": type(e).__name__}
        )


@router.get(
    "/{analysis_id}/peg-comparison/status",
    response_model=AsyncTaskResponse,
    summary="비동기 작업 상태 조회",
    description="비동기 PEG 비교분석 작업의 상태를 조회합니다."
)
async def get_async_task_status(
    analysis_id: str,
    task_id: Optional[str] = Query(None, description="작업 ID (선택사항)")
):
    """
    비동기 작업 상태 조회
    
    - **analysis_id**: 분석 ID
    - **task_id**: 작업 ID (선택사항)
    """
    try:
        req_logger = create_request_context_logger("app.peg_comparison.status")
        
        req_logger.info("비동기 작업 상태 조회 시작", extra={
            "analysis_id": analysis_id,
            "task_id": task_id
        })
        
        async_task_service = await get_async_task_service()
        
        # task_id가 제공되지 않은 경우 analysis_id로 활성 작업 찾기
        if not task_id:
            active_tasks = await async_task_service.get_active_tasks()
            matching_tasks = [
                task for task in active_tasks 
                if task.get("analysis_id") == analysis_id
            ]
            
            if not matching_tasks:
                raise AsyncTaskNotFoundError("unknown", message=f"분석 ID {analysis_id}에 대한 활성 작업을 찾을 수 없습니다")
            
            # 가장 최근 작업 선택
            task_id = matching_tasks[0]["task_id"]
        
        # 작업 상태 조회
        task_status = await async_task_service.get_task_status(task_id)
        
        if not task_status:
            raise AsyncTaskNotFoundError(task_id)
        
        processing_time = (datetime.utcnow() - task_status["created_at"]).total_seconds()
        
        req_logger.info("비동기 작업 상태 조회 완료", extra={
            "task_id": task_id,
            "analysis_id": analysis_id,
            "status": task_status["status"],
            "progress": task_status["progress"]
        })
        
        return AsyncTaskResponse(
            success=True,
            task_id=task_id,
            status=task_status["status"],
            progress=task_status["progress"],
            estimated_completion=task_status.get("estimated_completion"),
            message="작업 상태 조회 성공"
        )
        
    except AsyncTaskNotFoundError:
        raise
        
    except Exception as e:
        req_logger.error("비동기 작업 상태 조회 실패", extra={
            "analysis_id": analysis_id,
            "task_id": task_id,
            "error_type": type(e).__name__,
            "error_message": str(e)
        }, exc_info=True)
        
        raise PEGComparisonException(
            message=f"작업 상태 조회 실패: {str(e)}",
            details={"analysis_id": analysis_id, "task_id": task_id}
        )


@router.post(
    "/{analysis_id}/peg-comparison/cancel",
    summary="비동기 작업 취소",
    description="진행 중인 비동기 PEG 비교분석 작업을 취소합니다."
)
async def cancel_async_task(
    analysis_id: str,
    task_id: Optional[str] = Query(None, description="작업 ID (선택사항)")
):
    """
    비동기 작업 취소
    
    - **analysis_id**: 분석 ID
    - **task_id**: 작업 ID (선택사항)
    """
    try:
        req_logger = create_request_context_logger("app.peg_comparison.cancel")
        
        req_logger.info("비동기 작업 취소 요청", extra={
            "analysis_id": analysis_id,
            "task_id": task_id
        })
        
        async_task_service = await get_async_task_service()
        
        # task_id가 제공되지 않은 경우 analysis_id로 활성 작업 찾기
        if not task_id:
            active_tasks = await async_task_service.get_active_tasks()
            matching_tasks = [
                task for task in active_tasks 
                if task.get("analysis_id") == analysis_id
            ]
            
            if not matching_tasks:
                raise AsyncTaskNotFoundError("unknown", message=f"분석 ID {analysis_id}에 대한 활성 작업을 찾을 수 없습니다")
            
            # 가장 최근 작업 선택
            task_id = matching_tasks[0]["task_id"]
        
        # 작업 취소
        success = await async_task_service.cancel_task(task_id)
        
        if success:
            req_logger.info("비동기 작업 취소 완료", extra={
                "task_id": task_id,
                "analysis_id": analysis_id
            })
            
            return {
                "success": True,
                "message": f"작업 {task_id}이(가) 취소되었습니다.",
                "task_id": task_id,
                "analysis_id": analysis_id
            }
        else:
            return {
                "success": False,
                "message": "작업을 취소할 수 없습니다. (이미 완료되었거나 실패한 작업)",
                "task_id": task_id,
                "analysis_id": analysis_id
            }
        
    except AsyncTaskNotFoundError:
        raise
        
    except Exception as e:
        req_logger.error("비동기 작업 취소 실패", extra={
            "analysis_id": analysis_id,
            "task_id": task_id,
            "error_type": type(e).__name__,
            "error_message": str(e)
        }, exc_info=True)
        
        raise PEGComparisonException(
            message=f"작업 취소 실패: {str(e)}",
            details={"analysis_id": analysis_id, "task_id": task_id}
        )


@router.get(
    "/peg-comparison/cache/stats",
    summary="PEG 비교분석 캐시 통계",
    description="PEG 비교분석 캐시 시스템의 통계를 조회합니다."
)
async def get_peg_comparison_cache_stats():
    """PEG 비교분석 캐시 통계 조회"""
    try:
        req_logger = create_request_context_logger("app.peg_comparison.cache_stats")
        
        req_logger.info("PEG 비교분석 캐시 통계 조회 시작")
        
        cache_service = await get_peg_comparison_cache_service()
        stats = await cache_service.get_cache_statistics()
        
        req_logger.info("PEG 비교분석 캐시 통계 조회 완료", extra={
            "total_peg_cache_keys": stats.get("peg_comparison_cache", {}).get("total_peg_cache_keys", 0)
        })
        
        return {
            "success": True,
            "data": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        req_logger.error("PEG 비교분석 캐시 통계 조회 실패", extra={
            "error_type": type(e).__name__,
            "error_message": str(e)
        }, exc_info=True)
        
        raise PEGComparisonException(
            message=f"캐시 통계 조회 실패: {str(e)}"
        )


@router.post(
    "/{analysis_id}/peg-comparison/cache/clear",
    summary="PEG 비교분석 캐시 정리",
    description="특정 분석 ID의 PEG 비교분석 캐시를 정리합니다."
)
async def clear_peg_comparison_cache(analysis_id: str):
    """PEG 비교분석 캐시 정리"""
    try:
        req_logger = create_request_context_logger("app.peg_comparison.cache_clear")
        
        req_logger.info("PEG 비교분석 캐시 정리 시작", extra={
            "analysis_id": analysis_id
        })
        
        cache_service = await get_peg_comparison_cache_service()
        deleted_count = await cache_service.invalidate_peg_comparison_cache(analysis_id)
        
        req_logger.info("PEG 비교분석 캐시 정리 완료", extra={
            "analysis_id": analysis_id,
            "deleted_count": deleted_count
        })
        
        return {
            "success": True,
            "message": f"캐시 정리 완료: {deleted_count}개 항목 삭제",
            "analysis_id": analysis_id,
            "deleted_count": deleted_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        req_logger.error("PEG 비교분석 캐시 정리 실패", extra={
            "analysis_id": analysis_id,
            "error_type": type(e).__name__,
            "error_message": str(e)
        }, exc_info=True)
        
        raise PEGComparisonException(
            message=f"캐시 정리 실패: {str(e)}",
            details={"analysis_id": analysis_id}
        )


@router.get(
    "/peg-comparison/active-tasks",
    summary="활성 작업 목록 조회",
    description="현재 진행 중인 비동기 PEG 비교분석 작업 목록을 조회합니다."
)
async def get_active_tasks():
    """활성 작업 목록 조회"""
    try:
        req_logger = create_request_context_logger("app.peg_comparison.active_tasks")
        
        req_logger.info("활성 작업 목록 조회 시작")
        
        async_task_service = await get_async_task_service()
        active_tasks = await async_task_service.get_active_tasks()
        
        req_logger.info("활성 작업 목록 조회 완료", extra={
            "active_task_count": len(active_tasks)
        })
        
        return {
            "success": True,
            "data": {
                "active_tasks": active_tasks,
                "total_count": len(active_tasks)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        req_logger.error("활성 작업 목록 조회 실패", extra={
            "error_type": type(e).__name__,
            "error_message": str(e)
        }, exc_info=True)
        
        raise PEGComparisonException(
            message=f"활성 작업 목록 조회 실패: {str(e)}"
        )


@router.post(
    "/peg-comparison/cleanup",
    summary="만료된 작업 정리",
    description="만료된 비동기 작업들을 정리합니다."
)
async def cleanup_expired_tasks():
    """만료된 작업 정리"""
    try:
        req_logger = create_request_context_logger("app.peg_comparison.cleanup")
        
        req_logger.info("만료된 작업 정리 시작")
        
        async_task_service = await get_async_task_service()
        cleanup_stats = await async_task_service.cleanup_expired_tasks()
        
        req_logger.info("만료된 작업 정리 완료", extra={
            "total_expired": cleanup_stats.get("total_expired", 0),
            "failed_tasks": cleanup_stats.get("failed_tasks", 0),
            "cancelled_tasks": cleanup_stats.get("cancelled_tasks", 0)
        })
        
        return {
            "success": True,
            "data": cleanup_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        req_logger.error("만료된 작업 정리 실패", extra={
            "error_type": type(e).__name__,
            "error_message": str(e)
        }, exc_info=True)
        
        raise PEGComparisonException(
            message=f"작업 정리 실패: {str(e)}"
        )















