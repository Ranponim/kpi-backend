#!/usr/bin/env python3
"""
비동기 분석 API 라우터
LLM 분석을 비동기로 처리하고 상태를 관리합니다.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field

from ..services.async_analysis_service import async_analysis_service, AnalysisStatus

# 라우터 및 로거 설정
router = APIRouter()
logger = logging.getLogger(__name__)


class AsyncAnalysisRequest(BaseModel):
    """비동기 분석 요청 모델"""
    db_config: Dict[str, Any] = Field(..., description="데이터베이스 연결 정보")
    n_minus_1: str = Field(..., description="이전 기간 (예: 2025-01-01_00:00~2025-01-01_23:59)")
    n: str = Field(..., description="현재 기간 (예: 2025-01-02_00:00~2025-01-02_23:59)")
    ne_id: Optional[str] = Field(None, description="NE ID")
    cell_id: Optional[str] = Field(None, description="Cell ID")
    user_id: str = Field(default="default", description="사용자 ID")
    enable_mock: bool = Field(default=False, description="테스트 모드 여부")


class AnalysisStatusResponse(BaseModel):
    """분석 상태 응답 모델"""
    task_id: str
    analysis_id: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: int = Field(ge=0, le=100, description="진행률 (0-100)")
    error_message: Optional[str] = None
    request_params: Optional[Dict[str, Any]] = None
    result_data: Optional[Dict[str, Any]] = None


class AnalysisListResponse(BaseModel):
    """분석 목록 응답 모델"""
    tasks: List[AnalysisStatusResponse]
    total_count: int
    running_count: int


@router.post("/api/async-analysis/start", response_model=Dict[str, str], status_code=status.HTTP_202_ACCEPTED)
async def start_async_analysis(request: AsyncAnalysisRequest):
    """
    비동기 LLM 분석을 시작합니다.
    
    분석이 백그라운드에서 처리되며, 즉시 분석 ID를 반환합니다.
    상태 확인은 /api/async-analysis/status/{analysis_id} 엔드포인트를 사용하세요.
    """
    try:
        logger.info(f"비동기 분석 요청 시작: {request}")
        
        # 서비스 초기화
        await async_analysis_service.initialize()
        
        # 요청 파라미터 구성
        request_params = {
            "db_config": request.db_config,
            "n_minus_1": request.n_minus_1,
            "n": request.n,
            "ne_id": request.ne_id,
            "cell_id": request.cell_id,
            "user_id": request.user_id,
            "enable_mock": request.enable_mock
        }
        
        # 분석 작업 생성
        analysis_id = await async_analysis_service.create_analysis_task(
            request_params, 
            request.user_id
        )
        
        # 분석 시작
        started = await async_analysis_service.start_analysis(analysis_id)
        
        if not started:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="현재 처리 중인 작업이 많아 분석을 시작할 수 없습니다. 잠시 후 다시 시도해주세요."
            )
        
        return {
            "status": "started",
            "analysis_id": analysis_id,
            "message": "분석이 시작되었습니다. 상태 확인 API를 사용하여 진행 상황을 확인할 수 있습니다."
        }
        
    except Exception as e:
        logger.exception(f"비동기 분석 시작 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"분석 시작 중 오류 발생: {str(e)}"
        )


@router.get("/api/async-analysis/status/{analysis_id}", response_model=AnalysisStatusResponse)
async def get_analysis_status(analysis_id: str):
    """
    분석 상태를 조회합니다.
    
    Args:
        analysis_id: 분석 ID
        
    Returns:
        AnalysisStatusResponse: 분석 상태 정보
    """
    try:
        logger.info(f"분석 상태 조회: {analysis_id}")
        
        # 서비스 초기화
        await async_analysis_service.initialize()
        
        # 상태 조회
        status_data = await async_analysis_service.get_task_status(analysis_id)
        
        if not status_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"분석 ID를 찾을 수 없습니다: {analysis_id}"
            )
        
        return AnalysisStatusResponse(**status_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"분석 상태 조회 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"상태 조회 중 오류 발생: {str(e)}"
        )


@router.get("/api/async-analysis/list", response_model=AnalysisListResponse)
async def list_analysis_tasks():
    """
    실행 중인 분석 작업 목록을 조회합니다.
    
    Returns:
        AnalysisListResponse: 실행 중인 작업 목록
    """
    try:
        logger.info("분석 작업 목록 조회")
        
        # 서비스 초기화
        await async_analysis_service.initialize()
        
        # 실행 중인 작업 조회
        running_tasks = await async_analysis_service.get_running_tasks()
        
        # 응답 구성
        task_responses = [
            AnalysisStatusResponse(**task) for task in running_tasks
        ]
        
        return AnalysisListResponse(
            tasks=task_responses,
            total_count=len(task_responses),
            running_count=len([t for t in task_responses if t.status == AnalysisStatus.PROCESSING])
        )
        
    except Exception as e:
        logger.exception(f"분석 작업 목록 조회 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"작업 목록 조회 중 오류 발생: {str(e)}"
        )


@router.post("/api/async-analysis/cancel/{analysis_id}", response_model=Dict[str, str])
async def cancel_analysis(analysis_id: str):
    """
    분석 작업을 취소합니다.
    
    Args:
        analysis_id: 분석 ID
        
    Returns:
        Dict[str, str]: 취소 결과
    """
    try:
        logger.info(f"분석 취소 요청: {analysis_id}")
        
        # 서비스 초기화
        await async_analysis_service.initialize()
        
        # 작업 취소
        cancelled = await async_analysis_service.cancel_task(analysis_id)
        
        if not cancelled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="취소할 수 없는 작업입니다. 이미 완료되었거나 존재하지 않는 작업일 수 있습니다."
            )
        
        return {
            "status": "cancelled",
            "analysis_id": analysis_id,
            "message": "분석이 취소되었습니다."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"분석 취소 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"분석 취소 중 오류 발생: {str(e)}"
        )


@router.get("/api/async-analysis/result/{analysis_id}")
async def get_analysis_result(analysis_id: str):
    """
    완료된 분석 결과를 조회합니다.
    
    Args:
        analysis_id: 분석 ID
        
    Returns:
        Dict[str, Any]: 분석 결과
    """
    try:
        logger.info(f"분석 결과 조회: {analysis_id}")
        
        # 서비스 초기화
        await async_analysis_service.initialize()
        
        # 상태 조회
        status_data = await async_analysis_service.get_task_status(analysis_id)
        
        if not status_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"분석 ID를 찾을 수 없습니다: {analysis_id}"
            )
        
        # 완료되지 않은 작업인 경우
        if status_data["status"] not in [AnalysisStatus.COMPLETED]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"분석이 아직 완료되지 않았습니다. 현재 상태: {status_data['status']}"
            )
        
        # 실패한 작업인 경우
        if status_data["status"] == AnalysisStatus.FAILED:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"분석이 실패했습니다: {status_data.get('error_message', '알 수 없는 오류')}"
            )
        
        # 결과 반환
        return {
            "analysis_id": analysis_id,
            "status": status_data["status"],
            "completed_at": status_data["completed_at"],
            "result": status_data.get("result_data", {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"분석 결과 조회 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"결과 조회 중 오류 발생: {str(e)}"
        )


@router.get("/api/async-analysis/health")
async def health_check():
    """
    비동기 분석 서비스 상태를 확인합니다.
    
    Returns:
        Dict[str, Any]: 서비스 상태 정보
    """
    try:
        # 서비스 초기화
        await async_analysis_service.initialize()
        
        # 실행 중인 작업 수 조회
        running_tasks = await async_analysis_service.get_running_tasks()
        running_count = len([t for t in running_tasks if t["status"] == AnalysisStatus.PROCESSING])
        pending_count = len([t for t in running_tasks if t["status"] == AnalysisStatus.PENDING])
        
        return {
            "status": "healthy",
            "service": "async_analysis_service",
            "max_concurrent_tasks": async_analysis_service.max_concurrent_tasks,
            "running_tasks": running_count,
            "pending_tasks": pending_count,
            "total_active_tasks": len(running_tasks),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.exception(f"헬스 체크 실패: {e}")
        return {
            "status": "unhealthy",
            "service": "async_analysis_service",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


