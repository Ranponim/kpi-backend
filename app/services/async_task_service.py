"""
비동기 작업 관리 서비스

이 모듈은 PEG 비교분석의 비동기 작업을 관리하는 서비스입니다.
작업 상태 추적, 진행률 업데이트, 결과 저장 등을 담당합니다.
"""

import logging
import asyncio
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum

from ..models.peg_comparison import AsyncTaskStatus, PEGComparisonAnalysisModel
from ..exceptions.peg_comparison_exceptions import (
    AsyncTaskError,
    AsyncTaskNotFoundError,
    ProcessingTimeoutError
)
from .peg_comparison_cache_service import get_peg_comparison_cache_service

logger = logging.getLogger("app.services.async_task")


class TaskStatus(Enum):
    """작업 상태 열거형"""
    PENDING = "PENDING"
    PROGRESS = "PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class AsyncTaskService:
    """비동기 작업 관리 서비스 클래스"""
    
    def __init__(self):
        """비동기 작업 서비스 초기화"""
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_timeout = 300.0  # 5분 기본 타임아웃
        self.max_retries = 3  # 최대 재시도 횟수
        self.retry_delay = 1.0  # 재시도 간격 (초)
        
        logger.info("비동기 작업 관리 서비스 초기화 완료")
    
    async def create_task(
        self, 
        analysis_id: str,
        task_type: str = "peg_comparison",
        timeout: Optional[float] = None
    ) -> str:
        """
        새로운 비동기 작업 생성
        
        Args:
            analysis_id: 분석 ID
            task_type: 작업 타입
            timeout: 작업 타임아웃 (초)
            
        Returns:
            str: 생성된 작업 ID
        """
        try:
            task_id = str(uuid.uuid4())
            timeout = timeout or self.task_timeout
            
            # 작업 정보 구성
            task_info = {
                "task_id": task_id,
                "analysis_id": analysis_id,
                "task_type": task_type,
                "status": TaskStatus.PENDING.value,
                "progress": 0,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "timeout": timeout,
                "estimated_completion": datetime.utcnow() + timedelta(seconds=timeout),
                "error_message": None,
                "result_data": None
            }
            
            # 활성 작업 목록에 추가
            self.active_tasks[task_id] = task_info
            
            # 캐시에 작업 상태 저장
            cache_service = await get_peg_comparison_cache_service()
            await cache_service.cache_async_task_status(task_id, task_info)
            
            logger.info(f"비동기 작업 생성 완료: {task_id}", extra={
                "task_id": task_id,
                "analysis_id": analysis_id,
                "task_type": task_type,
                "timeout": timeout
            })
            
            return task_id
            
        except Exception as e:
            logger.error(f"비동기 작업 생성 실패: {analysis_id}", extra={
                "analysis_id": analysis_id,
                "task_type": task_type,
                "error_type": type(e).__name__,
                "error_message": str(e)
            }, exc_info=True)
            
            raise AsyncTaskError(
                task_id="unknown",
                message=f"작업 생성 실패: {str(e)}",
                details={"analysis_id": analysis_id, "task_type": task_type}
            )
    
    async def update_task_progress(
        self, 
        task_id: str, 
        progress: int,
        status: Optional[TaskStatus] = None,
        message: Optional[str] = None
    ) -> bool:
        """
        작업 진행률 업데이트
        
        Args:
            task_id: 작업 ID
            progress: 진행률 (0-100)
            status: 작업 상태 (선택사항)
            message: 상태 메시지 (선택사항)
            
        Returns:
            bool: 업데이트 성공 여부
        """
        try:
            # 진행률 범위 검증
            progress = max(0, min(100, progress))
            
            # 작업 정보 조회
            task_info = await self.get_task_status(task_id)
            if not task_info:
                raise AsyncTaskNotFoundError(task_id)
            
            # 상태 업데이트
            if status:
                task_info["status"] = status.value
            
            task_info["progress"] = progress
            task_info["updated_at"] = datetime.utcnow()
            
            if message:
                task_info["message"] = message
            
            # 예상 완료 시간 재계산
            if progress > 0 and task_info["status"] == TaskStatus.PROGRESS.value:
                elapsed = (datetime.utcnow() - task_info["created_at"]).total_seconds()
                if elapsed > 0:
                    estimated_total = elapsed * (100 / progress)
                    remaining = estimated_total - elapsed
                    task_info["estimated_completion"] = datetime.utcnow() + timedelta(seconds=remaining)
            
            # 활성 작업 목록 업데이트
            self.active_tasks[task_id] = task_info
            
            # 캐시 업데이트
            cache_service = await get_peg_comparison_cache_service()
            await cache_service.cache_async_task_status(task_id, task_info)
            
            logger.debug(f"작업 진행률 업데이트: {task_id}", extra={
                "task_id": task_id,
                "progress": progress,
                "status": task_info["status"],
                "message": message
            })
            
            return True
            
        except AsyncTaskNotFoundError:
            raise
        
        except Exception as e:
            logger.error(f"작업 진행률 업데이트 실패: {task_id}", extra={
                "task_id": task_id,
                "progress": progress,
                "status": status.value if status else None,
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            
            raise AsyncTaskError(
                task_id=task_id,
                message=f"진행률 업데이트 실패: {str(e)}",
                task_status=status.value if status else None
            )
    
    async def complete_task(
        self, 
        task_id: str, 
        result_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        작업 완료 처리
        
        Args:
            task_id: 작업 ID
            result_data: 결과 데이터
            
        Returns:
            bool: 완료 처리 성공 여부
        """
        try:
            # 작업 정보 조회
            task_info = await self.get_task_status(task_id)
            if not task_info:
                raise AsyncTaskNotFoundError(task_id)
            
            # 완료 상태로 업데이트
            task_info["status"] = TaskStatus.COMPLETED.value
            task_info["progress"] = 100
            task_info["updated_at"] = datetime.utcnow()
            task_info["result_data"] = result_data
            task_info["estimated_completion"] = datetime.utcnow()
            
            # 활성 작업 목록 업데이트
            self.active_tasks[task_id] = task_info
            
            # 캐시 업데이트
            cache_service = await get_peg_comparison_cache_service()
            await cache_service.cache_async_task_status(task_id, task_info)
            
            logger.info(f"작업 완료 처리: {task_id}", extra={
                "task_id": task_id,
                "analysis_id": task_info.get("analysis_id"),
                "processing_time": (task_info["updated_at"] - task_info["created_at"]).total_seconds()
            })
            
            return True
            
        except AsyncTaskNotFoundError:
            raise
        
        except Exception as e:
            logger.error(f"작업 완료 처리 실패: {task_id}", extra={
                "task_id": task_id,
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            
            raise AsyncTaskError(
                task_id=task_id,
                message=f"완료 처리 실패: {str(e)}",
                task_status=TaskStatus.FAILED.value
            )
    
    async def fail_task(
        self, 
        task_id: str, 
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        작업 실패 처리
        
        Args:
            task_id: 작업 ID
            error_message: 에러 메시지
            error_details: 에러 상세 정보
            
        Returns:
            bool: 실패 처리 성공 여부
        """
        try:
            # 작업 정보 조회
            task_info = await self.get_task_status(task_id)
            if not task_info:
                raise AsyncTaskNotFoundError(task_id)
            
            # 실패 상태로 업데이트
            task_info["status"] = TaskStatus.FAILED.value
            task_info["updated_at"] = datetime.utcnow()
            task_info["error_message"] = error_message
            task_info["error_details"] = error_details
            
            # 활성 작업 목록 업데이트
            self.active_tasks[task_id] = task_info
            
            # 캐시 업데이트
            cache_service = await get_peg_comparison_cache_service()
            await cache_service.cache_async_task_status(task_id, task_info)
            
            logger.warning(f"작업 실패 처리: {task_id}", extra={
                "task_id": task_id,
                "analysis_id": task_info.get("analysis_id"),
                "error_message": error_message,
                "error_details": error_details
            })
            
            return True
            
        except AsyncTaskNotFoundError:
            raise
        
        except Exception as e:
            logger.error(f"작업 실패 처리 중 오류: {task_id}", extra={
                "task_id": task_id,
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            
            raise AsyncTaskError(
                task_id=task_id,
                message=f"실패 처리 중 오류: {str(e)}",
                task_status=TaskStatus.FAILED.value
            )
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        작업 상태 조회
        
        Args:
            task_id: 작업 ID
            
        Returns:
            Optional[Dict[str, Any]]: 작업 상태 정보 또는 None
        """
        try:
            # 먼저 활성 작업 목록에서 조회
            if task_id in self.active_tasks:
                return self.active_tasks[task_id]
            
            # 캐시에서 조회
            cache_service = await get_peg_comparison_cache_service()
            cached_status = await cache_service.get_cached_async_task_status(task_id)
            
            if cached_status:
                # 활성 작업 목록에 복원
                self.active_tasks[task_id] = cached_status
                return cached_status
            
            logger.debug(f"작업 상태를 찾을 수 없음: {task_id}")
            return None
            
        except Exception as e:
            logger.error(f"작업 상태 조회 실패: {task_id}", extra={
                "task_id": task_id,
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            
            return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        작업 취소
        
        Args:
            task_id: 작업 ID
            
        Returns:
            bool: 취소 성공 여부
        """
        try:
            # 작업 정보 조회
            task_info = await self.get_task_status(task_id)
            if not task_info:
                raise AsyncTaskNotFoundError(task_id)
            
            # 이미 완료되거나 실패한 작업은 취소할 수 없음
            if task_info["status"] in [TaskStatus.COMPLETED.value, TaskStatus.FAILED.value]:
                logger.warning(f"완료된 작업은 취소할 수 없음: {task_id}", extra={
                    "task_id": task_id,
                    "current_status": task_info["status"]
                })
                return False
            
            # 취소 상태로 업데이트
            task_info["status"] = TaskStatus.CANCELLED.value
            task_info["updated_at"] = datetime.utcnow()
            
            # 활성 작업 목록 업데이트
            self.active_tasks[task_id] = task_info
            
            # 캐시 업데이트
            cache_service = await get_peg_comparison_cache_service()
            await cache_service.cache_async_task_status(task_id, task_info)
            
            logger.info(f"작업 취소 완료: {task_id}", extra={
                "task_id": task_id,
                "analysis_id": task_info.get("analysis_id")
            })
            
            return True
            
        except AsyncTaskNotFoundError:
            raise
        
        except Exception as e:
            logger.error(f"작업 취소 실패: {task_id}", extra={
                "task_id": task_id,
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            
            raise AsyncTaskError(
                task_id=task_id,
                message=f"취소 실패: {str(e)}",
                task_status=TaskStatus.FAILED.value
            )
    
    async def cleanup_expired_tasks(self) -> Dict[str, int]:
        """
        만료된 작업 정리
        
        Returns:
            Dict[str, int]: 정리 결과 통계
        """
        try:
            current_time = datetime.utcnow()
            expired_tasks = []
            
            # 만료된 작업 찾기
            for task_id, task_info in self.active_tasks.items():
                if task_info["status"] in [TaskStatus.PENDING.value, TaskStatus.PROGRESS.value]:
                    if current_time > task_info["estimated_completion"]:
                        expired_tasks.append(task_id)
            
            # 만료된 작업 실패 처리
            cleanup_stats = {
                "total_expired": len(expired_tasks),
                "failed_tasks": 0,
                "cancelled_tasks": 0
            }
            
            for task_id in expired_tasks:
                try:
                    await self.fail_task(
                        task_id,
                        "작업 타임아웃으로 인한 실패",
                        {"timeout": True, "expired_at": current_time.isoformat()}
                    )
                    cleanup_stats["failed_tasks"] += 1
                except Exception as e:
                    logger.error(f"만료된 작업 실패 처리 중 오류: {task_id}", extra={
                        "task_id": task_id,
                        "error": str(e)
                    })
            
            # 오래된 완료/실패 작업 정리 (24시간 이상)
            old_tasks = []
            cutoff_time = current_time - timedelta(hours=24)
            
            for task_id, task_info in self.active_tasks.items():
                if (task_info["status"] in [TaskStatus.COMPLETED.value, TaskStatus.FAILED.value, TaskStatus.CANCELLED.value] and
                    task_info["updated_at"] < cutoff_time):
                    old_tasks.append(task_id)
            
            for task_id in old_tasks:
                del self.active_tasks[task_id]
                cleanup_stats["cancelled_tasks"] += 1
            
            logger.info("만료된 작업 정리 완료", extra={
                "expired_tasks": cleanup_stats["total_expired"],
                "failed_tasks": cleanup_stats["failed_tasks"],
                "cancelled_tasks": cleanup_stats["cancelled_tasks"]
            })
            
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"만료된 작업 정리 실패: {e}", extra={
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            
            return {
                "error": str(e),
                "total_expired": 0,
                "failed_tasks": 0,
                "cancelled_tasks": 0
            }
    
    async def get_active_tasks(self) -> List[Dict[str, Any]]:
        """
        활성 작업 목록 조회
        
        Returns:
            List[Dict[str, Any]]: 활성 작업 목록
        """
        try:
            active_tasks = []
            
            for task_id, task_info in self.active_tasks.items():
                if task_info["status"] in [TaskStatus.PENDING.value, TaskStatus.PROGRESS.value]:
                    active_tasks.append({
                        "task_id": task_id,
                        "analysis_id": task_info.get("analysis_id"),
                        "task_type": task_info.get("task_type"),
                        "status": task_info["status"],
                        "progress": task_info["progress"],
                        "created_at": task_info["created_at"].isoformat(),
                        "estimated_completion": task_info.get("estimated_completion", {}).isoformat() if isinstance(task_info.get("estimated_completion"), datetime) else None
                    })
            
            logger.debug(f"활성 작업 목록 조회: {len(active_tasks)}개")
            
            return active_tasks
            
        except Exception as e:
            logger.error(f"활성 작업 목록 조회 실패: {e}", extra={
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            
            return []
    
    async def list_tasks(
        self, 
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        작업 목록 조회 (필터링 지원)
        
        Args:
            status: 상태 필터
            limit: 최대 결과 수
            offset: 오프셋
            
        Returns:
            Dict[str, Any]: 작업 목록과 메타데이터
        """
        try:
            all_tasks = list(self.active_tasks.values())
            
            # 상태 필터 적용
            if status:
                all_tasks = [task for task in all_tasks if task["status"] == status]
            
            # 정렬 (최신순)
            all_tasks.sort(key=lambda x: x["created_at"], reverse=True)
            
            # 페이징 적용
            total_count = len(all_tasks)
            paginated_tasks = all_tasks[offset:offset + limit]
            
            # 응답 데이터 구성
            result = {
                "tasks": [
                    {
                        "task_id": task["task_id"],
                        "analysis_id": task.get("analysis_id"),
                        "task_type": task.get("task_type"),
                        "status": task["status"],
                        "progress": task["progress"],
                        "created_at": task["created_at"].isoformat(),
                        "updated_at": task["updated_at"].isoformat(),
                        "estimated_completion": task.get("estimated_completion", {}).isoformat() if isinstance(task.get("estimated_completion"), datetime) else None
                    }
                    for task in paginated_tasks
                ],
                "total_count": total_count,
                "pagination": {
                    "total": total_count,
                    "limit": limit,
                    "offset": offset,
                    "has_more": offset + limit < total_count
                }
            }
            
            logger.debug(f"작업 목록 조회: {len(paginated_tasks)}개 (총 {total_count}개)")
            
            return result
            
        except Exception as e:
            logger.error(f"작업 목록 조회 실패: {e}", extra={
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            
            return {
                "tasks": [],
                "pagination": {
                    "total": 0,
                    "limit": limit,
                    "offset": offset,
                    "has_more": False
                },
                "error": str(e)
            }
    
    async def get_task_statistics(self) -> Dict[str, Any]:
        """
        작업 통계 조회
        
        Returns:
            Dict[str, Any]: 작업 통계 정보
        """
        try:
            stats = {
                "total_tasks": len(self.active_tasks),
                "by_status": {
                    "pending": 0,
                    "progress": 0,
                    "completed": 0,
                    "failed": 0,
                    "cancelled": 0
                },
                "status_distribution": {
                    "pending": 0,
                    "progress": 0,
                    "completed": 0,
                    "failed": 0,
                    "cancelled": 0
                },
                "by_type": {},
                "average_processing_time": 0.0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            total_processing_time = 0.0
            completed_count = 0
            
            for task_info in self.active_tasks.values():
                # 상태별 통계
                status = task_info["status"].lower()
                if status in stats["by_status"]:
                    stats["by_status"][status] += 1
                    stats["status_distribution"][status] += 1
                
                # 타입별 통계
                task_type = task_info.get("task_type", "unknown")
                stats["by_type"][task_type] = stats["by_type"].get(task_type, 0) + 1
                
                # 평균 처리 시간 계산
                if task_info["status"] == TaskStatus.COMPLETED.value:
                    processing_time = (task_info["updated_at"] - task_info["created_at"]).total_seconds()
                    total_processing_time += processing_time
                    completed_count += 1
            
            if completed_count > 0:
                stats["average_processing_time"] = total_processing_time / completed_count
            
            logger.debug("작업 통계 조회 완료", extra=stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"작업 통계 조회 실패: {e}", extra={
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def create_async_task(
        self, 
        task_data: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> str:
        """
        새로운 비동기 작업 생성 (create_task의 별칭)
        
        Args:
            task_data: 작업 데이터
            timeout: 작업 타임아웃 (초)
            
        Returns:
            str: 생성된 작업 ID
        """
        analysis_id = task_data.get("analysis_id", "unknown")
        task_type = task_data.get("task_type", "peg_comparison")
        return await self.create_task(analysis_id, task_type, timeout)

    async def update_task_status(
        self, 
        task_id: str, 
        status: str, 
        progress: Optional[int] = None,
        error_message: Optional[str] = None,
        result_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        작업 상태 업데이트
        
        Args:
            task_id: 작업 ID
            status: 새로운 상태
            progress: 진행률 (0-100)
            error_message: 오류 메시지
            result_data: 결과 데이터
            
        Returns:
            bool: 업데이트 성공 여부
        """
        try:
            if task_id not in self.active_tasks:
                raise AsyncTaskNotFoundError(f"작업을 찾을 수 없습니다: {task_id}")
            
            task_info = self.active_tasks[task_id]
            task_info["status"] = status
            task_info["updated_at"] = datetime.utcnow()
            
            if progress is not None:
                task_info["progress"] = progress
            
            if error_message is not None:
                task_info["error_message"] = error_message
            
            if result_data is not None:
                task_info["result_data"] = result_data
            
            # 캐시 업데이트
            cache_service = await get_peg_comparison_cache_service()
            await cache_service.cache_async_task_status(task_id, task_info)
            
            logger.info(f"작업 상태 업데이트 완료: {task_id} -> {status}", extra={
                "task_id": task_id,
                "status": status,
                "progress": progress
            })
            
            return True
            
        except Exception as e:
            logger.error(f"작업 상태 업데이트 실패: {task_id}", extra={
                "task_id": task_id,
                "status": status,
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            raise AsyncTaskError(f"작업 상태 업데이트 중 오류 발생: {str(e)}")

    async def get_task_result(self, task_id: str) -> Dict[str, Any]:
        """
        작업 결과 조회
        
        Args:
            task_id: 작업 ID
            
        Returns:
            Dict[str, Any]: 작업 결과
        """
        try:
            if task_id not in self.active_tasks:
                raise AsyncTaskNotFoundError(f"작업을 찾을 수 없습니다: {task_id}")
            
            task_info = self.active_tasks[task_id]
            
            if task_info["status"] != "COMPLETED":
                raise AsyncTaskError(f"작업이 아직 완료되지 않았습니다: {task_info['status']}")
            
            result = {
                "task_id": task_id,
                "status": task_info["status"],
                "result_data": task_info.get("result_data"),
                "completed_at": task_info.get("updated_at"),
                "processing_time": self._calculate_processing_time(task_info)
            }
            
            logger.info(f"작업 결과 조회 완료: {task_id}")
            return result
            
        except Exception as e:
            logger.error(f"작업 결과 조회 실패: {task_id}", extra={
                "task_id": task_id,
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            raise
    
    async def cleanup_completed_tasks(self, older_than_hours: int = 24, days_old: Optional[int] = None) -> Dict[str, int]:
        """
        완료된 작업 정리
        
        Args:
            older_than_hours: 정리할 작업의 최소 경과 시간 (시간)
            
        Returns:
            Dict[str, int]: 정리 결과 통계
        """
        try:
            # days_old가 제공되면 시간으로 변환
            if days_old is not None:
                older_than_hours = days_old * 24
            
            cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
            cleaned_tasks = []
            
            for task_id, task_info in list(self.active_tasks.items()):
                if (task_info["status"] in [TaskStatus.COMPLETED.value, TaskStatus.FAILED.value, TaskStatus.CANCELLED.value] and
                    task_info["updated_at"] < cutoff_time):
                    del self.active_tasks[task_id]
                    cleaned_tasks.append(task_id)
            
            result = {
                "cleaned_tasks": len(cleaned_tasks),
                "remaining_tasks": len(self.active_tasks)
            }
            
            logger.info(f"완료된 작업 정리 완료: {len(cleaned_tasks)}개", extra=result)
            
            return result
            
        except Exception as e:
            logger.error(f"완료된 작업 정리 실패: {e}", extra={
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            
            return {
                "error": str(e),
                "cleaned_tasks": 0,
                "remaining_tasks": len(self.active_tasks)
            }


# 전역 비동기 작업 서비스 인스턴스
_async_task_service: Optional[AsyncTaskService] = None


async def get_async_task_service() -> AsyncTaskService:
    """
    전역 비동기 작업 서비스 인스턴스 반환
    
    Returns:
        AsyncTaskService: 비동기 작업 서비스 인스턴스
    """
    global _async_task_service
    
    if _async_task_service is None:
        _async_task_service = AsyncTaskService()
        logger.info("전역 비동기 작업 서비스 생성")
    
    return _async_task_service
