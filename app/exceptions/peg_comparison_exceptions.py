"""
PEG 비교분석 전용 예외 클래스들

이 모듈은 PEG 비교분석 기능에서 발생할 수 있는 예외들을 정의합니다.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime

from .choi_exceptions import ChoiAlgorithmError

logger = logging.getLogger("app.exceptions.peg_comparison")


class PEGComparisonException(ChoiAlgorithmError):
    """PEG 비교분석 기본 예외 클래스"""
    
    def __init__(
        self, 
        message: str, 
        error_code: str = "PEG_COMPARISON_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, error_code, context=details)
        self.status_code = status_code
        self.details = details or {}
        self.timestamp = datetime.utcnow()
        
        logger.error(f"PEG 비교분석 예외 발생: {error_code} - {message}", extra={
            "error_code": error_code,
            "status_code": status_code,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        })


class MCPConnectionError(PEGComparisonException):
    """MCP 서버 연결 에러"""
    
    def __init__(
        self, 
        message: str = "MCP 서버 연결 실패",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="MCP_CONNECTION_FAILED",
            status_code=503,
            details=details
        )


class MCPTimeoutError(PEGComparisonException):
    """MCP 서버 타임아웃 에러"""
    
    def __init__(
        self, 
        message: str = "MCP 서버 응답 타임아웃",
        timeout_seconds: float = 30.0,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details.update({
            "timeout_seconds": timeout_seconds,
            "error_type": "timeout"
        })
        
        super().__init__(
            message=message,
            error_code="MCP_TIMEOUT_ERROR",
            status_code=504,
            details=details
        )


class DataValidationError(PEGComparisonException):
    """데이터 검증 에러"""
    
    def __init__(
        self, 
        message: str = "데이터 검증 실패",
        validation_errors: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if validation_errors:
            details["validation_errors"] = validation_errors
        
        super().__init__(
            message=message,
            error_code="DATA_VALIDATION_ERROR",
            status_code=400,
            details=details
        )


class AnalysisDataNotFoundError(PEGComparisonException):
    """분석 데이터를 찾을 수 없는 에러"""
    
    def __init__(
        self, 
        analysis_id: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if not message:
            message = f"분석 데이터를 찾을 수 없습니다: {analysis_id}"
        
        details = details or {}
        details.update({
            "analysis_id": analysis_id,
            "error_type": "data_not_found"
        })
        
        super().__init__(
            message=message,
            error_code="ANALYSIS_DATA_NOT_FOUND",
            status_code=404,
            details=details
        )


class CacheError(PEGComparisonException):
    """캐시 관련 에러"""
    
    def __init__(
        self, 
        message: str = "캐시 작업 실패",
        cache_operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if cache_operation:
            details["cache_operation"] = cache_operation
        
        super().__init__(
            message=message,
            error_code="CACHE_ERROR",
            status_code=500,
            details=details
        )


class AsyncTaskError(PEGComparisonException):
    """비동기 작업 에러"""
    
    def __init__(
        self, 
        task_id: str,
        message: str = "비동기 작업 실패",
        task_status: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details.update({
            "task_id": task_id,
            "task_status": task_status,
            "error_type": "async_task_error"
        })
        
        super().__init__(
            message=message,
            error_code="ASYNC_TASK_ERROR",
            status_code=500,
            details=details
        )


class AsyncTaskNotFoundError(PEGComparisonException):
    """비동기 작업을 찾을 수 없는 에러"""
    
    def __init__(
        self, 
        task_id: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if not message:
            message = f"비동기 작업을 찾을 수 없습니다: {task_id}"
        
        details = details or {}
        details.update({
            "task_id": task_id,
            "error_type": "task_not_found"
        })
        
        super().__init__(
            message=message,
            error_code="ASYNC_TASK_NOT_FOUND",
            status_code=404,
            details=details
        )


class RateLimitExceededError(PEGComparisonException):
    """Rate Limit 초과 에러"""
    
    def __init__(
        self, 
        message: str = "요청 한도 초과",
        limit: Optional[int] = None,
        window: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details.update({
            "limit": limit,
            "window": window,
            "error_type": "rate_limit_exceeded"
        })
        
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            details=details
        )


class PermissionDeniedError(PEGComparisonException):
    """권한 거부 에러"""
    
    def __init__(
        self, 
        message: str = "권한이 없습니다",
        user_id: Optional[str] = None,
        analysis_id: Optional[str] = None,
        required_permission: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details.update({
            "user_id": user_id,
            "analysis_id": analysis_id,
            "required_permission": required_permission,
            "error_type": "permission_denied"
        })
        
        super().__init__(
            message=message,
            error_code="PERMISSION_DENIED",
            status_code=403,
            details=details
        )


class AlgorithmVersionError(PEGComparisonException):
    """알고리즘 버전 에러"""
    
    def __init__(
        self, 
        message: str = "지원하지 않는 알고리즘 버전",
        requested_version: Optional[str] = None,
        supported_versions: Optional[list] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details.update({
            "requested_version": requested_version,
            "supported_versions": supported_versions,
            "error_type": "algorithm_version_error"
        })
        
        super().__init__(
            message=message,
            error_code="ALGORITHM_VERSION_ERROR",
            status_code=400,
            details=details
        )


class ProcessingTimeoutError(PEGComparisonException):
    """처리 타임아웃 에러"""
    
    def __init__(
        self, 
        message: str = "처리 시간 초과",
        timeout_seconds: float = 300.0,
        analysis_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details.update({
            "timeout_seconds": timeout_seconds,
            "analysis_id": analysis_id,
            "error_type": "processing_timeout"
        })
        
        super().__init__(
            message=message,
            error_code="PROCESSING_TIMEOUT",
            status_code=504,
            details=details
        )
