"""
PEG 비교분석 예외 핸들러

이 모듈은 PEG 비교분석 관련 예외들을 처리하는 핸들러 함수들을 정의합니다.
"""

import logging
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime

from .peg_comparison_exceptions import (
    PEGComparisonException,
    MCPConnectionError,
    MCPTimeoutError,
    DataValidationError,
    AnalysisDataNotFoundError,
    CacheError,
    AsyncTaskError,
    AsyncTaskNotFoundError,
    RateLimitExceededError,
    PermissionDeniedError,
    AlgorithmVersionError,
    ProcessingTimeoutError
)

logger = logging.getLogger("app.exceptions.peg_comparison_handlers")


def peg_comparison_exception_handler(request: Request, exc: PEGComparisonException) -> JSONResponse:
    """PEG 비교분석 기본 예외 핸들러"""
    logger.error(f"PEG 비교분석 예외 발생: {exc.error_code}", extra={
        "error_code": exc.error_code,
        "error_message": exc.message,
        "status_code": exc.status_code,
        "details": exc.details,
        "path": str(request.url),
        "method": request.method
    })
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "timestamp": exc.timestamp.isoformat(),
                "details": exc.details
            }
        }
    )


def mcp_connection_error_handler(request: Request, exc: MCPConnectionError) -> JSONResponse:
    """MCP 서버 연결 에러 핸들러"""
    logger.error(f"MCP 서버 연결 실패: {exc.message}", extra={
        "error_code": exc.error_code,
        "error_message": exc.message,
        "details": exc.details,
        "path": str(request.url),
        "method": request.method
    })
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "timestamp": exc.timestamp.isoformat(),
                "details": exc.details
            }
        }
    )


def mcp_timeout_error_handler(request: Request, exc: MCPTimeoutError) -> JSONResponse:
    """MCP 서버 타임아웃 에러 핸들러"""
    logger.warning(f"MCP 서버 응답 타임아웃: {exc.message}", extra={
        "error_code": exc.error_code,
        "error_message": exc.message,
        "timeout_seconds": exc.details.get("timeout_seconds"),
        "path": str(request.url),
        "method": request.method
    })
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "timestamp": exc.timestamp.isoformat(),
                "details": exc.details
            }
        }
    )


def data_validation_error_handler(request: Request, exc: DataValidationError) -> JSONResponse:
    """데이터 검증 에러 핸들러"""
    logger.warning(f"데이터 검증 실패: {exc.message}", extra={
        "error_code": exc.error_code,
        "error_message": exc.message,
        "validation_errors": exc.details.get("validation_errors"),
        "path": str(request.url),
        "method": request.method
    })
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "timestamp": exc.timestamp.isoformat(),
                "details": exc.details
            }
        }
    )


def analysis_data_not_found_error_handler(request: Request, exc: AnalysisDataNotFoundError) -> JSONResponse:
    """분석 데이터를 찾을 수 없는 에러 핸들러"""
    logger.warning(f"분석 데이터를 찾을 수 없음: {exc.message}", extra={
        "error_code": exc.error_code,
        "error_message": exc.message,
        "analysis_id": exc.details.get("analysis_id"),
        "path": str(request.url),
        "method": request.method
    })
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "timestamp": exc.timestamp.isoformat(),
                "details": exc.details
            }
        }
    )


def cache_error_handler(request: Request, exc: CacheError) -> JSONResponse:
    """캐시 에러 핸들러"""
    logger.error(f"캐시 작업 실패: {exc.message}", extra={
        "error_code": exc.error_code,
        "error_message": exc.message,
        "cache_operation": exc.details.get("cache_operation"),
        "path": str(request.url),
        "method": request.method
    })
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "timestamp": exc.timestamp.isoformat(),
                "details": exc.details
            }
        }
    )


def async_task_error_handler(request: Request, exc: AsyncTaskError) -> JSONResponse:
    """비동기 작업 에러 핸들러"""
    logger.error(f"비동기 작업 실패: {exc.message}", extra={
        "error_code": exc.error_code,
        "error_message": exc.message,
        "task_id": exc.details.get("task_id"),
        "task_status": exc.details.get("task_status"),
        "path": str(request.url),
        "method": request.method
    })
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "timestamp": exc.timestamp.isoformat(),
                "details": exc.details
            }
        }
    )


def async_task_not_found_error_handler(request: Request, exc: AsyncTaskNotFoundError) -> JSONResponse:
    """비동기 작업을 찾을 수 없는 에러 핸들러"""
    logger.warning(f"비동기 작업을 찾을 수 없음: {exc.message}", extra={
        "error_code": exc.error_code,
        "error_message": exc.message,
        "task_id": exc.details.get("task_id"),
        "path": str(request.url),
        "method": request.method
    })
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "timestamp": exc.timestamp.isoformat(),
                "details": exc.details
            }
        }
    )


def rate_limit_exceeded_error_handler(request: Request, exc: RateLimitExceededError) -> JSONResponse:
    """Rate Limit 초과 에러 핸들러"""
    logger.warning(f"요청 한도 초과: {exc.message}", extra={
        "error_code": exc.error_code,
        "error_message": exc.message,
        "limit": exc.details.get("limit"),
        "window": exc.details.get("window"),
        "path": str(request.url),
        "method": request.method
    })
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "timestamp": exc.timestamp.isoformat(),
                "details": exc.details
            }
        }
    )


def permission_denied_error_handler(request: Request, exc: PermissionDeniedError) -> JSONResponse:
    """권한 거부 에러 핸들러"""
    logger.warning(f"권한 거부: {exc.message}", extra={
        "error_code": exc.error_code,
        "error_message": exc.message,
        "user_id": exc.details.get("user_id"),
        "analysis_id": exc.details.get("analysis_id"),
        "required_permission": exc.details.get("required_permission"),
        "path": str(request.url),
        "method": request.method
    })
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "timestamp": exc.timestamp.isoformat(),
                "details": exc.details
            }
        }
    )


def algorithm_version_error_handler(request: Request, exc: AlgorithmVersionError) -> JSONResponse:
    """알고리즘 버전 에러 핸들러"""
    logger.warning(f"지원하지 않는 알고리즘 버전: {exc.message}", extra={
        "error_code": exc.error_code,
        "error_message": exc.message,
        "requested_version": exc.details.get("requested_version"),
        "supported_versions": exc.details.get("supported_versions"),
        "path": str(request.url),
        "method": request.method
    })
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "timestamp": exc.timestamp.isoformat(),
                "details": exc.details
            }
        }
    )


def processing_timeout_error_handler(request: Request, exc: ProcessingTimeoutError) -> JSONResponse:
    """처리 타임아웃 에러 핸들러"""
    logger.warning(f"처리 시간 초과: {exc.message}", extra={
        "error_code": exc.error_code,
        "error_message": exc.message,
        "timeout_seconds": exc.details.get("timeout_seconds"),
        "analysis_id": exc.details.get("analysis_id"),
        "path": str(request.url),
        "method": request.method
    })
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "timestamp": exc.timestamp.isoformat(),
                "details": exc.details
            }
        }
    )


















