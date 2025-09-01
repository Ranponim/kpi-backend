"""
커스텀 예외 클래스 정의

이 모듈은 애플리케이션에서 사용되는 커스텀 예외들과
중앙 집중식 예외 처리를 위한 클래스들을 정의합니다.
"""

import logging
from typing import Optional, Dict, Any
from fastapi import Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class BaseAPIException(Exception):
    """
    API 예외의 기본 클래스
    
    모든 커스텀 예외는 이 클래스를 상속받아야 합니다.
    """
    
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class AnalysisResultNotFoundException(BaseAPIException):
    """분석 결과를 찾을 수 없을 때 발생하는 예외"""
    
    def __init__(self, result_id: str):
        self.result_id = result_id
        message = f"Analysis result with id '{result_id}' not found"
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            details={"result_id": result_id}
        )


class InvalidAnalysisDataException(BaseAPIException):
    """분석 데이터가 유효하지 않을 때 발생하는 예외"""
    
    def __init__(self, message: str, validation_errors: Optional[list] = None):
        super().__init__(
            message=f"Invalid analysis data: {message}",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details={"validation_errors": validation_errors or []}
        )


class DatabaseConnectionException(BaseAPIException):
    """데이터베이스 연결 오류 시 발생하는 예외"""
    
    def __init__(self, message: str = "Database connection failed"):
        super().__init__(
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details={"service": "database"}
        )


class DuplicateAnalysisResultException(BaseAPIException):
    """중복된 분석 결과가 있을 때 발생하는 예외"""
    
    def __init__(self, ne_id: str, cell_id: str, analysis_date: str):
        message = f"Analysis result already exists for NE '{ne_id}', Cell '{cell_id}' on date '{analysis_date}'"
        super().__init__(
            message=message,
            status_code=status.HTTP_409_CONFLICT,
            details={
                "ne_id": ne_id,
                "cell_id": cell_id,
                "analysis_date": analysis_date
            }
        )


class InvalidFilterException(BaseAPIException):
    """필터 파라미터가 유효하지 않을 때 발생하는 예외"""
    
    def __init__(self, message: str, invalid_filters: Optional[Dict] = None):
        super().__init__(
            message=f"Invalid filter parameters: {message}",
            status_code=status.HTTP_400_BAD_REQUEST,
            details={"invalid_filters": invalid_filters or {}}
        )


class UserPreferenceNotFoundException(BaseAPIException):
    """사용자 설정을 찾을 수 없을 때 발생하는 예외"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        message = f"User preference for user '{user_id}' not found"
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            details={"user_id": user_id}
        )


class InvalidPreferenceDataException(BaseAPIException):
    """설정 데이터가 유효하지 않을 때 발생하는 예외"""
    
    def __init__(self, message: str, validation_errors: Optional[list] = None):
        super().__init__(
            message=f"Invalid preference data: {message}",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details={"validation_errors": validation_errors or []}
        )


class PreferenceImportException(BaseAPIException):
    """설정 가져오기 과정에서 오류가 발생할 때의 예외"""
    
    def __init__(self, message: str, import_errors: Optional[list] = None):
        super().__init__(
            message=f"Preference import failed: {message}",
            status_code=status.HTTP_400_BAD_REQUEST,
            details={"import_errors": import_errors or []}
        )


# ===== Host 필터링 관련 새로운 예외 클래스들 =====

class TargetValidationException(BaseAPIException):
    """
    타겟 검증 실패 시 발생하는 예외
    
    NE, Cell, Host 필터의 형식이나 존재 여부 검증에서 문제가 발생했을 때 사용됩니다.
    """
    
    def __init__(
        self, 
        message: str = "Target validation failed", 
        invalid_targets: Optional[Dict[str, list]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.invalid_targets = invalid_targets or {}
        combined_details = {
            "invalid_targets": self.invalid_targets,
            **(details or {})
        }
        super().__init__(
            message=message, 
            status_code=status.HTTP_400_BAD_REQUEST, 
            details=combined_details
        )


class HostValidationException(TargetValidationException):
    """
    Host 검증 실패 시 발생하는 예외
    
    Host ID의 형식 검증, IP 주소 유효성, 호스트명 검증, DNS 조회 등에서 
    문제가 발생했을 때 사용됩니다.
    """
    
    def __init__(
        self, 
        message: str = "Host validation failed",
        invalid_hosts: Optional[list] = None,
        validation_errors: Optional[Dict[str, str]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.invalid_hosts = invalid_hosts or []
        self.validation_errors = validation_errors or {}
        
        combined_details = {
            "invalid_hosts": self.invalid_hosts,
            "validation_errors": self.validation_errors,
            **(details or {})
        }
        
        super().__init__(
            message=message, 
            invalid_targets={"hosts": self.invalid_hosts}, 
            details=combined_details
        )


class RelationshipValidationException(TargetValidationException):
    """
    타겟 간 관계 검증 실패 시 발생하는 예외
    
    NE-Cell-Host 간의 연관성 검증에서 문제가 발생했을 때 사용됩니다.
    """
    
    def __init__(
        self, 
        message: str = "Relationship validation failed",
        missing_relationships: Optional[Dict[str, list]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.missing_relationships = missing_relationships or {}
        combined_details = {
            "missing_relationships": self.missing_relationships,
            **(details or {})
        }
        super().__init__(message=message, details=combined_details)


class FilterCombinationException(BaseAPIException):
    """
    필터 조합 오류 시 발생하는 예외
    
    호환되지 않는 필터 조합이나 논리적으로 불가능한 
    필터 조합이 발견되었을 때 사용됩니다.
    """
    
    def __init__(
        self, 
        message: str = "Filter combination error",
        conflicting_filters: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.conflicting_filters = conflicting_filters or {}
        combined_details = {
            "conflicting_filters": self.conflicting_filters,
            **(details or {})
        }
        super().__init__(
            message=message, 
            status_code=status.HTTP_400_BAD_REQUEST, 
            details=combined_details
        )


class MongoDBIndexException(BaseAPIException):
    """
    MongoDB 인덱스 관련 오류 시 발생하는 예외
    
    인덱스 생성, 삭제, 최적화 과정에서 문제가 발생했을 때 사용됩니다.
    """
    
    def __init__(
        self, 
        message: str = "MongoDB index operation failed",
        index_name: Optional[str] = None,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.index_name = index_name
        self.operation = operation
        combined_details = {
            "index_name": self.index_name,
            "operation": self.operation,
            **(details or {})
        }
        super().__init__(
            message=message, 
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            details=combined_details
        )


class LLMAnalysisException(BaseAPIException):
    """
    LLM 분석 관련 오류 시 발생하는 예외
    
    Host 정보를 포함한 LLM 분석 과정에서 문제가 발생했을 때 사용됩니다.
    """
    
    def __init__(
        self, 
        message: str = "LLM analysis failed",
        analysis_stage: Optional[str] = None,
        target_info: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.analysis_stage = analysis_stage
        self.target_info = target_info or {}
        combined_details = {
            "analysis_stage": self.analysis_stage,
            "target_info": self.target_info,
            **(details or {})
        }
        super().__init__(
            message=message, 
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            details=combined_details
        )


# 예외 핸들러 함수들
async def base_api_exception_handler(request: Request, exc: BaseAPIException) -> JSONResponse:
    """
    BaseAPIException에 대한 기본 예외 핸들러
    
    Args:
        request: FastAPI 요청 객체
        exc: 발생한 예외
        
    Returns:
        JSONResponse: 표준화된 에러 응답
    """
    logger.error(
        f"API Exception: {exc.message} | "
        f"Status: {exc.status_code} | "
        f"Details: {exc.details} | "
        f"Path: {request.url.path}"
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "message": exc.message,
                "type": exc.__class__.__name__,
                "status_code": exc.status_code,
                "details": exc.details
            },
            "path": str(request.url.path),
            "method": request.method
        }
    )


async def analysis_result_not_found_handler(request: Request, exc: AnalysisResultNotFoundException) -> JSONResponse:
    """분석 결과 찾을 수 없음 예외 핸들러"""
    logger.warning(f"Analysis result not found: {exc.result_id} | Path: {request.url.path}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "message": exc.message,
                "type": "AnalysisResultNotFound",
                "result_id": exc.result_id
            }
        }
    )


async def invalid_analysis_data_handler(request: Request, exc: InvalidAnalysisDataException) -> JSONResponse:
    """유효하지 않은 분석 데이터 예외 핸들러"""
    logger.warning(f"Invalid analysis data: {exc.message} | Path: {request.url.path}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "message": exc.message,
                "type": "InvalidAnalysisData",
                "validation_errors": exc.details.get("validation_errors", [])
            }
        }
    )


async def database_connection_handler(request: Request, exc: DatabaseConnectionException) -> JSONResponse:
    """데이터베이스 연결 오류 예외 핸들러"""
    logger.error(f"Database connection error: {exc.message} | Path: {request.url.path}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "message": "Service temporarily unavailable. Please try again later.",
                "type": "ServiceUnavailable",
                "service": "database"
            }
        }
    )


async def duplicate_analysis_result_handler(request: Request, exc: DuplicateAnalysisResultException) -> JSONResponse:
    """중복 분석 결과 예외 핸들러"""
    logger.warning(f"Duplicate analysis result: {exc.message} | Path: {request.url.path}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "message": exc.message,
                "type": "DuplicateAnalysisResult",
                "details": exc.details
            }
        }
    )


async def user_preference_not_found_handler(request: Request, exc: UserPreferenceNotFoundException) -> JSONResponse:
    """사용자 설정 찾을 수 없음 예외 핸들러"""
    logger.warning(f"User preference not found: {exc.user_id} | Path: {request.url.path}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "message": exc.message,
                "type": "UserPreferenceNotFound",
                "user_id": exc.user_id
            }
        }
    )


async def invalid_preference_data_handler(request: Request, exc: InvalidPreferenceDataException) -> JSONResponse:
    """유효하지 않은 설정 데이터 예외 핸들러"""
    logger.warning(f"Invalid preference data: {exc.message} | Path: {request.url.path}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "message": exc.message,
                "type": "InvalidPreferenceData",
                "validation_errors": exc.details.get("validation_errors", [])
            }
        }
    )


async def preference_import_handler(request: Request, exc: PreferenceImportException) -> JSONResponse:
    """설정 가져오기 오류 예외 핸들러"""
    logger.warning(f"Preference import error: {exc.message} | Path: {request.url.path}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "message": exc.message,
                "type": "PreferenceImportError",
                "import_errors": exc.details.get("import_errors", [])
            }
        }
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    일반적인 예외에 대한 핸들러
    
    예상하지 못한 예외가 발생했을 때 사용됩니다.
    """
    logger.error(f"Unexpected error: {str(exc)} | Path: {request.url.path}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "message": "An unexpected error occurred. Please try again later.",
                "type": "InternalServerError"
            }
        }
    )


# ===== Host 필터링 관련 새로운 예외 핸들러들 =====

async def target_validation_exception_handler(request: Request, exc: TargetValidationException) -> JSONResponse:
    """타겟 검증 실패 예외 핸들러"""
    logger.warning(f"Target validation error: {exc.message} | Path: {request.url.path}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "message": exc.message,
                "type": "TargetValidationError",
                "invalid_targets": exc.invalid_targets,
                "details": exc.details
            }
        }
    )


async def host_validation_exception_handler(request: Request, exc: HostValidationException) -> JSONResponse:
    """Host 검증 실패 예외 핸들러"""
    logger.warning(f"Host validation error: {exc.message} | Invalid hosts: {exc.invalid_hosts} | Path: {request.url.path}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "message": exc.message,
                "type": "HostValidationError",
                "invalid_hosts": exc.invalid_hosts,
                "validation_errors": exc.validation_errors,
                "details": exc.details
            }
        }
    )


async def relationship_validation_exception_handler(request: Request, exc: RelationshipValidationException) -> JSONResponse:
    """관계 검증 실패 예외 핸들러"""
    logger.warning(f"Relationship validation error: {exc.message} | Path: {request.url.path}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "message": exc.message,
                "type": "RelationshipValidationError",
                "missing_relationships": exc.missing_relationships,
                "details": exc.details
            }
        }
    )


async def filter_combination_exception_handler(request: Request, exc: FilterCombinationException) -> JSONResponse:
    """필터 조합 오류 예외 핸들러"""
    logger.warning(f"Filter combination error: {exc.message} | Path: {request.url.path}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "message": exc.message,
                "type": "FilterCombinationError",
                "conflicting_filters": exc.conflicting_filters,
                "details": exc.details
            }
        }
    )


async def mongodb_index_exception_handler(request: Request, exc: MongoDBIndexException) -> JSONResponse:
    """MongoDB 인덱스 오류 예외 핸들러"""
    logger.error(f"MongoDB index error: {exc.message} | Index: {exc.index_name} | Operation: {exc.operation} | Path: {request.url.path}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "message": "Database index operation failed. Please contact administrator.",
                "type": "DatabaseIndexError",
                "details": {
                    "index_name": exc.index_name,
                    "operation": exc.operation
                }
            }
        }
    )


async def llm_analysis_exception_handler(request: Request, exc: LLMAnalysisException) -> JSONResponse:
    """LLM 분석 오류 예외 핸들러"""
    logger.error(f"LLM analysis error: {exc.message} | Stage: {exc.analysis_stage} | Path: {request.url.path}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "message": exc.message,
                "type": "LLMAnalysisError",
                "analysis_stage": exc.analysis_stage,
                "target_info": exc.target_info,
                "details": exc.details
            }
        }
    )


# 편의 함수들

def raise_host_validation_error(
    invalid_hosts: list, 
    validation_errors: Dict[str, str],
    custom_message: Optional[str] = None
) -> None:
    """
    Host 검증 오류를 발생시키는 편의 함수
    
    Args:
        invalid_hosts: 유효하지 않은 호스트 목록
        validation_errors: 각 호스트별 오류 메시지
        custom_message: 커스텀 오류 메시지
    """
    message = custom_message or f"Host validation failed for {len(invalid_hosts)} hosts"
    raise HostValidationException(
        message=message,
        invalid_hosts=invalid_hosts,
        validation_errors=validation_errors
    )


def raise_relationship_validation_error(
    missing_combinations: list,
    custom_message: Optional[str] = None
) -> None:
    """
    관계 검증 오류를 발생시키는 편의 함수
    
    Args:
        missing_combinations: 누락된 NE-Cell-Host 조합 목록
        custom_message: 커스텀 오류 메시지
    """
    message = custom_message or f"Relationship validation failed for {len(missing_combinations)} combinations"
    raise RelationshipValidationException(
        message=message,
        missing_relationships={"ne_cell_host_combinations": missing_combinations}
    )


def raise_filter_combination_error(
    conflicting_filters: Dict[str, Any],
    custom_message: Optional[str] = None
) -> None:
    """
    필터 조합 오류를 발생시키는 편의 함수
    
    Args:
        conflicting_filters: 충돌하는 필터 정보
        custom_message: 커스텀 오류 메시지
    """
    message = custom_message or "Incompatible filter combination detected"
    raise FilterCombinationException(
        message=message,
        conflicting_filters=conflicting_filters
    )
