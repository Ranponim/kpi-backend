"""
Choi 알고리즘 전용 커스텀 예외 클래스들

이 모듈은 Choi 알고리즘 실행 중 발생할 수 있는 다양한 예외 상황을 
구체적으로 표현하는 커스텀 예외 클래스들을 정의합니다.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import json


class ChoiAlgorithmError(Exception):
    """
    Choi 알고리즘 관련 모든 예외의 기본 클래스
    
    모든 Choi 알고리즘 예외는 이 클래스를 상속받아 구현됩니다.
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "CHOI_ERROR",
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        Choi 알고리즘 예외를 초기화합니다.
        
        Args:
            message: 오류 메시지
            error_code: 오류 코드 (로깅 및 모니터링용)
            context: 오류 발생 시점의 컨텍스트 정보
            cause: 원인이 된 예외 (있는 경우)
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.cause = cause
        self.timestamp = datetime.now().isoformat()
        
        # 디버깅을 위한 상세 정보 추가
        self.context.update({
            "error_type": self.__class__.__name__,
            "timestamp": self.timestamp,
            "error_code": error_code
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """예외 정보를 딕셔너리로 변환합니다."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "timestamp": self.timestamp,
            "context": self.context,
            "cause": str(self.cause) if self.cause else None
        }
    
    def to_json(self) -> str:
        """예외 정보를 JSON 문자열로 변환합니다."""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


class InsufficientDataError(ChoiAlgorithmError):
    """
    데이터가 부족하여 알고리즘을 실행할 수 없는 경우 발생하는 예외
    """
    
    def __init__(
        self,
        message: str = "데이터가 부족하여 Choi 알고리즘을 실행할 수 없습니다",
        required_data: Optional[List[str]] = None,
        provided_data: Optional[Dict[str, Any]] = None,
        min_samples: Optional[int] = None,
        actual_samples: Optional[int] = None
    ):
        context = {
            "required_data": required_data or [],
            "provided_data_keys": list(provided_data.keys()) if provided_data else [],
            "min_samples": min_samples,
            "actual_samples": actual_samples
        }
        
        super().__init__(
            message=message,
            error_code="INSUFFICIENT_DATA",
            context=context
        )


class ConfigurationError(ChoiAlgorithmError):
    """
    설정 관련 오류가 발생한 경우의 예외
    """
    
    def __init__(
        self,
        message: str = "Choi 알고리즘 설정에 오류가 있습니다",
        config_section: Optional[str] = None,
        invalid_keys: Optional[List[str]] = None,
        expected_type: Optional[str] = None,
        actual_value: Optional[Any] = None
    ):
        context = {
            "config_section": config_section,
            "invalid_keys": invalid_keys or [],
            "expected_type": expected_type,
            "actual_value": str(actual_value) if actual_value is not None else None
        }
        
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            context=context
        )


class FilteringError(ChoiAlgorithmError):
    """
    필터링 단계에서 발생하는 예외
    """
    
    def __init__(
        self,
        message: str = "필터링 단계에서 오류가 발생했습니다",
        step: Optional[str] = None,
        input_data_shape: Optional[Dict[str, Any]] = None,
        filter_ratio: Optional[float] = None,
        threshold_values: Optional[Dict[str, float]] = None
    ):
        context = {
            "filtering_step": step,
            "input_data_shape": input_data_shape,
            "filter_ratio": filter_ratio,
            "threshold_values": threshold_values or {}
        }
        
        super().__init__(
            message=message,
            error_code="FILTERING_ERROR",
            context=context
        )


class AbnormalDetectionError(ChoiAlgorithmError):
    """
    이상 탐지 단계에서 발생하는 예외
    """
    
    def __init__(
        self,
        message: str = "이상 탐지 단계에서 오류가 발생했습니다",
        detector_name: Optional[str] = None,
        detection_type: Optional[str] = None,
        input_data_info: Optional[Dict[str, Any]] = None,
        detector_config: Optional[Dict[str, Any]] = None
    ):
        context = {
            "detector_name": detector_name,
            "detection_type": detection_type,
            "input_data_info": input_data_info,
            "detector_config": detector_config or {}
        }
        
        super().__init__(
            message=message,
            error_code="ABNORMAL_DETECTION_ERROR",
            context=context
        )


class KPIAnalysisError(ChoiAlgorithmError):
    """
    KPI 분석 단계에서 발생하는 예외
    """
    
    def __init__(
        self,
        message: str = "KPI 분석 단계에서 오류가 발생했습니다",
        analyzer_name: Optional[str] = None,
        kpi_topic: Optional[str] = None,
        analysis_parameters: Optional[Dict[str, Any]] = None,
        input_statistics: Optional[Dict[str, Any]] = None
    ):
        context = {
            "analyzer_name": analyzer_name,
            "kpi_topic": kpi_topic,
            "analysis_parameters": analysis_parameters or {},
            "input_statistics": input_statistics or {}
        }
        
        super().__init__(
            message=message,
            error_code="KPI_ANALYSIS_ERROR",
            context=context
        )


class DataValidationError(ChoiAlgorithmError):
    """
    데이터 검증 실패 시 발생하는 예외
    """
    
    def __init__(
        self,
        message: str = "데이터 검증에 실패했습니다",
        validation_rule: Optional[str] = None,
        field_name: Optional[str] = None,
        expected_value: Optional[Any] = None,
        actual_value: Optional[Any] = None,
        validation_errors: Optional[List[str]] = None
    ):
        context = {
            "validation_rule": validation_rule,
            "field_name": field_name,
            "expected_value": str(expected_value) if expected_value is not None else None,
            "actual_value": str(actual_value) if actual_value is not None else None,
            "validation_errors": validation_errors or []
        }
        
        super().__init__(
            message=message,
            error_code="DATA_VALIDATION_ERROR",
            context=context
        )


class StrategyExecutionError(ChoiAlgorithmError):
    """
    Strategy 실행 중 발생하는 예외
    """
    
    def __init__(
        self,
        message: str = "Strategy 실행 중 오류가 발생했습니다",
        strategy_name: Optional[str] = None,
        strategy_type: Optional[str] = None,
        execution_step: Optional[str] = None,
        input_data_summary: Optional[Dict[str, Any]] = None
    ):
        context = {
            "strategy_name": strategy_name,
            "strategy_type": strategy_type,
            "execution_step": execution_step,
            "input_data_summary": input_data_summary or {}
        }
        
        super().__init__(
            message=message,
            error_code="STRATEGY_EXECUTION_ERROR",
            context=context
        )


class DIMSDataError(ChoiAlgorithmError):
    """
    DIMS 데이터 관련 예외
    """
    
    def __init__(
        self,
        message: str = "DIMS 데이터 처리 중 오류가 발생했습니다",
        peg_name: Optional[str] = None,
        dims_operation: Optional[str] = None,
        provider_info: Optional[Dict[str, Any]] = None,
        availability_ratio: Optional[float] = None
    ):
        context = {
            "peg_name": peg_name,
            "dims_operation": dims_operation,
            "provider_info": provider_info or {},
            "availability_ratio": availability_ratio
        }
        
        super().__init__(
            message=message,
            error_code="DIMS_DATA_ERROR",
            context=context
        )


class PerformanceError(ChoiAlgorithmError):
    """
    성능 관련 예외 (타임아웃, 메모리 부족 등)
    """
    
    def __init__(
        self,
        message: str = "성능 관련 오류가 발생했습니다",
        operation: Optional[str] = None,
        duration_ms: Optional[float] = None,
        memory_usage_mb: Optional[float] = None,
        threshold_exceeded: Optional[str] = None
    ):
        context = {
            "operation": operation,
            "duration_ms": duration_ms,
            "memory_usage_mb": memory_usage_mb,
            "threshold_exceeded": threshold_exceeded
        }
        
        super().__init__(
            message=message,
            error_code="PERFORMANCE_ERROR",
            context=context
        )


# 예외 매핑 딕셔너리 (오류 코드로 예외 클래스 찾기)
EXCEPTION_MAPPING = {
    "CHOI_ERROR": ChoiAlgorithmError,
    "INSUFFICIENT_DATA": InsufficientDataError,
    "CONFIGURATION_ERROR": ConfigurationError,
    "FILTERING_ERROR": FilteringError,
    "ABNORMAL_DETECTION_ERROR": AbnormalDetectionError,
    "KPI_ANALYSIS_ERROR": KPIAnalysisError,
    "DATA_VALIDATION_ERROR": DataValidationError,
    "STRATEGY_EXECUTION_ERROR": StrategyExecutionError,
    "DIMS_DATA_ERROR": DIMSDataError,
    "PERFORMANCE_ERROR": PerformanceError
}


def create_exception(
    error_code: str,
    message: str,
    **kwargs
) -> ChoiAlgorithmError:
    """
    오류 코드를 기반으로 적절한 예외를 생성합니다.
    
    Args:
        error_code: 오류 코드
        message: 오류 메시지
        **kwargs: 예외별 추가 매개변수
        
    Returns:
        생성된 예외 인스턴스
    """
    exception_class = EXCEPTION_MAPPING.get(error_code, ChoiAlgorithmError)
    
    try:
        return exception_class(message=message, **kwargs)
    except TypeError:
        # 매개변수가 맞지 않는 경우 기본 예외 사용
        return ChoiAlgorithmError(
            message=message,
            error_code=error_code,
            context=kwargs
        )


def handle_exception(
    exception: Exception,
    context: Optional[Dict[str, Any]] = None,
    logger_name: str = "app.exceptions"
) -> ChoiAlgorithmError:
    """
    일반 예외를 Choi 알고리즘 예외로 변환합니다.
    
    Args:
        exception: 원본 예외
        context: 추가 컨텍스트 정보
        logger_name: 로거 이름
        
    Returns:
        변환된 Choi 알고리즘 예외
    """
    import logging
    
    logger = logging.getLogger(logger_name)
    
    # 이미 Choi 예외인 경우 그대로 반환
    if isinstance(exception, ChoiAlgorithmError):
        logger.error(f"Choi 알고리즘 예외 발생: {exception.message}", extra=exception.context)
        return exception
    
    # 일반 예외를 Choi 예외로 변환
    choi_exception = ChoiAlgorithmError(
        message=f"예상치 못한 오류가 발생했습니다: {str(exception)}",
        error_code="UNEXPECTED_ERROR",
        context=context or {},
        cause=exception
    )
    
    logger.error(
        f"예상치 못한 예외를 Choi 예외로 변환: {str(exception)}",
        extra=choi_exception.context,
        exc_info=True
    )
    
    return choi_exception
