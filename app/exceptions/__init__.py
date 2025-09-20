"""
Choi 알고리즘 예외 처리 모듈

이 패키지는 Choi 알고리즘 실행 중 발생할 수 있는 
다양한 예외 상황을 처리하기 위한 커스텀 예외 클래스들을 제공합니다.
"""

from .choi_exceptions import (
    # 기본 예외
    ChoiAlgorithmError,
    
    # 데이터 관련 예외
    InsufficientDataError,
    DataValidationError,
    DIMSDataError,
    
    # 설정 관련 예외
    ConfigurationError,
    
    # 알고리즘 단계별 예외
    FilteringError,
    AbnormalDetectionError,
    KPIAnalysisError,
    StrategyExecutionError,
    
    # 성능 관련 예외
    PerformanceError,
    
    # 유틸리티 함수
    create_exception,
    handle_exception,
    EXCEPTION_MAPPING
)

__all__ = [
    # 예외 클래스들
    "ChoiAlgorithmError",
    "InsufficientDataError",
    "DataValidationError",
    "DIMSDataError",
    "ConfigurationError",
    "FilteringError",
    "AbnormalDetectionError",
    "KPIAnalysisError",
    "StrategyExecutionError",
    "PerformanceError",
    
    # 유틸리티
    "create_exception",
    "handle_exception",
    "EXCEPTION_MAPPING"
]
