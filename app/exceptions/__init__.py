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

# 순환 참조를 피하기 위해 API 예외들은 별도 모듈에서 import
# (app/api_exceptions.py로 분리하여 __init__ 모듈 초기화 시 순환 import 방지)
try:
    from ..api_exceptions import (
        # 기본 API 예외
        BaseAPIException,
        
        # 분석 결과 관련 예외
        AnalysisResultNotFoundException,
        InvalidAnalysisDataException,
        DatabaseConnectionException,
        DuplicateAnalysisResultException,
        
        # 사용자 설정 관련 예외
        UserPreferenceNotFoundException,
        InvalidPreferenceDataException,
        PreferenceImportException,
        
        # 필터 관련 예외
        InvalidFilterException,
        TargetValidationException,
        HostValidationException,
        RelationshipValidationException,
        FilterCombinationException,
        
        # 데이터베이스 관련 예외
        MongoDBIndexException,
        
        # LLM 분석 관련 예외
        LLMAnalysisException,
        
        # 예외 핸들러 함수들
        base_api_exception_handler,
        analysis_result_not_found_handler,
        invalid_analysis_data_handler,
        database_connection_handler,
        duplicate_analysis_result_handler,
        user_preference_not_found_handler,
        invalid_preference_data_handler,
        preference_import_handler,
        general_exception_handler,
        target_validation_exception_handler,
        host_validation_exception_handler,
        relationship_validation_exception_handler,
        filter_combination_exception_handler,
        mongodb_index_exception_handler,
        llm_analysis_exception_handler,
        
        # 편의 함수들
        raise_host_validation_error,
        raise_relationship_validation_error,
        raise_filter_combination_error
    )
except Exception:
    # 일부 런타임(예: 문서 생성/정적 분석)에서 api_exceptions가 아직 준비되지 않은 경우를 대비
    # 실제 실행 환경에서는 반드시 존재해야 함
    pass

# PEG 비교분석 예외들
from .peg_comparison_exceptions import (
    PEGComparisonException,
    MCPConnectionError,
    MCPTimeoutError,
    DataValidationError as PEGDataValidationError,
    AnalysisDataNotFoundError,
    CacheError,
    AsyncTaskError,
    AsyncTaskNotFoundError,
    RateLimitExceededError,
    PermissionDeniedError,
    AlgorithmVersionError,
    ProcessingTimeoutError
)

__all__ = [
    # Choi 알고리즘 예외 클래스들
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
    
    # API 예외 클래스들
    "BaseAPIException",
    "AnalysisResultNotFoundException",
    "InvalidAnalysisDataException",
    "DatabaseConnectionException",
    "DuplicateAnalysisResultException",
    "UserPreferenceNotFoundException",
    "InvalidPreferenceDataException",
    "PreferenceImportException",
    "InvalidFilterException",
    "TargetValidationException",
    "HostValidationException",
    "RelationshipValidationException",
    "FilterCombinationException",
    "MongoDBIndexException",
    "LLMAnalysisException",
    
    # PEG 비교분석 예외 클래스들
    "PEGComparisonException",
    "MCPConnectionError",
    "MCPTimeoutError",
    "PEGDataValidationError",
    "AnalysisDataNotFoundError",
    "CacheError",
    "AsyncTaskError",
    "AsyncTaskNotFoundError",
    "RateLimitExceededError",
    "PermissionDeniedError",
    "AlgorithmVersionError",
    "ProcessingTimeoutError",
    
    # 예외 핸들러 함수들
    "base_api_exception_handler",
    "analysis_result_not_found_handler",
    "invalid_analysis_data_handler",
    "database_connection_handler",
    "duplicate_analysis_result_handler",
    "user_preference_not_found_handler",
    "invalid_preference_data_handler",
    "preference_import_handler",
    "general_exception_handler",
    "target_validation_exception_handler",
    "host_validation_exception_handler",
    "relationship_validation_exception_handler",
    "filter_combination_exception_handler",
    "mongodb_index_exception_handler",
    "llm_analysis_exception_handler",
    
    # 편의 함수들
    "raise_host_validation_error",
    "raise_relationship_validation_error",
    "raise_filter_combination_error",
    
    # Choi 알고리즘 유틸리티
    "create_exception",
    "handle_exception",
    "EXCEPTION_MAPPING"
]
