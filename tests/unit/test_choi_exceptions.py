"""
Choi 알고리즘 커스텀 예외 처리 테스트

이 모듈은 Choi 알고리즘에서 사용하는 커스텀 예외 클래스들의
정상 동작을 검증합니다.
"""

import pytest
import json
from datetime import datetime
from typing import Dict, Any

from app.exceptions import (
    ChoiAlgorithmError,
    InsufficientDataError,
    ConfigurationError,
    FilteringError,
    AbnormalDetectionError,
    KPIAnalysisError,
    DataValidationError,
    StrategyExecutionError,
    DIMSDataError,
    PerformanceError,
    create_exception,
    handle_exception,
    EXCEPTION_MAPPING
)


class TestChoiAlgorithmError:
    """기본 Choi 알고리즘 예외 테스트"""
    
    def test_basic_exception_creation(self):
        """기본 예외 생성 테스트"""
        error = ChoiAlgorithmError("테스트 오류 메시지")
        
        assert error.message == "테스트 오류 메시지"
        assert error.error_code == "CHOI_ERROR"
        assert isinstance(error.context, dict)
        assert "error_type" in error.context
        assert "timestamp" in error.context
    
    def test_exception_with_context(self):
        """컨텍스트 정보가 포함된 예외 테스트"""
        context = {
            "input_data_size": 100,
            "operation": "filtering",
            "parameters": {"threshold": 0.5}
        }
        
        error = ChoiAlgorithmError(
            "컨텍스트 포함 오류",
            error_code="TEST_ERROR",
            context=context
        )
        
        assert error.error_code == "TEST_ERROR"
        assert error.context["input_data_size"] == 100
        assert error.context["operation"] == "filtering"
        assert error.context["error_type"] == "ChoiAlgorithmError"
    
    def test_exception_with_cause(self):
        """원인 예외가 포함된 예외 테스트"""
        original_error = ValueError("원본 오류")
        
        error = ChoiAlgorithmError(
            "래핑된 오류",
            cause=original_error
        )
        
        assert error.cause == original_error
        assert str(error.cause) == "원본 오류"
    
    def test_exception_serialization(self):
        """예외 직렬화 테스트"""
        error = ChoiAlgorithmError(
            "직렬화 테스트",
            error_code="SERIALIZATION_TEST",
            context={"test_field": "test_value"}
        )
        
        # 딕셔너리 변환
        error_dict = error.to_dict()
        assert error_dict["message"] == "직렬화 테스트"
        assert error_dict["error_code"] == "SERIALIZATION_TEST"
        assert error_dict["context"]["test_field"] == "test_value"
        
        # JSON 변환
        error_json = error.to_json()
        parsed = json.loads(error_json)
        assert parsed["message"] == "직렬화 테스트"


class TestSpecificExceptions:
    """특정 예외 클래스들 테스트"""
    
    def test_insufficient_data_error(self):
        """데이터 부족 예외 테스트"""
        error = InsufficientDataError(
            "데이터가 부족합니다",
            required_data=["peg_data", "cell_ids"],
            provided_data={"peg_data": []},
            min_samples=10,
            actual_samples=5
        )
        
        assert error.error_code == "INSUFFICIENT_DATA"
        assert error.context["required_data"] == ["peg_data", "cell_ids"]
        assert error.context["min_samples"] == 10
        assert error.context["actual_samples"] == 5
    
    def test_configuration_error(self):
        """설정 오류 예외 테스트"""
        error = ConfigurationError(
            "설정이 잘못되었습니다",
            config_section="filtering",
            invalid_keys=["min_threshold", "max_threshold"],
            expected_type="float",
            actual_value="invalid_string"
        )
        
        assert error.error_code == "CONFIGURATION_ERROR"
        assert error.context["config_section"] == "filtering"
        assert error.context["invalid_keys"] == ["min_threshold", "max_threshold"]
    
    def test_filtering_error(self):
        """필터링 오류 예외 테스트"""
        error = FilteringError(
            "필터링 단계 오류",
            step="normalization",
            input_data_shape={"cells": 5, "pegs": 15},
            filter_ratio=0.3,
            threshold_values={"min": 0.1, "max": 10.0}
        )
        
        assert error.error_code == "FILTERING_ERROR"
        assert error.context["filtering_step"] == "normalization"
        assert error.context["filter_ratio"] == 0.3
    
    def test_abnormal_detection_error(self):
        """이상 탐지 오류 예외 테스트"""
        error = AbnormalDetectionError(
            "Range 탐지 오류",
            detector_name="RangeAnomalyDetector",
            detection_type="Range",
            input_data_info={"cells": 3, "pegs": 9},
            detector_config={"enable_range_check": True}
        )
        
        assert error.error_code == "ABNORMAL_DETECTION_ERROR"
        assert error.context["detector_name"] == "RangeAnomalyDetector"
        assert error.context["detection_type"] == "Range"
    
    def test_kpi_analysis_error(self):
        """KPI 분석 오류 예외 테스트"""
        error = KPIAnalysisError(
            "Similar 분석 오류",
            analyzer_name="SimilarAnalyzer",
            kpi_topic="AirMacDLThruAvg",
            analysis_parameters={"beta_1": 5, "beta_2": 10},
            input_statistics={"pre_mean": 100, "post_mean": 150}
        )
        
        assert error.error_code == "KPI_ANALYSIS_ERROR"
        assert error.context["analyzer_name"] == "SimilarAnalyzer"
        assert error.context["kpi_topic"] == "AirMacDLThruAvg"
    
    def test_dims_data_error(self):
        """DIMS 데이터 오류 예외 테스트"""
        error = DIMSDataError(
            "DIMS 연결 실패",
            peg_name="AirMacDLThruAvg",
            dims_operation="get_peg_range",
            provider_info={"provider": "MockDimsDataProvider"},
            availability_ratio=0.7
        )
        
        assert error.error_code == "DIMS_DATA_ERROR"
        assert error.context["peg_name"] == "AirMacDLThruAvg"
        assert error.context["availability_ratio"] == 0.7
    
    def test_performance_error(self):
        """성능 오류 예외 테스트"""
        error = PerformanceError(
            "처리 시간 초과",
            operation="filtering",
            duration_ms=6000.0,
            memory_usage_mb=512.0,
            threshold_exceeded="duration"
        )
        
        assert error.error_code == "PERFORMANCE_ERROR"
        assert error.context["operation"] == "filtering"
        assert error.context["duration_ms"] == 6000.0


class TestExceptionUtilities:
    """예외 유틸리티 함수들 테스트"""
    
    def test_create_exception_known_code(self):
        """알려진 오류 코드로 예외 생성 테스트"""
        error = create_exception(
            "INSUFFICIENT_DATA",
            "데이터 부족",
            required_data=["peg_data"],
            min_samples=10
        )
        
        assert isinstance(error, InsufficientDataError)
        assert error.message == "데이터 부족"
        assert error.context["required_data"] == ["peg_data"]
    
    def test_create_exception_unknown_code(self):
        """알 수 없는 오류 코드로 예외 생성 테스트"""
        error = create_exception(
            "UNKNOWN_ERROR",
            "알 수 없는 오류",
            custom_field="custom_value"
        )
        
        assert isinstance(error, ChoiAlgorithmError)
        assert error.error_code == "UNKNOWN_ERROR"
        assert error.context["custom_field"] == "custom_value"
    
    def test_handle_exception_choi_exception(self):
        """이미 Choi 예외인 경우 처리 테스트"""
        original_error = FilteringError("원본 필터링 오류")
        
        handled_error = handle_exception(original_error)
        
        # 원본 예외가 그대로 반환되어야 함
        assert handled_error is original_error
        assert isinstance(handled_error, FilteringError)
    
    def test_handle_exception_general_exception(self):
        """일반 예외를 Choi 예외로 변환 테스트"""
        original_error = ValueError("일반 ValueError")
        
        handled_error = handle_exception(
            original_error,
            context={"operation": "test_operation"}
        )
        
        assert isinstance(handled_error, ChoiAlgorithmError)
        assert handled_error.error_code == "UNEXPECTED_ERROR"
        assert handled_error.cause == original_error
        assert handled_error.context["operation"] == "test_operation"
    
    def test_exception_mapping_completeness(self):
        """예외 매핑 딕셔너리 완전성 테스트"""
        expected_codes = [
            "CHOI_ERROR",
            "INSUFFICIENT_DATA",
            "CONFIGURATION_ERROR",
            "FILTERING_ERROR",
            "ABNORMAL_DETECTION_ERROR",
            "KPI_ANALYSIS_ERROR",
            "DATA_VALIDATION_ERROR",
            "STRATEGY_EXECUTION_ERROR",
            "DIMS_DATA_ERROR",
            "PERFORMANCE_ERROR"
        ]
        
        for code in expected_codes:
            assert code in EXCEPTION_MAPPING
            exception_class = EXCEPTION_MAPPING[code]
            assert issubclass(exception_class, ChoiAlgorithmError)


class TestExceptionIntegration:
    """예외 처리 통합 테스트"""
    
    def test_exception_context_preservation(self):
        """예외 컨텍스트 보존 테스트"""
        # 복잡한 컨텍스트 정보
        complex_context = {
            "algorithm_step": "filtering",
            "data_summary": {
                "cells": 10,
                "pegs": 30,
                "time_slots": 240
            },
            "config_values": {
                "min_threshold": 0.1,
                "max_threshold": 10.0
            },
            "performance_info": {
                "start_time": datetime.now().isoformat(),
                "memory_usage_mb": 45.2
            }
        }
        
        error = FilteringError(
            "복잡한 컨텍스트 테스트",
            step="threshold_application",
            input_data_shape=complex_context["data_summary"],
            threshold_values=complex_context["config_values"]
        )
        
        # 직렬화/역직렬화 테스트
        error_json = error.to_json()
        parsed = json.loads(error_json)
        
        assert parsed["error_code"] == "FILTERING_ERROR"
        assert parsed["context"]["filtering_step"] == "threshold_application"
        assert parsed["context"]["input_data_shape"]["cells"] == 10
    
    def test_nested_exception_handling(self):
        """중첩된 예외 처리 테스트"""
        # 원본 예외
        original = ValueError("원본 ValueError")
        
        # 첫 번째 래핑
        wrapped_once = handle_exception(
            original,
            context={"level": 1, "operation": "first_wrap"}
        )
        
        # 두 번째 래핑 (이미 Choi 예외)
        wrapped_twice = handle_exception(
            wrapped_once,
            context={"level": 2, "operation": "second_wrap"}
        )
        
        # 두 번째 래핑은 원본을 그대로 반환해야 함
        assert wrapped_twice is wrapped_once
        assert wrapped_twice.cause == original
        assert wrapped_twice.context["level"] == 1  # 첫 번째 컨텍스트 유지
