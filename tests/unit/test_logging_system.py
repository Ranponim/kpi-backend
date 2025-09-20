"""
로깅 시스템 테스트

이 모듈은 Choi 알고리즘의 포괄적 로깅 시스템이
정상적으로 동작하는지 검증합니다.
"""

import pytest
import logging
import time
from unittest.mock import Mock, patch
from typing import Dict, Any

from app.utils.logging_decorators import (
    log_service_method,
    log_strategy_execution,
    log_detector_execution,
    log_analyzer_execution
)
from app.exceptions import ChoiAlgorithmError, FilteringError


class TestLoggingDecorators:
    """로깅 데코레이터 테스트"""
    
    def test_service_method_logging_success(self, caplog):
        """서비스 메서드 성공 로깅 테스트"""
        
        @log_service_method(log_params=True, log_result=True)
        def test_method(param1: str, param2: int = 10):
            """테스트용 메서드"""
            time.sleep(0.001)  # 짧은 지연
            return {"result": "success", "processed": param2}
        
        with caplog.at_level(logging.INFO):
            result = test_method("test_value", param2=20)
        
        # 결과 검증
        assert result == {"result": "success", "processed": 20}
        
        # 로그 검증
        log_messages = [record.message for record in caplog.records]
        
        # 시작 로그 확인
        start_logs = [msg for msg in log_messages if "메서드 시작" in msg]
        assert len(start_logs) == 1
        assert "🔵 test_method 메서드 시작" in start_logs[0]
        
        # 완료 로그 확인
        success_logs = [msg for msg in log_messages if "메서드 완료" in msg]
        assert len(success_logs) == 1
        assert "✅ test_method 메서드 완료" in success_logs[0]
    
    def test_service_method_logging_error(self, caplog):
        """서비스 메서드 오류 로깅 테스트"""
        
        @log_service_method(log_params=True)
        def failing_method(param1: str):
            """실패하는 테스트 메서드"""
            raise ValueError("테스트 오류")
        
        with caplog.at_level(logging.INFO):
            with pytest.raises(ChoiAlgorithmError):
                failing_method("test_param")
        
        # 로그 검증
        log_messages = [record.message for record in caplog.records]
        
        # 오류 로그 확인
        error_logs = [msg for msg in log_messages if "메서드 실패" in msg]
        assert len(error_logs) == 1
        assert "❌ failing_method 메서드 실패" in error_logs[0]
    
    def test_strategy_execution_logging(self, caplog):
        """Strategy 실행 로깅 테스트"""
        
        class TestStrategy:
            @log_strategy_execution("test_strategy")
            def apply(self, data: Dict[str, Any]):
                """테스트 Strategy 메서드"""
                time.sleep(0.001)
                return {"status": "completed"}
        
        strategy = TestStrategy()
        
        with caplog.at_level(logging.INFO):
            result = strategy.apply({"test": "data"})
        
        assert result == {"status": "completed"}
        
        # 로그 검증
        log_messages = [record.message for record in caplog.records]
        
        # Strategy 시작 로그
        start_logs = [msg for msg in log_messages if "Strategy 시작" in msg]
        assert len(start_logs) == 1
        assert "🚀 test_strategy Strategy 시작: TestStrategy" in start_logs[0]
        
        # Strategy 완료 로그
        success_logs = [msg for msg in log_messages if "Strategy 완료" in msg]
        assert len(success_logs) == 1
        assert "✨ test_strategy Strategy 완료: TestStrategy" in success_logs[0]
    
    def test_detector_execution_logging(self, caplog):
        """탐지기 실행 로깅 테스트"""
        
        class TestDetector:
            @log_detector_execution()
            def detect(self, peg_data: Dict, config: Dict):
                """테스트 탐지기 메서드"""
                # 결과 객체 모킹
                result = Mock()
                result.affected_cells = 2
                result.affected_pegs = 5
                return result
        
        detector = TestDetector()
        
        with caplog.at_level(logging.DEBUG):
            result = detector.detect({}, {})
        
        assert result.affected_cells == 2
        assert result.affected_pegs == 5
        
        # 로그 검증 (DEBUG 레벨이므로 시작 로그 확인)
        debug_messages = [record.message for record in caplog.records if record.levelno == logging.DEBUG]
        assert any("🔍 이상 탐지 시작: TestDetector" in msg for msg in debug_messages)
    
    def test_analyzer_execution_logging(self, caplog):
        """분석기 실행 로깅 테스트"""
        
        class TestAnalyzer:
            @log_analyzer_execution()
            def analyze(self, pre_stats, post_stats, compare_metrics, config):
                """테스트 분석기 메서드"""
                result = Mock()
                result.judgement = "Similar"
                result.confidence = 0.95
                return result
        
        analyzer = TestAnalyzer()
        
        with caplog.at_level(logging.DEBUG):
            result = analyzer.analyze(None, None, None, {})
        
        assert result.judgement == "Similar"
        assert result.confidence == 0.95
        
        # 로그 검증
        debug_messages = [record.message for record in caplog.records if record.levelno == logging.DEBUG]
        assert any("📊 KPI 분석 시작: TestAnalyzer" in msg for msg in debug_messages)
        assert any("📊 KPI 분석 완료: TestAnalyzer → Similar" in msg for msg in debug_messages)


class TestLoggingIntegration:
    """로깅 시스템 통합 테스트"""
    
    def test_performance_threshold_warning(self, caplog):
        """성능 임계값 경고 테스트"""
        
        @log_service_method(performance_threshold_ms=10.0)  # 매우 낮은 임계값
        def slow_method():
            """느린 메서드 시뮬레이션"""
            time.sleep(0.02)  # 20ms 지연
            return "completed"
        
        with caplog.at_level(logging.WARNING):
            result = slow_method()
        
        assert result == "completed"
        
        # 경고 로그 확인
        warning_logs = [record for record in caplog.records if record.levelno == logging.WARNING]
        assert len(warning_logs) >= 1
        
        # 성능 경고가 포함되어야 함
        warning_messages = [record.message for record in warning_logs]
        assert any("메서드 완료" in msg for msg in warning_messages)
    
    def test_sensitive_data_masking(self, caplog):
        """민감한 데이터 마스킹 테스트"""
        
        @log_service_method(log_params=True, mask_sensitive_fields=["password", "api_key"])
        def method_with_sensitive_data(username: str, password: str, api_key: str):
            """민감한 데이터가 포함된 메서드"""
            return {"status": "authenticated"}
        
        with caplog.at_level(logging.INFO):
            result = method_with_sensitive_data("user123", "secret_password", "secret_key")
        
        assert result == {"status": "authenticated"}
        
        # 로그에서 민감한 정보가 마스킹되었는지 확인
        log_text = "\n".join([record.message for record in caplog.records])
        assert "secret_password" not in log_text
        assert "secret_key" not in log_text
        assert "***MASKED***" in log_text or "user123" in log_text  # username은 마스킹되지 않음
    
    def test_exception_logging_with_context(self, caplog):
        """컨텍스트 정보가 포함된 예외 로깅 테스트"""
        
        @log_service_method()
        def method_with_context_error(data_size: int):
            """컨텍스트 정보와 함께 예외를 발생시키는 메서드"""
            if data_size < 10:
                raise FilteringError(
                    "데이터 크기가 너무 작습니다",
                    step="validation",
                    input_data_shape={"size": data_size}
                )
            return "success"
        
        with caplog.at_level(logging.ERROR):
            with pytest.raises(ChoiAlgorithmError):
                method_with_context_error(5)
        
        # 오류 로그 확인
        error_logs = [record for record in caplog.records if record.levelno == logging.ERROR]
        assert len(error_logs) >= 1
        
        # 컨텍스트 정보가 로그에 포함되었는지 확인
        error_log = error_logs[0]
        assert hasattr(error_log, 'choi_error_code')
        assert error_log.choi_error_code == "FILTERING_ERROR"
    
    def test_multiple_decorators_compatibility(self, caplog):
        """여러 데코레이터 호환성 테스트"""
        
        class TestService:
            @log_service_method()
            @log_strategy_execution("test")
            def complex_method(self, data: Dict[str, Any]):
                """여러 데코레이터가 적용된 메서드"""
                return {"processed": True}
        
        service = TestService()
        
        with caplog.at_level(logging.INFO):
            result = service.complex_method({"test": "data"})
        
        assert result == {"processed": True}
        
        # 두 데코레이터의 로그가 모두 기록되었는지 확인
        log_messages = [record.message for record in caplog.records]
        assert any("메서드 시작" in msg for msg in log_messages)
        assert any("Strategy 시작" in msg for msg in log_messages)
