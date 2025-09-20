"""
로깅 데코레이터 모듈

이 모듈은 Choi 알고리즘 서비스 메서드들의 진입/종료 지점에서
자동으로 로깅을 수행하는 데코레이터들을 제공합니다.
"""

import functools
import time
import logging
import traceback
from typing import Any, Callable, Dict, Optional, Union
from datetime import datetime

from app.exceptions import ChoiAlgorithmError, handle_exception


def log_service_method(
    logger_name: Optional[str] = None,
    log_params: bool = True,
    log_result: bool = False,
    performance_threshold_ms: float = 1000.0,
    mask_sensitive_fields: Optional[list] = None
):
    """
    서비스 메서드의 진입/종료 지점을 로깅하는 데코레이터
    
    Args:
        logger_name: 사용할 로거 이름 (None이면 모듈명 기반 자동 생성)
        log_params: 매개변수 로깅 여부
        log_result: 결과 로깅 여부 (민감한 데이터 주의)
        performance_threshold_ms: 성능 경고 임계값 (밀리초)
        mask_sensitive_fields: 마스킹할 민감한 필드명 목록
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 로거 설정
            if logger_name:
                logger = logging.getLogger(logger_name)
            else:
                module_name = func.__module__
                logger = logging.getLogger(f"{module_name}.{func.__qualname__}")
            
            # 시작 시간 기록
            start_time = time.perf_counter()
            start_timestamp = datetime.now().isoformat()
            
            # 매개변수 정보 준비
            params_info = {}
            if log_params:
                params_info = _prepare_params_info(
                    func, args, kwargs, mask_sensitive_fields or []
                )
            
            # 메서드 진입 로깅
            logger.info(
                f"🔵 {func.__name__} 메서드 시작",
                extra={
                    "method_name": func.__name__,
                    "source_module": func.__module__,
                    "start_timestamp": start_timestamp,
                    "parameters": params_info if log_params else {},
                    "event_type": "method_start"
                }
            )
            
            try:
                # 실제 메서드 실행
                result = func(*args, **kwargs)
                
                # 실행 시간 계산
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                
                # 결과 정보 준비
                result_info = {}
                if log_result and result is not None:
                    result_info = _prepare_result_info(result, mask_sensitive_fields or [])
                
                # 성공 로깅
                log_level = logging.WARNING if duration_ms > performance_threshold_ms else logging.INFO
                logger.log(
                    log_level,
                    f"✅ {func.__name__} 메서드 완료 ({duration_ms:.2f}ms)",
                    extra={
                        "method_name": func.__name__,
                        "source_module": func.__module__,
                        "duration_ms": round(duration_ms, 2),
                        "success": True,
                        "result_info": result_info if log_result else {},
                        "performance_warning": duration_ms > performance_threshold_ms,
                        "event_type": "method_success"
                    }
                )
                
                return result
                
            except Exception as e:
                # 실행 시간 계산 (오류 발생 시에도)
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                
                # 예외를 Choi 예외로 변환
                choi_exception = handle_exception(
                    e,
                    context={
                        "method_name": func.__name__,
                        "module": func.__module__,
                        "parameters": params_info if log_params else {},
                        "duration_ms": round(duration_ms, 2)
                    },
                    logger_name=logger.name
                )
                
                # 오류 로깅
                logger.error(
                    f"❌ {func.__name__} 메서드 실패 ({duration_ms:.2f}ms): {str(e)}",
                    extra={
                        "method_name": func.__name__,
                        "source_module": func.__module__,
                        "duration_ms": round(duration_ms, 2),
                        "success": False,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "choi_error_code": choi_exception.error_code,
                        "event_type": "method_error"
                    },
                    exc_info=True
                )
                
                # Choi 예외 다시 발생
                raise choi_exception
        
        return wrapper
    return decorator


def log_strategy_execution(
    strategy_type: str,
    logger_name: Optional[str] = None
):
    """
    Strategy 실행을 로깅하는 전용 데코레이터
    
    Args:
        strategy_type: Strategy 타입 ('filtering', 'judgement' 등)
        logger_name: 사용할 로거 이름
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # 로거 설정
            if logger_name:
                logger = logging.getLogger(logger_name)
            else:
                logger = logging.getLogger(f"app.services.strategies.{self.__class__.__name__}")
            
            start_time = time.perf_counter()
            strategy_name = self.__class__.__name__
            
            # Strategy 실행 시작 로깅
            logger.info(
                f"🚀 {strategy_type} Strategy 시작: {strategy_name}",
                extra={
                    "strategy_name": strategy_name,
                    "strategy_type": strategy_type,
                    "method_name": func.__name__,
                    "event_type": "strategy_start"
                }
            )
            
            try:
                result = func(self, *args, **kwargs)
                
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                logger.info(
                    f"✨ {strategy_type} Strategy 완료: {strategy_name} ({duration_ms:.2f}ms)",
                    extra={
                        "strategy_name": strategy_name,
                        "strategy_type": strategy_type,
                        "method_name": func.__name__,
                        "duration_ms": round(duration_ms, 2),
                        "success": True,
                        "event_type": "strategy_success"
                    }
                )
                
                return result
                
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                logger.error(
                    f"💥 {strategy_type} Strategy 실패: {strategy_name} ({duration_ms:.2f}ms) - {str(e)}",
                    extra={
                        "strategy_name": strategy_name,
                        "strategy_type": strategy_type,
                        "method_name": func.__name__,
                        "duration_ms": round(duration_ms, 2),
                        "success": False,
                        "error_message": str(e),
                        "event_type": "strategy_error"
                    },
                    exc_info=True
                )
                
                raise
        
        return wrapper
    return decorator


def log_detector_execution(logger_name: Optional[str] = None):
    """
    이상 탐지기 실행을 로깅하는 데코레이터
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if logger_name:
                logger = logging.getLogger(logger_name)
            else:
                logger = logging.getLogger(f"app.services.anomaly_detectors.{self.__class__.__name__}")
            
            start_time = time.perf_counter()
            detector_name = self.__class__.__name__
            
            logger.debug(
                f"🔍 이상 탐지 시작: {detector_name}",
                extra={
                    "detector_name": detector_name,
                    "method_name": func.__name__,
                    "event_type": "detector_start"
                }
            )
            
            try:
                result = func(self, *args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                # 탐지 결과 요약
                affected_cells = getattr(result, 'affected_cells', 0)
                affected_pegs = getattr(result, 'affected_pegs', 0)
                
                logger.info(
                    f"🔍 {detector_name} detection completed: {affected_cells} cells, {affected_pegs} PEGs affected",
                    extra={
                        "detector_name": detector_name,
                        "method_name": func.__name__,
                        "duration_ms": round(duration_ms, 2),
                        "affected_cells": affected_cells,
                        "affected_pegs": affected_pegs,
                        "event_type": "detector_success"
                    }
                )
                
                return result
                
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                logger.error(
                    f"🔍 이상 탐지 실패: {detector_name} - {str(e)}",
                    extra={
                        "detector_name": detector_name,
                        "method_name": func.__name__,
                        "duration_ms": round(duration_ms, 2),
                        "error_message": str(e),
                        "event_type": "detector_error"
                    },
                    exc_info=True
                )
                
                raise
        
        return wrapper
    return decorator


def log_analyzer_execution(logger_name: Optional[str] = None):
    """
    KPI 분석기 실행을 로깅하는 데코레이터
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if logger_name:
                logger = logging.getLogger(logger_name)
            else:
                logger = logging.getLogger(f"app.services.kpi_analyzers.{self.__class__.__name__}")
            
            start_time = time.perf_counter()
            analyzer_name = self.__class__.__name__
            
            logger.debug(
                f"📊 KPI 분석 시작: {analyzer_name}",
                extra={
                    "analyzer_name": analyzer_name,
                    "method_name": func.__name__,
                    "event_type": "analyzer_start"
                }
            )
            
            try:
                result = func(self, *args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                # 분석 결과 요약
                judgement = getattr(result, 'judgement', 'Unknown') if result else 'No Result'
                confidence = getattr(result, 'confidence', 0.0) if result else 0.0
                
                logger.debug(
                    f"📊 KPI 분석 완료: {analyzer_name} → {judgement} (신뢰도: {confidence:.2f})",
                    extra={
                        "analyzer_name": analyzer_name,
                        "method_name": func.__name__,
                        "duration_ms": round(duration_ms, 2),
                        "judgement": judgement,
                        "confidence": confidence,
                        "event_type": "analyzer_success"
                    }
                )
                
                return result
                
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                logger.error(
                    f"📊 KPI 분석 실패: {analyzer_name} - {str(e)}",
                    extra={
                        "analyzer_name": analyzer_name,
                        "method_name": func.__name__,
                        "duration_ms": round(duration_ms, 2),
                        "error_message": str(e),
                        "event_type": "analyzer_error"
                    },
                    exc_info=True
                )
                
                raise
        
        return wrapper
    return decorator


def _prepare_params_info(
    func: Callable,
    args: tuple,
    kwargs: dict,
    mask_sensitive_fields: list
) -> Dict[str, Any]:
    """매개변수 정보를 준비합니다 (민감한 정보 마스킹 포함)."""
    params_info = {}
    
    try:
        # 함수 시그니처 분석
        import inspect
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        for param_name, value in bound_args.arguments.items():
            if param_name == 'self':
                params_info[param_name] = f"<{type(value).__name__} instance>"
            elif param_name in mask_sensitive_fields:
                params_info[param_name] = "***MASKED***"
            elif isinstance(value, (dict, list)):
                params_info[param_name] = f"<{type(value).__name__} with {len(value)} items>"
            elif hasattr(value, '__len__') and not isinstance(value, str):
                params_info[param_name] = f"<{type(value).__name__} with {len(value)} items>"
            else:
                params_info[param_name] = str(value)[:100]  # 최대 100자로 제한
                
    except Exception:
        # 매개변수 분석 실패 시 기본 정보만 제공
        params_info = {
            "args_count": len(args),
            "kwargs_keys": list(kwargs.keys())
        }
    
    return params_info


def _prepare_result_info(result: Any, mask_sensitive_fields: list) -> Dict[str, Any]:
    """결과 정보를 준비합니다 (민감한 정보 마스킹 포함)."""
    result_info = {}
    
    try:
        if hasattr(result, '__dict__'):
            # 객체인 경우 주요 속성만 추출
            result_info["type"] = type(result).__name__
            for attr_name in dir(result):
                if not attr_name.startswith('_') and not callable(getattr(result, attr_name)):
                    if attr_name in mask_sensitive_fields:
                        result_info[attr_name] = "***MASKED***"
                    else:
                        attr_value = getattr(result, attr_name)
                        if isinstance(attr_value, (dict, list)):
                            result_info[attr_name] = f"<{type(attr_value).__name__} with {len(attr_value)} items>"
                        else:
                            result_info[attr_name] = str(attr_value)[:50]
        elif isinstance(result, (dict, list)):
            result_info = {
                "type": type(result).__name__,
                "length": len(result),
                "sample": str(result)[:100] if result else "empty"
            }
        else:
            result_info = {
                "type": type(result).__name__,
                "value": str(result)[:100]
            }
            
    except Exception:
        result_info = {
            "type": type(result).__name__,
            "note": "결과 정보 추출 실패"
        }
    
    return result_info
