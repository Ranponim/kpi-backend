"""
고급 로깅 설정 유틸리티

이 모듈은 애플리케이션의 포괄적이고 구조화된 로깅을 위한 설정을 제공합니다.
개발, 테스트, 프로덕션 환경에 따른 다양한 로깅 레벨과 포맷을 지원합니다.
"""

import logging
import logging.config
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import json
from pathlib import Path


class StructuredFormatter(logging.Formatter):
    """
    구조화된 JSON 형태의 로그 포맷터
    
    로그를 JSON 구조로 출력하여 로그 분석 도구에서 쉽게 파싱할 수 있도록 합니다.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """로그 레코드를 JSON 형태로 포맷팅합니다."""
        
        # 기본 로그 정보
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process_id": os.getpid(),
            "thread_id": record.thread
        }
        
        # 예외 정보가 있는 경우 추가
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info)
            }
        
        # 추가 컨텍스트 정보가 있는 경우 포함
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                'thread', 'threadName', 'processName', 'process', 'message'
            }:
                extra_fields[key] = value
        
        if extra_fields:
            log_data["extra"] = extra_fields
        
        return json.dumps(log_data, ensure_ascii=False, default=str)


class ColoredConsoleFormatter(logging.Formatter):
    """
    개발 환경용 컬러 콘솔 포맷터
    
    로그 레벨에 따라 다른 색상을 적용하여 가독성을 높입니다.
    """
    
    # ANSI 색상 코드
    COLORS = {
        'DEBUG': '\033[36m',     # 청록색
        'INFO': '\033[32m',      # 녹색
        'WARNING': '\033[33m',   # 노란색
        'ERROR': '\033[31m',     # 빨간색
        'CRITICAL': '\033[35m',  # 자주색
        'RESET': '\033[0m'       # 리셋
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """컬러가 적용된 로그 메시지를 생성합니다."""
        
        # 기본 포맷팅
        formatted = super().format(record)
        
        # 색상 적용 (터미널에서만)
        if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            reset = self.COLORS['RESET']
            formatted = f"{color}{formatted}{reset}"
        
        return formatted


def get_logging_config(
    environment: str = "development",
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    enable_json_logs: bool = False
) -> Dict[str, Any]:
    """
    로깅 설정을 생성합니다.
    
    Args:
        environment: 환경 (development, testing, production)
        log_level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: 로그 파일을 저장할 디렉토리
        enable_json_logs: JSON 형태 로그 활성화 여부
        
    Returns:
        logging.config에서 사용할 수 있는 설정 딕셔너리
    """
    
    # 로그 디렉토리 설정
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
    else:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
    
    # 기본 설정
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": (
                    "%(asctime)s | %(levelname)-8s | %(name)-25s | "
                    "%(funcName)-20s:%(lineno)-4d | %(message)s"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "simple": {
                "format": "%(levelname)s | %(name)s | %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "detailed" if environment == "development" else "simple",
                "stream": "ext://sys.stdout"
            }
        },
        "loggers": {
            # 애플리케이션 로거
            "app": {
                "level": log_level,
                "handlers": ["console"],
                "propagate": False
            },
            # FastAPI 관련 로거
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False
            },
            # 데이터베이스 로거
            "motor": {
                "level": "WARNING",
                "handlers": ["console"],
                "propagate": False
            },
            "pymongo": {
                "level": "WARNING", 
                "handlers": ["console"],
                "propagate": False
            }
        },
        "root": {
            "level": log_level,
            "handlers": ["console"]
        }
    }
    
    # 개발 환경: 컬러 콘솔 포맷터 사용
    if environment == "development":
        config["formatters"]["colored_console"] = {
            "()": ColoredConsoleFormatter,
            "format": (
                "%(asctime)s | %(levelname)-8s | %(name)-20s | "
                "%(funcName)-15s:%(lineno)-3d | %(message)s"
            ),
            "datefmt": "%H:%M:%S"
        }
        config["handlers"]["console"]["formatter"] = "colored_console"
    
    # JSON 로그 활성화
    if enable_json_logs:
        config["formatters"]["json"] = {
            "()": StructuredFormatter
        }
        config["handlers"]["json_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": "json",
            "filename": str(log_dir / "app.json"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf-8"
        }
        # 모든 로거에 JSON 핸들러 추가
        for logger_name in config["loggers"]:
            config["loggers"][logger_name]["handlers"].append("json_file")
        config["root"]["handlers"].append("json_file")
    
    # 프로덕션 환경: 파일 로깅 추가
    if environment == "production":
        config["handlers"].update({
            "file_info": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "detailed",
                "filename": str(log_dir / "app.log"),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 10,
                "encoding": "utf-8"
            },
            "file_error": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filename": str(log_dir / "error.log"),
                "maxBytes": 5242880,   # 5MB
                "backupCount": 10,
                "encoding": "utf-8"
            }
        })
        
        # 모든 로거에 파일 핸들러 추가
        for logger_name in config["loggers"]:
            config["loggers"][logger_name]["handlers"].extend(["file_info", "file_error"])
        config["root"]["handlers"].extend(["file_info", "file_error"])
    
    return config


def setup_logging(
    environment: Optional[str] = None,
    log_level: Optional[str] = None,
    log_dir: Optional[str] = None,
    enable_json_logs: Optional[bool] = None
) -> None:
    """
    로깅 설정을 초기화합니다.
    
    환경변수에서 설정을 읽어오거나 기본값을 사용합니다.
    """
    
    # 환경변수에서 설정 읽기
    environment = environment or os.getenv("ENVIRONMENT", "development")
    log_level = log_level or os.getenv("LOG_LEVEL", "INFO")
    log_dir = log_dir or os.getenv("LOG_DIR", "./logs")
    enable_json_logs = enable_json_logs or os.getenv("ENABLE_JSON_LOGS", "false").lower() == "true"
    
    # 로깅 설정 생성 및 적용
    config = get_logging_config(
        environment=environment,
        log_level=log_level,
        log_dir=log_dir,
        enable_json_logs=enable_json_logs
    )
    
    logging.config.dictConfig(config)
    
    # 설정 완료 로그
    logger = logging.getLogger("app.logging")
    logger.info(f"로깅 시스템 초기화 완료 - 환경: {environment}, 레벨: {log_level}")


def get_request_logger(request_id: str) -> logging.LoggerAdapter:
    """
    요청별 로거를 생성합니다.
    
    Args:
        request_id: 요청 고유 ID
        
    Returns:
        요청 정보가 포함된 LoggerAdapter
    """
    
    logger = logging.getLogger("app.request")
    
    return logging.LoggerAdapter(logger, {
        "request_id": request_id,
        "timestamp": datetime.now().isoformat()
    })


def log_performance(func_name: str, duration: float, **kwargs) -> None:
    """
    성능 정보를 로깅합니다.
    
    Args:
        func_name: 함수명
        duration: 실행 시간 (초)
        **kwargs: 추가 컨텍스트 정보
    """
    
    logger = logging.getLogger("app.performance")
    
    # 성능 로그용 추가 정보
    extra = {
        "function": func_name,
        "duration_ms": round(duration * 1000, 2),
        "performance": True,
        **kwargs
    }
    
    # 실행 시간에 따른 로그 레벨 결정
    if duration > 5.0:      # 5초 이상
        logger.warning(f"성능 경고: {func_name} 실행시간 {duration:.2f}초", extra=extra)
    elif duration > 1.0:    # 1초 이상
        logger.info(f"성능 정보: {func_name} 실행시간 {duration:.2f}초", extra=extra)
    else:
        logger.debug(f"성능 정보: {func_name} 실행시간 {duration:.2f}초", extra=extra)


# 성능 측정 데코레이터
def log_execution_time(logger_name: str = "app.performance"):
    """
    함수 실행 시간을 측정하고 로깅하는 데코레이터
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                log_performance(func.__name__, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger = logging.getLogger(logger_name)
                logger.error(
                    f"함수 실행 중 오류 발생: {func.__name__} (실행시간: {duration:.2f}초)",
                    extra={
                        "function": func.__name__,
                        "duration_ms": round(duration * 1000, 2),
                        "error": str(e),
                        "performance": True
                    },
                    exc_info=True
                )
                raise
        return wrapper
    return decorator
