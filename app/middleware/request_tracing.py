"""
요청 추적 미들웨어

각 HTTP 요청에 고유 ID를 할당하고, 요청의 전체 라이프사이클을 추적합니다.
분산 시스템에서 요청을 추적하고 디버깅을 용이하게 합니다.
"""

import time
import uuid
import logging
from typing import Callable, Dict, Any, Optional
from fastapi import Request, Response
from contextvars import ContextVar
import json

from ..utils.logging_config import get_request_logger

# 요청 컨텍스트 변수
request_id_context: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
request_start_time_context: ContextVar[Optional[float]] = ContextVar('request_start_time', default=None)

# 메인 로거
logger = logging.getLogger("app.request_tracing")


class RequestTracingMiddleware:
    """
    요청 추적 미들웨어
    
    각 요청에 대해:
    1. 고유 ID 생성 (X-Request-ID 헤더 또는 새로 생성)
    2. 요청 시작/완료 로깅
    3. 요청 컨텍스트 설정
    4. 성능 메트릭 수집
    """
    
    def __init__(self, 
                 log_requests: bool = True,
                 log_responses: bool = True,
                 log_headers: bool = False,
                 log_body: bool = False,
                 max_body_size: int = 1024):
        """
        미들웨어 초기화
        
        Args:
            log_requests: 요청 정보 로깅 여부
            log_responses: 응답 정보 로깅 여부  
            log_headers: 헤더 정보 로깅 여부
            log_body: 요청/응답 바디 로깅 여부
            max_body_size: 로깅할 최대 바디 크기 (바이트)
        """
        self.log_requests = log_requests
        self.log_responses = log_responses
        self.log_headers = log_headers
        self.log_body = log_body
        self.max_body_size = max_body_size
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """미들웨어 실행"""
        
        # 1. 요청 ID 생성 또는 추출
        request_id = self._get_or_create_request_id(request)
        
        # 2. 컨텍스트 설정
        request_id_context.set(request_id)
        start_time = time.time()
        request_start_time_context.set(start_time)
        
        # 3. 요청별 로거 생성
        req_logger = get_request_logger(request_id)
        
        try:
            # 4. 요청 정보 로깅
            if self.log_requests:
                await self._log_request(req_logger, request)
            
            # 5. 다음 미들웨어/핸들러 실행
            response = await call_next(request)
            
            # 6. 응답 처리
            response.headers["X-Request-ID"] = request_id
            
            # 7. 응답 정보 로깅
            if self.log_responses:
                self._log_response(req_logger, request, response, start_time)
            
            return response
            
        except Exception as e:
            # 8. 예외 처리 및 로깅
            duration = time.time() - start_time
            req_logger.error(
                f"요청 처리 중 예외 발생: {str(e)}",
                extra={
                    "method": request.method,
                    "url": str(request.url),
                    "duration_ms": round(duration * 1000, 2),
                    "exception_type": type(e).__name__,
                    "status": "exception"
                },
                exc_info=True
            )
            raise
    
    def _get_or_create_request_id(self, request: Request) -> str:
        """요청 ID를 가져오거나 새로 생성합니다."""
        
        # 헤더에서 기존 요청 ID 확인
        request_id = request.headers.get("X-Request-ID")
        
        if not request_id:
            # 새로운 요청 ID 생성 (타임스탬프 + UUID)
            timestamp = int(time.time() * 1000)  # 밀리초
            short_uuid = str(uuid.uuid4())[:8]
            request_id = f"req_{timestamp}_{short_uuid}"
        
        return request_id
    
    async def _log_request(self, req_logger: logging.LoggerAdapter, request: Request) -> None:
        """요청 정보를 로깅합니다."""
        
        # 기본 요청 정보
        request_info = {
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_host": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "content_type": request.headers.get("content-type"),
            "content_length": request.headers.get("content-length"),
            "event": "request_started"
        }
        
        # 헤더 정보 포함 (선택적)
        if self.log_headers:
            request_info["headers"] = dict(request.headers)
        
        # 요청 바디 로깅 (선택적, 작은 크기만)
        if self.log_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if len(body) <= self.max_body_size:
                    content_type = request.headers.get("content-type", "")
                    if "application/json" in content_type:
                        try:
                            request_info["body"] = json.loads(body.decode("utf-8"))
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            request_info["body"] = f"<binary data: {len(body)} bytes>"
                    else:
                        request_info["body"] = f"<{content_type}: {len(body)} bytes>"
                else:
                    request_info["body"] = f"<large body: {len(body)} bytes>"
            except Exception:
                request_info["body"] = "<body read error>"
        
        req_logger.info("요청 시작", extra=request_info)
    
    def _log_response(self, 
                     req_logger: logging.LoggerAdapter, 
                     request: Request, 
                     response: Response, 
                     start_time: float) -> None:
        """응답 정보를 로깅합니다."""
        
        duration = time.time() - start_time
        
        # 기본 응답 정보
        response_info = {
            "method": request.method,
            "url": str(request.url),
            "status_code": response.status_code,
            "duration_ms": round(duration * 1000, 2),
            "content_type": response.headers.get("content-type"),
            "content_length": response.headers.get("content-length"),
            "event": "request_completed"
        }
        
        # 헤더 정보 포함 (선택적)
        if self.log_headers:
            response_info["response_headers"] = dict(response.headers)
        
        # 응답 바디 로깅 (선택적, 작은 크기만)
        if self.log_body and hasattr(response, "body"):
            try:
                if isinstance(response.body, (bytes, bytearray)):
                    body_size = len(response.body)
                    if body_size <= self.max_body_size:
                        content_type = response.headers.get("content-type", "")
                        if "application/json" in content_type:
                            try:
                                response_info["response_body"] = json.loads(response.body.decode("utf-8"))
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                response_info["response_body"] = f"<binary data: {body_size} bytes>"
                        else:
                            response_info["response_body"] = f"<{content_type}: {body_size} bytes>"
                    else:
                        response_info["response_body"] = f"<large body: {body_size} bytes>"
            except Exception:
                response_info["response_body"] = "<body read error>"
        
        # 로그 레벨 결정 (상태 코드와 응답 시간 기반)
        if response.status_code >= 500:
            req_logger.error("요청 완료 (서버 오류)", extra=response_info)
        elif response.status_code >= 400:
            req_logger.warning("요청 완료 (클라이언트 오류)", extra=response_info)
        elif duration > 2.0:  # 2초 이상
            req_logger.warning("요청 완료 (느린 응답)", extra=response_info)
        else:
            req_logger.info("요청 완료", extra=response_info)


def get_current_request_id() -> Optional[str]:
    """현재 요청의 ID를 반환합니다."""
    return request_id_context.get()


def get_current_request_duration() -> Optional[float]:
    """현재 요청의 진행 시간을 반환합니다."""
    start_time = request_start_time_context.get()
    if start_time:
        return time.time() - start_time
    return None


def create_request_context_logger(name: str) -> logging.LoggerAdapter:
    """
    현재 요청 컨텍스트가 포함된 로거를 생성합니다.
    
    Args:
        name: 로거 이름
        
    Returns:
        요청 컨텍스트가 포함된 LoggerAdapter
    """
    logger = logging.getLogger(name)
    
    extra = {}
    request_id = get_current_request_id()
    if request_id:
        extra["request_id"] = request_id
    
    duration = get_current_request_duration()
    if duration:
        extra["request_duration_ms"] = round(duration * 1000, 2)
    
    return logging.LoggerAdapter(logger, extra)


# 편의 함수들
def log_business_event(event_name: str, details: Dict[str, Any] = None) -> None:
    """
    비즈니스 이벤트를 로깅합니다.
    
    Args:
        event_name: 이벤트 이름
        details: 추가 세부 정보
    """
    logger = create_request_context_logger("app.business")
    
    event_info = {
        "event": event_name,
        "event_type": "business",
        **(details or {})
    }
    
    logger.info(f"비즈니스 이벤트: {event_name}", extra=event_info)


def log_security_event(event_name: str, details: Dict[str, Any] = None) -> None:
    """
    보안 이벤트를 로깅합니다.
    
    Args:
        event_name: 이벤트 이름
        details: 추가 세부 정보
    """
    logger = create_request_context_logger("app.security")
    
    event_info = {
        "event": event_name,
        "event_type": "security",
        **(details or {})
    }
    
    logger.warning(f"보안 이벤트: {event_name}", extra=event_info)


def log_data_event(event_name: str, details: Dict[str, Any] = None) -> None:
    """
    데이터 관련 이벤트를 로깅합니다.
    
    Args:
        event_name: 이벤트 이름  
        details: 추가 세부 정보
    """
    logger = create_request_context_logger("app.data")
    
    event_info = {
        "event": event_name,
        "event_type": "data",
        **(details or {})
    }
    
    logger.info(f"데이터 이벤트: {event_name}", extra=event_info)
