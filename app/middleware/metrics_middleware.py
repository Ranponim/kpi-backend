"""
메트릭 수집 미들웨어

모든 API 요청에 대해 성능 메트릭을 수집하고 분석합니다.
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ..utils.metrics_collector import get_metrics_collector

logger = logging.getLogger("app.metrics_middleware")


class MetricsCollectionMiddleware(BaseHTTPMiddleware):
    """메트릭 수집 미들웨어"""
    
    def __init__(self, app, collect_request_body: bool = False, 
                 collect_response_body: bool = False):
        super().__init__(app)
        self.collect_request_body = collect_request_body
        self.collect_response_body = collect_response_body
        self.metrics_collector = get_metrics_collector()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 요청 시작 시간 기록
        start_time = time.time()
        
        # 요청 정보 수집
        endpoint = self._get_endpoint_path(request)
        method = request.method
        user_agent = request.headers.get("user-agent", "")
        remote_addr = self._get_client_ip(request)
        
        # 요청 크기 계산
        request_size = self._calculate_request_size(request)
        
        # 응답 처리
        response = await call_next(request)
        
        # 응답 시간 계산
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        # 응답 크기 계산
        response_size = self._calculate_response_size(response)
        
        # 메트릭 기록
        self.metrics_collector.record_api_call(
            endpoint=endpoint,
            method=method,
            status_code=response.status_code,
            duration_ms=duration_ms,
            request_size=request_size,
            response_size=response_size,
            user_agent=user_agent,
            remote_addr=remote_addr
        )
        
        # 비즈니스 이벤트 기록
        self._record_business_events(request, response)
        
        # 성능 로깅 (느린 요청)
        if duration_ms > 1000:  # 1초 이상
            logger.warning(f"느린 API 요청: {method} {endpoint} - {duration_ms:.0f}ms", extra={
                "endpoint": endpoint,
                "method": method,
                "duration_ms": duration_ms,
                "status_code": response.status_code,
                "remote_addr": remote_addr
            })
        
        return response
    
    def _get_endpoint_path(self, request: Request) -> str:
        """엔드포인트 경로 추출 (파라미터 정규화)"""
        path = request.url.path
        
        # 공통 파라미터 패턴을 정규화
        # 예: /api/analysis/results/123 -> /api/analysis/results/{id}
        import re
        
        # ObjectId 패턴 (24자리 hex)
        path = re.sub(r'/[0-9a-f]{24}', '/{id}', path)
        
        # UUID 패턴
        path = re.sub(r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '/{uuid}', path)
        
        # 숫자 ID 패턴
        path = re.sub(r'/\d+', '/{id}', path)
        
        return path
    
    def _get_client_ip(self, request: Request) -> str:
        """클라이언트 IP 추출 (프록시 고려)"""
        # X-Forwarded-For 헤더 확인 (프록시 환경)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # 여러 IP가 있을 경우 첫 번째 IP 사용
            return forwarded_for.split(",")[0].strip()
        
        # X-Real-IP 헤더 확인
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # 직접 연결된 클라이언트 IP
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _calculate_request_size(self, request: Request) -> int:
        """요청 크기 계산"""
        try:
            # Content-Length 헤더 확인
            content_length = request.headers.get("content-length")
            if content_length:
                return int(content_length)
            
            # 헤더 크기 추정
            header_size = sum(len(k) + len(v) + 4 for k, v in request.headers.items())  # +4 for ": \r\n"
            return header_size
            
        except (ValueError, TypeError):
            return 0
    
    def _calculate_response_size(self, response: Response) -> int:
        """응답 크기 계산"""
        try:
            # Content-Length 헤더 확인
            content_length = response.headers.get("content-length")
            if content_length:
                return int(content_length)
            
            # 헤더 크기 추정
            header_size = sum(len(k) + len(v) + 4 for k, v in response.headers.items())
            return header_size
            
        except (ValueError, TypeError):
            return 0
    
    def _record_business_events(self, request: Request, response: Response):
        """비즈니스 이벤트 기록"""
        path = request.url.path
        method = request.method
        status_code = response.status_code
        
        # 성공적인 요청만 비즈니스 이벤트로 기록
        if status_code < 400:
            # 분석 결과 생성
            if method == "POST" and "/api/analysis/results" in path:
                self.metrics_collector.record_business_event("analysis_created")
            
            # 분석 결과 조회
            elif method == "GET" and "/api/analysis/results" in path:
                if "/{id}" in path or path.endswith("/api/analysis/results"):
                    self.metrics_collector.record_business_event("analysis_viewed")
            
            # 통계 조회
            elif method == "GET" and "/stats" in path:
                self.metrics_collector.record_business_event("stats_viewed")
            
            # KPI 조회
            elif method == "GET" and "/api/kpi" in path:
                self.metrics_collector.record_business_event("kpi_viewed")
        
        # 캐시 관련 이벤트 (응답 헤더에서 확인)
        cache_status = response.headers.get("x-cache-status")
        if cache_status == "hit":
            self.metrics_collector.record_business_event("cache_hits")
        elif cache_status == "miss":
            self.metrics_collector.record_business_event("cache_misses")
    
    @staticmethod
    def add_cache_headers(response: Response, cache_hit: bool):
        """응답에 캐시 상태 헤더 추가"""
        response.headers["x-cache-status"] = "hit" if cache_hit else "miss"
        return response
