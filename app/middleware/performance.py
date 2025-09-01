"""
성능 모니터링 미들웨어

API 요청의 응답 시간, 메모리 사용량, 데이터베이스 쿼리 성능을 측정하고 로깅합니다.
"""

import time
import logging
import psutil
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
import pymongo.monitoring

# 성능 로거 설정
perf_logger = logging.getLogger("performance")
perf_logger.setLevel(logging.INFO)

# 파일 핸들러 추가
if not perf_logger.handlers:
    handler = logging.FileHandler("performance.log")
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    perf_logger.addHandler(handler)

class MongoPerformanceMonitor(pymongo.monitoring.CommandListener):
    """MongoDB 쿼리 성능 모니터링"""
    
    def started(self, event):
        """쿼리 시작 시 호출"""
        self.start_time = time.time()
        perf_logger.info(f"MongoDB 쿼리 시작: {event.command_name} - {event.database_name}")
    
    def succeeded(self, event):
        """쿼리 성공 시 호출"""
        duration = time.time() - self.start_time
        perf_logger.info(
            f"MongoDB 쿼리 완료: {event.command_name} - "
            f"소요시간: {duration:.3f}s - "
            f"DB: {event.database_name}"
        )
        
        # 느린 쿼리 경고 (100ms 이상)
        if duration > 0.1:
            perf_logger.warning(
                f"느린 MongoDB 쿼리 감지: {event.command_name} - "
                f"소요시간: {duration:.3f}s - "
                f"Request ID: {event.request_id}"
            )
    
    def failed(self, event):
        """쿼리 실패 시 호출"""
        duration = time.time() - self.start_time
        perf_logger.error(
            f"MongoDB 쿼리 실패: {event.command_name} - "
            f"소요시간: {duration:.3f}s - "
            f"오류: {event.failure}"
        )

# MongoDB 모니터 인스턴스
mongo_monitor = MongoPerformanceMonitor()

def setup_mongo_monitoring():
    """MongoDB 성능 모니터링 설정"""
    pymongo.monitoring.register(mongo_monitor)
    perf_logger.info("MongoDB 성능 모니터링 활성화")

async def performance_middleware(request: Request, call_next: Callable) -> Response:
    """
    성능 측정 미들웨어
    
    각 API 요청의 처리 시간, 메모리 사용량을 측정하고 로깅합니다.
    """
    
    # 요청 시작 시점 기록
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    # 요청 정보 로깅
    perf_logger.info(
        f"요청 시작 - Method: {request.method} - "
        f"URL: {request.url.path} - "
        f"Client: {request.client.host if request.client else 'unknown'}"
    )
    
    try:
        # 다음 미들웨어/핸들러 실행
        response = await call_next(request)
        
        # 처리 완료 후 성능 지표 계산
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        duration = end_time - start_time
        memory_diff = end_memory - start_memory
        
        # 응답 크기 계산(가능한 경우)
        resp_size = None
        try:
            cl = response.headers.get("content-length")
            if cl is not None:
                resp_size = int(cl)
            elif hasattr(response, "body") and isinstance(response.body, (bytes, bytearray)):
                resp_size = len(response.body)
        except Exception:
            resp_size = None

        # 성능 로그 기록
        perf_logger.info(
            f"요청 완료 - Method: {request.method} - "
            f"URL: {request.url.path} - "
            f"상태: {response.status_code} - "
            f"소요시간: {duration:.3f}s - "
            f"메모리 변화: {memory_diff:+.2f}MB - "
            f"현재 메모리: {end_memory:.1f}MB - "
            f"응답 크기: {resp_size if resp_size is not None else 'unknown'}B"
        )
        
        # 성능 헤더 추가
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        response.headers["X-Memory-Usage"] = f"{end_memory:.1f}MB"
        
        # 느린 요청 경고 (1초 이상)
        if duration > 1.0:
            perf_logger.warning(
                f"느린 API 요청 감지 - URL: {request.url.path} - "
                f"소요시간: {duration:.3f}s - "
                f"최적화 필요"
            )
        
        # 메모리 누수 경고 (10MB 이상 증가)
        if memory_diff > 10:
            perf_logger.warning(
                f"높은 메모리 사용 감지 - URL: {request.url.path} - "
                f"메모리 증가: {memory_diff:.2f}MB - "
                f"메모리 누수 의심"
            )
        
        # 큰 응답 경고(2MB 이상)
        if resp_size is not None and resp_size > 2 * 1024 * 1024:
            perf_logger.warning(
                f"큰 응답 감지 - URL: {request.url.path} - size={resp_size}B"
            )

        return response
        
    except Exception as e:
        # 오류 발생 시에도 성능 지표 기록
        end_time = time.time()
        duration = end_time - start_time
        
        perf_logger.error(
            f"요청 오류 - Method: {request.method} - "
            f"URL: {request.url.path} - "
            f"소요시간: {duration:.3f}s - "
            f"오류: {str(e)}"
        )
        
        # 오류 응답 반환
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "duration": f"{duration:.3f}s"}
        )

def get_performance_stats():
    """현재 성능 통계 반환"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        "memory": {
            "rss": memory_info.rss / 1024 / 1024,  # MB
            "vms": memory_info.vms / 1024 / 1024,  # MB
        },
        "cpu_percent": process.cpu_percent(),
        "num_threads": process.num_threads(),
        "create_time": process.create_time()
    }
