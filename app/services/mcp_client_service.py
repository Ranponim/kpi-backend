"""
MCP 서버 통신 서비스

이 모듈은 MCP 서버와의 통신을 담당하는 서비스입니다.
PEG 비교분석 요청을 MCP 서버로 전달하고 응답을 처리합니다.
"""

import logging
import asyncio
import aiohttp
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json

from ..models.peg_comparison import MCPRequest, MCPResponse
from ..exceptions.peg_comparison_exceptions import (
    MCPConnectionError,
    MCPTimeoutError,
    DataValidationError
)

logger = logging.getLogger("app.services.mcp_client")


class MCPClientService:
    """MCP 서버 통신 서비스 클래스"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8001",  # MCP 서버 기본 URL
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        MCP 클라이언트 서비스 초기화
        
        Args:
            base_url: MCP 서버 기본 URL
            timeout: 요청 타임아웃 (초)
            max_retries: 최대 재시도 횟수
            retry_delay: 재시도 간격 (초)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # 연결 풀 설정 (지연 초기화)
        self.connector = None
        
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info(f"MCP 클라이언트 서비스 초기화 완료: {self.base_url}")
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        await self.close()
    
    async def _ensure_session(self):
        """세션이 없으면 생성"""
        if self.session is None or self.session.closed:
            # connector가 없으면 생성
            if self.connector is None:
                self.connector = aiohttp.TCPConnector(
                    limit=100,  # 전체 연결 수 제한
                    limit_per_host=30,  # 호스트당 연결 수 제한
                    keepalive_timeout=30,  # Keep-alive 타임아웃
                    enable_cleanup_closed=True
                )
            
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'KPI-Dashboard-Backend/1.0.0'
                }
            )
            logger.debug("MCP 클라이언트 세션 생성")
    
    async def close(self):
        """세션 종료"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.debug("MCP 클라이언트 세션 종료")
        
        if self.connector:
            await self.connector.close()
            self.connector = None
            logger.debug("MCP 클라이언트 connector 종료")
    
    async def call_mcp(
        self, 
        endpoint: str,
        data: Dict[str, Any],
        method: str = "POST",
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        일반적인 MCP 서버 호출 메서드
        
        Args:
            endpoint: MCP 서버 엔드포인트
            data: 요청 데이터
            method: HTTP 메서드 (GET, POST, PUT, DELETE)
            timeout: 요청 타임아웃 (초)
            
        Returns:
            Dict[str, Any]: MCP 서버 응답
            
        Raises:
            MCPConnectionError: MCP 서버 연결 실패
            MCPTimeoutError: MCP 서버 응답 타임아웃
        """
        await self._ensure_session()
        
        url = f"{self.base_url}{endpoint}"
        request_timeout = timeout or self.timeout
        
        logger.info(f"MCP 서버 호출 시작: {method} {endpoint}", extra={
            "endpoint": endpoint,
            "method": method,
            "url": url,
            "data_size": len(json.dumps(data)) if data else 0
        })
        
        start_time = datetime.utcnow()
        
        # 재시도 로직
        for attempt in range(1, self.max_retries + 1):
            try:
                if method.upper() == "GET":
                    async with self.session.get(
                        url,
                        params=data,
                        timeout=aiohttp.ClientTimeout(total=request_timeout)
                    ) as response:
                        return await self._handle_response(response, endpoint, attempt, start_time)
                
                elif method.upper() == "POST":
                    async with self.session.post(
                        url,
                        json=data,
                        timeout=aiohttp.ClientTimeout(total=request_timeout)
                    ) as response:
                        return await self._handle_response(response, endpoint, attempt, start_time)
                
                elif method.upper() == "PUT":
                    async with self.session.put(
                        url,
                        json=data,
                        timeout=aiohttp.ClientTimeout(total=request_timeout)
                    ) as response:
                        return await self._handle_response(response, endpoint, attempt, start_time)
                
                elif method.upper() == "DELETE":
                    async with self.session.delete(
                        url,
                        json=data,
                        timeout=aiohttp.ClientTimeout(total=request_timeout)
                    ) as response:
                        return await self._handle_response(response, endpoint, attempt, start_time)
                
                else:
                    raise ValueError(f"지원하지 않는 HTTP 메서드: {method}")
            
            except asyncio.TimeoutError:
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                
                logger.warning(f"MCP 서버 응답 타임아웃: {endpoint}", extra={
                    "endpoint": endpoint,
                    "timeout": request_timeout,
                    "attempt": attempt,
                    "processing_time": processing_time
                })
                
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * attempt)
                    continue
                else:
                    raise MCPTimeoutError(
                        message=f"MCP 서버 응답 타임아웃 ({request_timeout}초)",
                        timeout_seconds=request_timeout,
                        details={"attempts": attempt, "processing_time": processing_time}
                    )
            
            except aiohttp.ClientError as e:
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                
                logger.error(f"MCP 서버 연결 오류: {endpoint}", extra={
                    "endpoint": endpoint,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "attempt": attempt,
                    "processing_time": processing_time
                })
                
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * attempt)
                    continue
                else:
                    raise MCPConnectionError(
                        message=f"MCP 서버 연결 실패: {str(e)}",
                        details={"error_type": type(e).__name__, "attempts": attempt}
                    )
            
            except Exception as e:
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                
                logger.error(f"MCP 서버 호출 중 예상치 못한 오류: {endpoint}", extra={
                    "endpoint": endpoint,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "attempt": attempt,
                    "processing_time": processing_time
                }, exc_info=True)
                
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * attempt)
                    continue
                else:
                    raise MCPConnectionError(
                        message=f"예상치 못한 오류: {str(e)}",
                        details={"error_type": type(e).__name__, "attempts": attempt}
                    )
        
        # 이 지점에 도달하면 안 됨
        raise MCPConnectionError(
            message="최대 재시도 횟수 초과",
            details={"max_retries": self.max_retries}
        )
    
    async def _handle_response(
        self, 
        response: aiohttp.ClientResponse, 
        endpoint: str, 
        attempt: int, 
        start_time: datetime
    ) -> Dict[str, Any]:
        """응답 처리 헬퍼 메서드"""
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        if response.status == 200:
            response_data = await response.json()
            
            logger.info(f"MCP 서버 호출 성공: {endpoint}", extra={
                "endpoint": endpoint,
                "attempt": attempt,
                "processing_time": processing_time,
                "response_size": len(json.dumps(response_data))
            })
            
            return response_data
        
        elif response.status == 400:
            error_data = await response.json()
            error_message = error_data.get("error", {}).get("message", "잘못된 요청")
            
            logger.warning(f"MCP 서버 요청 오류: {endpoint}", extra={
                "endpoint": endpoint,
                "status_code": response.status,
                "error_message": error_message
            })
            
            raise DataValidationError(
                message=error_message,
                validation_errors=error_data.get("error", {}).get("details", {})
            )
        
        elif response.status == 503:
            error_message = "MCP 서버가 일시적으로 사용할 수 없습니다"
            
            logger.warning(f"MCP 서버 일시적 오류: {endpoint}", extra={
                "endpoint": endpoint,
                "status_code": response.status,
                "attempt": attempt
            })
            
            raise MCPConnectionError(
                message=error_message,
                details={"status_code": response.status, "attempts": attempt}
            )
        
        else:
            error_message = f"MCP 서버 응답 오류: {response.status}"
            
            logger.error(f"MCP 서버 응답 오류: {endpoint}", extra={
                "endpoint": endpoint,
                "status_code": response.status,
                "attempt": attempt
            })
            
            raise MCPConnectionError(
                message=error_message,
                details={"status_code": response.status, "attempts": attempt}
            )

    async def call_peg_comparison(
        self, 
        analysis_id: str, 
        raw_data: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None
    ) -> MCPResponse:
        """
        PEG 비교분석을 MCP 서버에 요청
        
        Args:
            analysis_id: 분석 ID
            raw_data: 원시 데이터
            options: 추가 옵션
            
        Returns:
            MCPResponse: MCP 서버 응답
            
        Raises:
            MCPConnectionError: MCP 서버 연결 실패
            MCPTimeoutError: MCP 서버 응답 타임아웃
            DataValidationError: 데이터 검증 실패
        """
        await self._ensure_session()
        
        # 요청 데이터 구성
        request_data = MCPRequest(
            analysis_id=analysis_id,
            raw_data=raw_data,
            options=options or {}
        )
        
        url = f"{self.base_url}/mcp/peg-comparison/analysis"
        
        logger.info(f"PEG 비교분석 요청 시작: {analysis_id}", extra={
            "analysis_id": analysis_id,
            "url": url,
            "data_size": len(json.dumps(raw_data))
        })
        
        start_time = datetime.utcnow()
        
        # 재시도 로직
        for attempt in range(1, self.max_retries + 1):
            try:
                async with self.session.post(
                    url,
                    json=request_data.model_dump(),
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    
                    processing_time = (datetime.utcnow() - start_time).total_seconds()
                    
                    if response.status == 200:
                        response_data = await response.json()
                        
                        logger.info(f"PEG 비교분석 요청 성공: {analysis_id}", extra={
                            "analysis_id": analysis_id,
                            "attempt": attempt,
                            "processing_time": processing_time,
                            "response_size": len(json.dumps(response_data))
                        })
                        
                        return MCPResponse(
                            success=True,
                            data=response_data,
                            processing_time=processing_time,
                            algorithm_version=response_data.get("algorithm_version", "v2.1.0")
                        )
                    
                    elif response.status == 400:
                        error_data = await response.json()
                        error_message = error_data.get("error", {}).get("message", "데이터 검증 실패")
                        
                        logger.warning(f"PEG 비교분석 데이터 검증 실패: {analysis_id}", extra={
                            "analysis_id": analysis_id,
                            "status_code": response.status,
                            "error_message": error_message
                        })
                        
                        raise DataValidationError(
                            message=error_message,
                            validation_errors=error_data.get("error", {}).get("details", {})
                        )
                    
                    elif response.status == 503:
                        error_message = "MCP 서버가 일시적으로 사용할 수 없습니다"
                        
                        logger.warning(f"MCP 서버 일시적 오류: {analysis_id}", extra={
                            "analysis_id": analysis_id,
                            "status_code": response.status,
                            "attempt": attempt
                        })
                        
                        if attempt < self.max_retries:
                            await asyncio.sleep(self.retry_delay * attempt)
                            continue
                        else:
                            raise MCPConnectionError(
                                message=error_message,
                                details={"status_code": response.status, "attempts": attempt}
                            )
                    
                    else:
                        error_message = f"MCP 서버 응답 오류: {response.status}"
                        
                        logger.error(f"MCP 서버 응답 오류: {analysis_id}", extra={
                            "analysis_id": analysis_id,
                            "status_code": response.status,
                            "attempt": attempt
                        })
                        
                        raise MCPConnectionError(
                            message=error_message,
                            details={"status_code": response.status, "attempts": attempt}
                        )
            
            except asyncio.TimeoutError:
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                
                logger.warning(f"MCP 서버 응답 타임아웃: {analysis_id}", extra={
                    "analysis_id": analysis_id,
                    "timeout": self.timeout,
                    "attempt": attempt,
                    "processing_time": processing_time
                })
                
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * attempt)
                    continue
                else:
                    raise MCPTimeoutError(
                        message=f"MCP 서버 응답 타임아웃 ({self.timeout}초)",
                        timeout_seconds=self.timeout,
                        details={"attempts": attempt, "processing_time": processing_time}
                    )
            
            except aiohttp.ClientError as e:
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                
                logger.error(f"MCP 서버 연결 오류: {analysis_id}", extra={
                    "analysis_id": analysis_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "attempt": attempt,
                    "processing_time": processing_time
                })
                
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * attempt)
                    continue
                else:
                    raise MCPConnectionError(
                        message=f"MCP 서버 연결 실패: {str(e)}",
                        details={"error_type": type(e).__name__, "attempts": attempt}
                    )
            
            except Exception as e:
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                
                logger.error(f"PEG 비교분석 요청 중 예상치 못한 오류: {analysis_id}", extra={
                    "analysis_id": analysis_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "attempt": attempt,
                    "processing_time": processing_time
                }, exc_info=True)
                
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * attempt)
                    continue
                else:
                    raise MCPConnectionError(
                        message=f"예상치 못한 오류: {str(e)}",
                        details={"error_type": type(e).__name__, "attempts": attempt}
                    )
        
        # 이 지점에 도달하면 안 됨
        raise MCPConnectionError(
            message="최대 재시도 횟수 초과",
            details={"max_retries": self.max_retries}
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        MCP 서버 상태 확인
        
        Returns:
            Dict[str, Any]: 서버 상태 정보
        """
        await self._ensure_session()
        
        try:
            url = f"{self.base_url}/health"
            
            async with self.session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=5.0)
            ) as response:
                
                if response.status == 200:
                    health_data = await response.json()
                    
                    logger.debug("MCP 서버 상태 확인 성공", extra={
                        "status": health_data.get("status"),
                        "version": health_data.get("version")
                    })
                    
                    return {
                        "status": "healthy",
                        "response_time": response.headers.get("X-Response-Time", "unknown"),
                        "server_info": health_data
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "status_code": response.status,
                        "error": f"HTTP {response.status}"
                    }
        
        except asyncio.TimeoutError:
            logger.warning("MCP 서버 상태 확인 타임아웃")
            return {
                "status": "timeout",
                "error": "Health check timeout"
            }
        
        except Exception as e:
            logger.error(f"MCP 서버 상태 확인 실패: {e}", extra={
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def get_supported_versions(self) -> Dict[str, Any]:
        """
        지원되는 알고리즘 버전 조회
        
        Returns:
            Dict[str, Any]: 지원되는 버전 정보
        """
        await self._ensure_session()
        
        try:
            url = f"{self.base_url}/mcp/peg-comparison/versions"
            
            async with self.session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=10.0)
            ) as response:
                
                if response.status == 200:
                    versions_data = await response.json()
                    
                    logger.debug("지원 버전 조회 성공", extra={
                        "supported_versions": versions_data.get("supported_versions", [])
                    })
                    
                    return versions_data
                else:
                    logger.warning(f"지원 버전 조회 실패: {response.status}")
                    return {
                        "supported_versions": ["v2.1.0"],  # 기본값
                        "default_version": "v2.1.0"
                    }
        
        except Exception as e:
            logger.warning(f"지원 버전 조회 중 오류: {e}", extra={
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            return {
                "supported_versions": ["v2.1.0"],  # 기본값
                "default_version": "v2.1.0"
            }


# 전역 MCP 클라이언트 인스턴스
_mcp_client: Optional[MCPClientService] = None


async def get_mcp_client() -> MCPClientService:
    """
    전역 MCP 클라이언트 인스턴스 반환
    
    Returns:
        MCPClientService: MCP 클라이언트 서비스 인스턴스
    """
    global _mcp_client
    
    if _mcp_client is None:
        # 환경 변수에서 설정 읽기 (기본값 사용)
        import os
        mcp_base_url = os.getenv("MCP_SERVER_URL", "http://localhost:8001")
        mcp_timeout = float(os.getenv("MCP_TIMEOUT", "30.0"))
        mcp_max_retries = int(os.getenv("MCP_MAX_RETRIES", "3"))
        
        _mcp_client = MCPClientService(
            base_url=mcp_base_url,
            timeout=mcp_timeout,
            max_retries=mcp_max_retries
        )
        
        logger.info(f"전역 MCP 클라이언트 생성: {mcp_base_url}")
    
    return _mcp_client


async def close_mcp_client():
    """전역 MCP 클라이언트 종료"""
    global _mcp_client
    
    if _mcp_client is not None:
        await _mcp_client.close()
        _mcp_client = None
        logger.info("전역 MCP 클라이언트 종료")
