"""
고성능 캐싱 관리자

Redis를 사용한 다층 캐싱 시스템으로 API 응답 시간을 극적으로 개선합니다.
"""

import json
import logging
import hashlib
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
import asyncio
import pickle
import zlib

try:
    import redis.asyncio as redis
    from redis.asyncio import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    Redis = None

from ..middleware.request_tracing import create_request_context_logger

logger = logging.getLogger("app.cache_manager")

# 캐시 설정
DEFAULT_TTL = 300  # 5분
ANALYSIS_RESULTS_TTL = 3600  # 1시간
STATISTICS_TTL = 1800  # 30분
KPI_DATA_TTL = 900  # 15분
OPTIMIZATION_STATS_TTL = 600  # 10분

# 압축 임계값 (1KB 이상)
COMPRESSION_THRESHOLD = 1024


class CacheManager:
    """고성능 다층 캐싱 관리자"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", enable_compression: bool = True):
        self.redis_url = redis_url
        self.enable_compression = enable_compression
        self.redis_client: Optional[Redis] = None
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._memory_cache_size = 1000  # 메모리 캐시 최대 항목 수
        self._initialized = False
        
    async def initialize(self):
        """Redis 연결 초기화"""
        if self._initialized:
            return
            
        if not REDIS_AVAILABLE:
            logger.warning("Redis 라이브러리를 찾을 수 없습니다. 메모리 캐시만 사용합니다.")
            self._initialized = True
            return
            
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,  # 바이너리 데이터 지원을 위해
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # 연결 테스트
            await self.redis_client.ping()
            logger.info("Redis 연결 성공")
            
        except Exception as e:
            logger.warning(f"Redis 연결 실패, 메모리 캐시만 사용: {e}")
            self.redis_client = None
            
        self._initialized = True
    
    async def close(self):
        """연결 종료"""
        if self.redis_client:
            await self.redis_client.close()
    
    def _generate_cache_key(self, prefix: str, **kwargs) -> str:
        """캐시 키 생성"""
        # 키 구성 요소들을 정렬하여 일관성 보장
        key_parts = [prefix]
        for key, value in sorted(kwargs.items()):
            if value is not None:
                key_parts.append(f"{key}={value}")
        
        key_string = ":".join(key_parts)
        
        # 긴 키는 해시로 단축
        if len(key_string) > 200:
            hash_suffix = hashlib.md5(key_string.encode()).hexdigest()[:12]
            key_string = f"{prefix}:hash:{hash_suffix}"
        
        return key_string
    
    def _serialize_data(self, data: Any) -> bytes:
        """데이터 직렬화 및 압축"""
        try:
            # JSON 시도
            serialized = json.dumps(data, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        except (TypeError, ValueError):
            # JSON 실패 시 pickle 사용
            serialized = pickle.dumps(data)
        
        # 압축 적용
        if self.enable_compression and len(serialized) > COMPRESSION_THRESHOLD:
            compressed = zlib.compress(serialized, level=6)
            # 압축 효과가 있는 경우만 사용
            if len(compressed) < len(serialized) * 0.9:
                return b'compressed:' + compressed
        
        return b'raw:' + serialized
    
    def _deserialize_data(self, data: bytes) -> Any:
        """데이터 역직렬화 및 압축 해제"""
        if data.startswith(b'compressed:'):
            # 압축 해제
            compressed_data = data[11:]  # 'compressed:' 제거
            decompressed = zlib.decompress(compressed_data)
            
            # JSON 먼저 시도
            try:
                return json.loads(decompressed.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return pickle.loads(decompressed)
                
        elif data.startswith(b'raw:'):
            # 원본 데이터
            raw_data = data[4:]  # 'raw:' 제거
            
            # JSON 먼저 시도
            try:
                return json.loads(raw_data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return pickle.loads(raw_data)
        
        # 레거시 데이터 처리
        try:
            return json.loads(data.decode('utf-8'))
        except:
            return pickle.loads(data)
    
    async def get(self, key: str) -> Optional[Any]:
        """캐시에서 데이터 조회"""
        await self.initialize()
        
        req_logger = create_request_context_logger("app.cache.get")
        
        # 1. 메모리 캐시 확인 (L1)
        if key in self._memory_cache:
            cache_entry = self._memory_cache[key]
            if cache_entry['expires_at'] > datetime.now():
                req_logger.debug(f"메모리 캐시 히트: {key}")
                return cache_entry['data']
            else:
                # 만료된 항목 제거
                del self._memory_cache[key]
        
        # 2. Redis 캐시 확인 (L2)
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(key)
                if cached_data:
                    data = self._deserialize_data(cached_data)
                    
                    # 메모리 캐시에도 저장 (작은 데이터만)
                    if len(cached_data) < 10240:  # 10KB 미만
                        self._set_memory_cache(key, data, DEFAULT_TTL)
                    
                    req_logger.debug(f"Redis 캐시 히트: {key}")
                    return data
                    
            except Exception as e:
                req_logger.warning(f"Redis 조회 실패: {e}")
        
        req_logger.debug(f"캐시 미스: {key}")
        return None
    
    async def set(self, key: str, value: Any, ttl: int = DEFAULT_TTL) -> bool:
        """캐시에 데이터 저장"""
        await self.initialize()
        
        req_logger = create_request_context_logger("app.cache.set")
        
        try:
            # 직렬화
            serialized_data = self._serialize_data(value)
            data_size = len(serialized_data)
            
            # Redis에 저장
            if self.redis_client:
                try:
                    await self.redis_client.setex(key, ttl, serialized_data)
                    req_logger.debug(f"Redis 캐시 저장: {key} ({data_size} bytes, TTL={ttl}s)")
                except Exception as e:
                    req_logger.warning(f"Redis 저장 실패: {e}")
            
            # 메모리 캐시에도 저장 (작은 데이터만)
            if data_size < 10240:  # 10KB 미만
                self._set_memory_cache(key, value, ttl)
                req_logger.debug(f"메모리 캐시 저장: {key}")
            
            return True
            
        except Exception as e:
            req_logger.error(f"캐시 저장 실패: {e}")
            return False
    
    def _set_memory_cache(self, key: str, value: Any, ttl: int):
        """메모리 캐시에 데이터 저장"""
        # 캐시 크기 제한
        if len(self._memory_cache) >= self._memory_cache_size:
            # 가장 오래된 항목 제거 (LRU)
            oldest_key = min(self._memory_cache.keys(), 
                           key=lambda k: self._memory_cache[k]['created_at'])
            del self._memory_cache[oldest_key]
        
        self._memory_cache[key] = {
            'data': value,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(seconds=ttl)
        }
    
    async def delete(self, key: str) -> bool:
        """캐시에서 데이터 삭제"""
        await self.initialize()
        
        req_logger = create_request_context_logger("app.cache.delete")
        
        # 메모리 캐시에서 삭제
        if key in self._memory_cache:
            del self._memory_cache[key]
        
        # Redis에서 삭제
        if self.redis_client:
            try:
                result = await self.redis_client.delete(key)
                req_logger.debug(f"캐시 삭제: {key}")
                return bool(result)
            except Exception as e:
                req_logger.warning(f"Redis 삭제 실패: {e}")
        
        return True
    
    async def delete_pattern(self, pattern: str) -> int:
        """패턴에 매칭되는 모든 캐시 삭제"""
        await self.initialize()
        
        req_logger = create_request_context_logger("app.cache.delete_pattern")
        deleted_count = 0
        
        # 메모리 캐시에서 패턴 매칭 삭제
        import fnmatch
        keys_to_delete = [k for k in self._memory_cache.keys() if fnmatch.fnmatch(k, pattern)]
        for key in keys_to_delete:
            del self._memory_cache[key]
            deleted_count += 1
        
        # Redis에서 패턴 매칭 삭제
        if self.redis_client:
            try:
                keys = []
                async for key in self.redis_client.scan_iter(match=pattern):
                    keys.append(key.decode('utf-8') if isinstance(key, bytes) else key)
                
                if keys:
                    await self.redis_client.delete(*keys)
                    deleted_count += len(keys)
                    
            except Exception as e:
                req_logger.warning(f"Redis 패턴 삭제 실패: {e}")
        
        req_logger.info(f"패턴 '{pattern}' 매칭 캐시 {deleted_count}개 삭제")
        return deleted_count
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 정보"""
        await self.initialize()
        
        stats = {
            "memory_cache": {
                "size": len(self._memory_cache),
                "max_size": self._memory_cache_size,
                "usage_percentage": len(self._memory_cache) / self._memory_cache_size * 100
            },
            "redis_available": self.redis_client is not None,
            "compression_enabled": self.enable_compression
        }
        
        if self.redis_client:
            try:
                redis_info = await self.redis_client.info()
                stats["redis"] = {
                    "used_memory": redis_info.get("used_memory", 0),
                    "used_memory_human": redis_info.get("used_memory_human", "0B"),
                    "connected_clients": redis_info.get("connected_clients", 0),
                    "total_commands_processed": redis_info.get("total_commands_processed", 0),
                    "keyspace_hits": redis_info.get("keyspace_hits", 0),
                    "keyspace_misses": redis_info.get("keyspace_misses", 0)
                }
                
                # 히트율 계산
                hits = stats["redis"]["keyspace_hits"]
                misses = stats["redis"]["keyspace_misses"]
                if hits + misses > 0:
                    stats["redis"]["hit_rate"] = hits / (hits + misses) * 100
                else:
                    stats["redis"]["hit_rate"] = 0
                    
            except Exception as e:
                logger.warning(f"Redis 통계 조회 실패: {e}")
                stats["redis"] = {"error": str(e)}
        
        return stats
    
    # === 도메인별 캐시 헬퍼 메서드들 ===
    
    async def cache_analysis_results(self, filters: Dict[str, Any], page: int, size: int, results: List[Dict[str, Any]]) -> str:
        """분석 결과 목록 캐시"""
        cache_key = self._generate_cache_key(
            "analysis_results",
            **filters,
            page=page,
            size=size
        )
        await self.set(cache_key, results, ANALYSIS_RESULTS_TTL)
        return cache_key
    
    async def get_cached_analysis_results(self, filters: Dict[str, Any], page: int, size: int) -> Optional[List[Dict[str, Any]]]:
        """캐시된 분석 결과 목록 조회"""
        cache_key = self._generate_cache_key(
            "analysis_results",
            **filters,
            page=page,
            size=size
        )
        return await self.get(cache_key)
    
    async def cache_analysis_detail(self, result_id: str, result: Dict[str, Any]) -> str:
        """분석 결과 상세 캐시"""
        cache_key = f"analysis_detail:{result_id}"
        await self.set(cache_key, result, ANALYSIS_RESULTS_TTL)
        return cache_key
    
    async def get_cached_analysis_detail(self, result_id: str) -> Optional[Dict[str, Any]]:
        """캐시된 분석 결과 상세 조회"""
        cache_key = f"analysis_detail:{result_id}"
        return await self.get(cache_key)
    
    async def cache_statistics(self, stat_type: str, filters: Dict[str, Any], stats: Dict[str, Any]) -> str:
        """통계 데이터 캐시"""
        cache_key = self._generate_cache_key(
            f"statistics_{stat_type}",
            **filters
        )
        await self.set(cache_key, stats, STATISTICS_TTL)
        return cache_key
    
    async def get_cached_statistics(self, stat_type: str, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """캐시된 통계 데이터 조회"""
        cache_key = self._generate_cache_key(
            f"statistics_{stat_type}",
            **filters
        )
        return await self.get(cache_key)
    
    async def cache_kpi_data(self, params: Dict[str, Any], data: Any) -> str:
        """KPI 데이터 캐시"""
        cache_key = self._generate_cache_key(
            "kpi_data",
            **params
        )
        await self.set(cache_key, data, KPI_DATA_TTL)
        return cache_key
    
    async def get_cached_kpi_data(self, params: Dict[str, Any]) -> Optional[Any]:
        """캐시된 KPI 데이터 조회"""
        cache_key = self._generate_cache_key(
            "kpi_data",
            **params
        )
        return await self.get(cache_key)
    
    async def invalidate_analysis_caches(self, ne_id: Optional[str] = None, cell_id: Optional[str] = None):
        """분석 관련 캐시 무효화"""
        patterns = [
            "analysis_results:*",
            "statistics_*",
        ]
        
        if ne_id:
            patterns.append(f"*ne_id={ne_id}*")
        if cell_id:
            patterns.append(f"*cell_id={cell_id}*")
        
        for pattern in patterns:
            await self.delete_pattern(pattern)


# 전역 캐시 관리자 인스턴스
_cache_manager: Optional[CacheManager] = None


async def get_cache_manager() -> CacheManager:
    """캐시 관리자 인스턴스 반환"""
    global _cache_manager
    
    if _cache_manager is None:
        import os
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        _cache_manager = CacheManager(redis_url=redis_url)
        await _cache_manager.initialize()
    
    return _cache_manager


async def close_cache_manager():
    """캐시 관리자 종료"""
    global _cache_manager
    
    if _cache_manager:
        await _cache_manager.close()
        _cache_manager = None
