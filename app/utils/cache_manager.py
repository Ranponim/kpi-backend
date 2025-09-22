"""
캐시 관리자 모듈

전역 캐시 인스턴스와 설정을 관리합니다.

본 모듈은 동기 캐시 구현(`CacheInterface`)을 비동기 컨텍스트에서 안전하게
사용할 수 있도록 래핑한 `GlobalCacheManager`를 제공합니다.
분석 결과 목록/상세, 마할라노비스, 모니터링 라우터에서 사용되는
비동기 메서드 시그니처를 모두 구현하여 await 사용 시 오류가 발생하지 않도록 합니다.
"""

import logging
import os
import threading
from typing import Optional, Any, Dict, List
import fnmatch

from .cache_in_memory import InMemoryCache
from .cache_interface import CacheInterface, CacheKeyGenerator
from .cache_decorator import cached as decorator_cached, cache_method as decorator_cache_method

logger = logging.getLogger(__name__)


class CacheConfig:
    """캐시 설정"""

    def __init__(self):
        # 환경 변수에서 설정 읽기
        self.cache_type = os.getenv("CACHE_TYPE", "memory")  # memory 또는 redis
        self.max_size = int(os.getenv("CACHE_MAX_SIZE", "1000"))
        self.default_ttl = int(os.getenv("CACHE_DEFAULT_TTL", "3600"))  # 1시간
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

        # 메모리 캐시 설정
        self.memory_max_size = int(os.getenv("CACHE_MEMORY_MAX_SIZE", "1000"))
        self.memory_default_ttl = int(os.getenv("CACHE_MEMORY_DEFAULT_TTL", "3600"))

        # Redis 캐시 설정
        self.redis_max_connections = int(os.getenv("REDIS_MAX_CONNECTIONS", "10"))
        self.redis_socket_timeout = int(os.getenv("REDIS_SOCKET_TIMEOUT", "5"))


# 전역 캐시 설정
cache_config = CacheConfig()

# 전역 캐시 인스턴스
_cache_instance = None
_cache_manager = None


def get_cache_instance() -> CacheInterface:
    """전역 캐시 인스턴스 조회"""
    global _cache_instance

    if _cache_instance is None:
        if cache_config.cache_type == "redis":
            try:
                from .cache_redis import RedisCache
                _cache_instance = RedisCache(
                    redis_url=cache_config.redis_url,
                    max_connections=cache_config.redis_max_connections,
                    socket_timeout=cache_config.redis_socket_timeout,
                    default_ttl=cache_config.default_ttl
                )
                logger.info("Redis 캐시 초기화 완료")
            except ImportError:
                logger.warning("Redis 라이브러리가 설치되지 않아 메모리 캐시로 전환합니다")
                _cache_instance = InMemoryCache(
                    max_size=cache_config.memory_max_size,
                    default_ttl=cache_config.memory_default_ttl
                )
            except Exception as e:
                logger.error(f"Redis 캐시 초기화 실패: {e}, 메모리 캐시로 전환합니다")
                _cache_instance = InMemoryCache(
                    max_size=cache_config.memory_max_size,
                    default_ttl=cache_config.memory_default_ttl
                )
        else:
            _cache_instance = InMemoryCache(
                max_size=cache_config.memory_max_size,
                default_ttl=cache_config.memory_default_ttl
            )
            logger.info("In-memory 캐시 초기화 완료")

    return _cache_instance


class GlobalCacheManager:
    """비동기 친화 캐시 관리자 래퍼

    - 동기 `CacheInterface`를 비동기 메서드로 감싸 FastAPI 컨텍스트에서 안전하게 사용
    - 라우터들이 기대하는 메서드 시그니처를 통일 제공
    """

    def __init__(self, cache: CacheInterface, default_ttl: Optional[int] = None):
        self._cache: CacheInterface = cache
        self._default_ttl: Optional[int] = default_ttl
        self._logger = logging.getLogger(f"{__name__}.GlobalCacheManager")
        # 내부 키 인덱스 (패턴 삭제, 전체 나열 등을 위해 추적)
        self._tracked_keys: set[str] = set()
        self._lock = threading.RLock()

    # ----- 기본 캐시 연산 (비동기 래핑) -----
    async def get(self, key: str) -> Optional[Any]:
        with self._lock:
            return self._cache.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        with self._lock:
            stored = self._cache.set(key, value, ttl if ttl is not None else self._default_ttl)
            if stored:
                self._tracked_keys.add(key)
            return stored

    async def delete(self, key: str) -> bool:
        with self._lock:
            deleted = self._cache.delete(key)
            if deleted and key in self._tracked_keys:
                self._tracked_keys.remove(key)
            return deleted

    async def exists(self, key: str) -> bool:
        with self._lock:
            return self._cache.has_key(key)

    async def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return self._cache.get_stats()

    async def get_all_keys(self) -> List[str]:
        # InMemoryCache가 키 열람을 제공하지 않아 내부 추적 키 반환
        with self._lock:
            return list(self._tracked_keys)

    async def clear_pattern(self, pattern: str) -> int:
        """와일드카드 패턴으로 일치하는 키 삭제

        Args:
            pattern: fnmatch 패턴 (예: "mahalanobis:*" 또는 "analysis:list:*")
        Returns:
            삭제된 키 개수
        """
        with self._lock:
            targets = [k for k in self._tracked_keys if fnmatch.fnmatch(k, pattern)]
            count = 0
            for key in targets:
                if self._cache.delete(key):
                    count += 1
                    self._tracked_keys.discard(key)
            return count

    # alias for compatibility
    async def delete_pattern(self, pattern: str) -> int:
        return await self.clear_pattern(pattern)

    # ----- 분석 결과 전용 헬퍼 -----
    def _build_analysis_list_key(self, filters: Dict[str, Any], page: int, size: int) -> str:
        # 필터는 키 이름 기준 정렬하여 일관적 키 생성
        filters = filters or {}
        ordered_items = {k: filters[k] for k in sorted(filters.keys())}
        return CacheKeyGenerator.generate_key("analysis:list", page=page, size=size, **ordered_items)

    async def get_cached_analysis_results(self, filters: Dict[str, Any], page: int, size: int) -> Optional[Dict[str, Any]]:
        key = self._build_analysis_list_key(filters, page, size)
        with self._lock:
            return self._cache.get(key)

    async def cache_analysis_results(self, filters: Dict[str, Any], page: int, size: int, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        key = self._build_analysis_list_key(filters, page, size)
        with self._lock:
            stored = self._cache.set(key, data, ttl if ttl is not None else self._default_ttl)
            if stored:
                self._tracked_keys.add(key)
            return stored

    async def invalidate_analysis_caches(self, ne_id: Optional[str] = None, cell_id: Optional[str] = None) -> int:
        """분석 목록 관련 캐시 무효화. 간단히 프리픽스 기반 삭제로 처리."""
        pattern = "analysis:list*"
        return await self.clear_pattern(pattern)

    # ----- 모니터링 호환 헬퍼 -----
    async def get_cache_stats(self) -> Dict[str, Any]:
        """상위 레이어(모니터링)에서 기대하는 형태로 통계 제공"""
        stats = await self.get_stats()
        # InMemory 기준 보강 필드
        memory_stats = {
            "size": stats.get("size", 0),
            "max_size": stats.get("max_size", 0),
            "hit_rate": stats.get("hit_rate", 0),
        }
        usage_percentage = 0
        if memory_stats["max_size"]:
            usage_percentage = round(memory_stats["size"] / memory_stats["max_size"] * 100, 2)

        return {
            "redis_available": False,
            "memory_cache": {
                **memory_stats,
                "usage_percentage": usage_percentage,
            },
            "type": stats.get("cache_type", stats.get("type", "in_memory")),
            **stats,
        }


async def get_cache_manager() -> GlobalCacheManager:
    """전역 캐시 관리자 조회 (비동기)"""
    global _cache_manager

    if _cache_manager is None:
        cache_instance = get_cache_instance()
        _cache_manager = GlobalCacheManager(cache_instance, default_ttl=cache_config.default_ttl)

    return _cache_manager


def clear_global_cache():
    """전역 캐시 정리"""
    global _cache_instance, _cache_manager

    if _cache_instance:
        _cache_instance.clear()
        logger.info("전역 캐시 정리 완료")

    # 캐시 관리자도 재초기화
    _cache_manager = None


def get_cache_stats():
    """캐시 통계 조회"""
    cache_instance = get_cache_instance()
    return cache_instance.get_stats()


# 편의 함수들 (동기 데코레이터는 직접 CacheInstance를 사용)
def cached(ttl: Optional[int] = None):
    """캐시 데코레이터"""
    cache = get_cache_instance()
    return decorator_cached(cache, ttl)


def cache_method(ttl: Optional[int] = None):
    """메서드 캐시 데코레이터"""
    cache = get_cache_instance()
    return decorator_cache_method(cache, ttl)


async def close_cache_manager():
    """
    캐시 매니저 종료 (앱 종료 시 호출)
    
    현재 In-memory 캐시를 사용하므로 특별한 정리 작업은 없지만,
    향후 Redis 등 외부 캐시 사용 시 연결 종료 등에 사용될 수 있습니다.
    """
    global _cache_manager
    if _cache_manager:
        # 현재는 특별한 정리 작업 없음 (In-memory 캐시)
        _cache_manager = None