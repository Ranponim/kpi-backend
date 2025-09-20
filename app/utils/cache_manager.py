"""
캐시 관리자 모듈

전역 캐시 인스턴스와 설정을 관리합니다.
"""

import logging
import os
from typing import Optional

from .cache_in_memory import InMemoryCache
from .cache_decorator import CacheManager

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


def get_cache_instance():
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


def get_cache_manager() -> CacheManager:
    """전역 캐시 관리자 조회"""
    global _cache_manager

    if _cache_manager is None:
        cache_instance = get_cache_instance()
        _cache_manager = CacheManager(cache_instance)

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


# 편의 함수들
def cached(ttl: Optional[int] = None):
    """캐시 데코레이터"""
    manager = get_cache_manager()
    return manager.cached(ttl)


def cache_method(ttl: Optional[int] = None):
    """메서드 캐시 데코레이터"""
    manager = get_cache_manager()
    return manager.cache_method(ttl)


def close_cache_manager():
    """
    캐시 매니저 종료 (앱 종료 시 호출)
    
    현재 In-memory 캐시를 사용하므로 특별한 정리 작업은 없지만,
    향후 Redis 등 외부 캐시 사용 시 연결 종료 등에 사용될 수 있습니다.
    """
    global _cache_manager
    if _cache_manager:
        # 현재는 특별한 정리 작업 없음 (In-memory 캐시)
        _cache_manager = None