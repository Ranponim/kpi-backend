"""
캐시 데코레이터 모듈

함수에 캐싱 기능을 쉽게 추가할 수 있는 데코레이터를 제공합니다.
"""

import functools
import hashlib
import inspect
import logging
from typing import Any, Callable, Optional, Union
from datetime import datetime

from .cache_interface import CacheInterface, CacheKeyGenerator


logger = logging.getLogger(__name__)


class CacheDecorator:
    """캐시 데코레이터 클래스"""

    def __init__(self, cache: CacheInterface, ttl: Optional[int] = None):
        """
        캐시 데코레이터 초기화

        Args:
            cache: 사용할 캐시 인스턴스
            ttl: 캐시 TTL (초)
        """
        self.cache = cache
        self.ttl = ttl

    def __call__(self, func: Callable) -> Callable:
        """데코레이터 함수"""
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return self._execute_with_cache(func, *args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return self._execute_with_cache(func, *args, **kwargs)

        # async 함수인지 확인
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    def _execute_with_cache(self, func: Callable, *args, **kwargs) -> Any:
        """캐싱 로직 실행"""
        # 캐시 키 생성
        cache_key = self._generate_cache_key(func, *args, **kwargs)

        # 캐시 조회
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"캐시 히트: {cache_key}")
            return cached_result

        # 캐시 미스: 함수 실행
        logger.debug(f"캐시 미스: {cache_key}")
        try:
            if inspect.iscoroutinefunction(func):
                result = self._await_result(func(*args, **kwargs))
            else:
                result = func(*args, **kwargs)

            # 결과 캐싱
            if result is not None:  # None은 캐싱하지 않음
                self.cache.set(cache_key, result, self.ttl)

            return result

        except Exception as e:
            logger.warning(f"함수 실행 실패: {func.__name__}, 오류: {e}")
            raise

    def _await_result(self, coro):
        """코루틴 실행"""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 이미 실행 중인 이벤트 루프에서 실행
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(coro)
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            # 이벤트 루프가 없는 경우
            return asyncio.run(coro)

    def _generate_cache_key(self, func: Callable, *args, **kwargs) -> str:
        """함수 호출을 위한 캐시 키 생성"""
        # 함수 이름
        func_name = f"{func.__module__}.{func.__qualname__}"

        # 인자 처리 (self 제외)
        if args and hasattr(args[0], '__class__'):
            # 인스턴스 메서드의 경우 self 제외
            args = args[1:]

        # 키워드 인자에서 제외할 매개변수들
        exclude_params = {'request', 'response', 'background_tasks'}

        # 키워드 인자 필터링
        filtered_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in exclude_params and not callable(v)
        }

        return CacheKeyGenerator.generate_function_key(func_name, *args, **filtered_kwargs)


def cached(cache: CacheInterface, ttl: Optional[int] = None):
    """
    캐시 데코레이터 함수

    Args:
        cache: 사용할 캐시 인스턴스
        ttl: 캐시 TTL (초)

    Returns:
        데코레이터 함수
    """
    def decorator(func: Callable) -> Callable:
        decorator_instance = CacheDecorator(cache, ttl)
        return decorator_instance(func)

    return decorator


def cache_method(cache: CacheInterface, ttl: Optional[int] = None, key_func: Optional[Callable] = None):
    """
    클래스 메서드용 캐시 데코레이터

    Args:
        cache: 사용할 캐시 인스턴스
        ttl: 캐시 TTL (초)
        key_func: 사용자 정의 키 생성 함수

    Returns:
        데코레이터 함수
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # 사용자 정의 키 함수 사용
            if key_func:
                cache_key = key_func(self, *args, **kwargs)
            else:
                # 기본 키 생성
                class_name = self.__class__.__name__
                method_name = func.__name__
                func_name = f"{class_name}.{method_name}"

                # 인스턴스 ID를 키에 포함하여 인스턴스별 캐싱
                instance_id = id(self)
                cache_key = CacheKeyGenerator.generate_function_key(
                    func_name, instance_id, *args, **kwargs
                )

            # 캐시 조회
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"메서드 캐시 히트: {cache_key}")
                return cached_result

            # 캐시 미스: 메서드 실행
            logger.debug(f"메서드 캐시 미스: {cache_key}")
            result = func(self, *args, **kwargs)

            # 결과 캐싱
            if result is not None:
                cache.set(cache_key, result, ttl)

            return result

        return wrapper

    return decorator


class CacheManager:
    """캐시 관리자"""

    def __init__(self, cache: CacheInterface):
        self.cache = cache
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def cached(self, ttl: Optional[int] = None):
        """캐시 데코레이터 생성"""
        return cached(self.cache, ttl)

    def cache_method(self, ttl: Optional[int] = None, key_func: Optional[Callable] = None):
        """메서드 캐시 데코레이터 생성"""
        return cache_method(self.cache, ttl, key_func)

    def get_stats(self) -> dict:
        """캐시 통계 조회"""
        return self.cache.get_stats()

    def clear_cache(self, pattern: Optional[str] = None):
        """캐시 정리"""
        if pattern:
            # 패턴 기반 삭제 (미구현)
            self.logger.warning("패턴 기반 캐시 삭제는 아직 지원되지 않습니다")
            return False
        else:
            return self.cache.clear()

    def invalidate_keys(self, keys: list):
        """특정 키들 무효화"""
        invalidated = 0
        for key in keys:
            if self.cache.delete(key):
                invalidated += 1

        self.logger.info(f"캐시 키 {invalidated}/{len(keys)}개 무효화됨")
        return invalidated


