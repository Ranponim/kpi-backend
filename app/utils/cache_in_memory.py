"""
In-memory 캐시 구현 모듈

메모리 기반 캐시로, 빠른 속도와 간단한 구현을 제공합니다.
"""

import time
import threading
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from .cache_interface import CacheInterface, CacheSerializer


@dataclass
class CacheEntry:
    """캐시 엔트리"""
    value: Any
    expires_at: Optional[float] = None

    def is_expired(self) -> bool:
        """만료 여부 확인"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class InMemoryCache(CacheInterface):
    """In-memory 캐시 구현"""

    def __init__(self, max_size: int = 1000, default_ttl: Optional[int] = None):
        """
        In-memory 캐시 초기화

        Args:
            max_size: 최대 캐시 엔트리 수 (LRU 방식으로 제거)
            default_ttl: 기본 TTL (초)
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = threading.RLock()

        # 통계 정보
        self._hits = 0
        self._misses = 0
        self._sets = 0
        self._deletes = 0
        self._evictions = 0

        # LRU를 위한 접근 시간 추적
        self._access_times: Dict[str, float] = {}

    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 조회"""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            if entry.is_expired():
                # 만료된 엔트리 제거
                del self._cache[key]
                if key in self._access_times:
                    del self._access_times[key]
                self._misses += 1
                self._evictions += 1
                return None

            # LRU 업데이트
            self._access_times[key] = time.time()
            self._hits += 1

            # 역직렬화하여 반환
            try:
                return CacheSerializer.deserialize(entry.value)
            except Exception:
                # 역직렬화 실패 시 엔트리 제거
                del self._cache[key]
                if key in self._access_times:
                    del self._access_times[key]
                self._misses += 1
                return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """캐시에 값 저장"""
        with self._lock:
            try:
                # 직렬화
                serialized_value = CacheSerializer.serialize(value)

                # TTL 설정
                expires_at = None
                if ttl is not None:
                    expires_at = time.time() + ttl
                elif self._default_ttl is not None:
                    expires_at = time.time() + self._default_ttl

                # 캐시 엔트리 생성
                entry = CacheEntry(value=serialized_value, expires_at=expires_at)

                # 크기 제한 확인 및 LRU 제거
                if key not in self._cache and len(self._cache) >= self._max_size:
                    self._evict_lru()

                # 저장
                self._cache[key] = entry
                self._access_times[key] = time.time()
                self._sets += 1

                return True

            except Exception:
                return False

    def delete(self, key: str) -> bool:
        """캐시에서 값 삭제"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._access_times:
                    del self._access_times[key]
                self._deletes += 1
                return True
            return False

    def clear(self) -> bool:
        """캐시 전체 삭제"""
        with self._lock:
            cache_size = len(self._cache)
            self._cache.clear()
            self._access_times.clear()
            self._deletes += cache_size
            return True

    def has_key(self, key: str) -> bool:
        """키 존재 여부 확인"""
        with self._lock:
            if key not in self._cache:
                return False

            entry = self._cache[key]
            if entry.is_expired():
                # 만료된 엔트리 제거
                del self._cache[key]
                if key in self._access_times:
                    del self._access_times[key]
                self._evictions += 1
                return False

            return True

    def get_stats(self) -> dict:
        """캐시 통계 정보"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

            return {
                "cache_type": "in_memory",
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 2),
                "sets": self._sets,
                "deletes": self._deletes,
                "evictions": self._evictions,
                "default_ttl": self._default_ttl,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

    def _evict_lru(self):
        """LRU 방식으로 가장 오래된 엔트리 제거"""
        if not self._access_times:
            return

        # 가장 오래된 접근 시간 찾기
        oldest_key = min(self._access_times, key=self._access_times.get)

        # 제거
        del self._cache[oldest_key]
        del self._access_times[oldest_key]
        self._evictions += 1

    def cleanup_expired(self):
        """만료된 엔트리 정리"""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]

            for key in expired_keys:
                del self._cache[key]
                if key in self._access_times:
                    del self._access_times[key]

            if expired_keys:
                self._evictions += len(expired_keys)


