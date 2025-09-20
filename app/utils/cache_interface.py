"""
캐시 인터페이스 모듈

다양한 캐시 구현(In-memory, Redis)을 위한 추상 인터페이스를 제공합니다.
"""

import json
import hashlib
from abc import ABC, abstractmethod
from typing import Any, Optional, Union
from datetime import datetime, timedelta


class CacheInterface(ABC):
    """캐시 인터페이스"""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 조회"""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """캐시에 값 저장"""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """캐시에서 값 삭제"""
        pass

    @abstractmethod
    def clear(self) -> bool:
        """캐시 전체 삭제"""
        pass

    @abstractmethod
    def has_key(self, key: str) -> bool:
        """키 존재 여부 확인"""
        pass

    @abstractmethod
    def get_stats(self) -> dict:
        """캐시 통계 정보"""
        pass


class CacheKeyGenerator:
    """캐시 키 생성 유틸리티"""

    @staticmethod
    def generate_key(*args, **kwargs) -> str:
        """함수 인자를 기반으로 캐시 키 생성"""
        # 인자들을 정렬하여 일관된 키 생성
        key_parts = []

        # 위치 인자
        for arg in args:
            key_parts.append(str(arg))

        # 키워드 인자 (정렬하여 일관성 유지)
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{v}")

        # 키 조합
        key_string = "|".join(key_parts)

        # 해시 생성 (길이 제한 및 특수문자 처리)
        key_hash = hashlib.md5(key_string.encode('utf-8')).hexdigest()
        return f"cache:{key_hash}"

    @staticmethod
    def generate_function_key(func_name: str, *args, **kwargs) -> str:
        """함수 이름과 인자를 기반으로 캐시 키 생성"""
        return CacheKeyGenerator.generate_key(func_name, *args, **kwargs)

    @staticmethod
    def generate_model_key(model_name: str, model_data: dict) -> str:
        """모델 데이터를 기반으로 캐시 키 생성"""
        # 중요한 필드만 선택하여 키 생성
        key_data = {
            k: v for k, v in model_data.items()
            if k not in ['timestamp', 'created_at', 'updated_at']  # 시간 필드 제외
        }
        return CacheKeyGenerator.generate_key(model_name, **key_data)


class CacheSerializer:
    """캐시 데이터 직렬화/역직렬화 유틸리티"""

    @staticmethod
    def serialize(data: Any) -> str:
        """데이터를 JSON 문자열로 직렬화"""
        try:
            # datetime 객체 처리
            if isinstance(data, datetime):
                return json.dumps({
                    "_type": "datetime",
                    "value": data.isoformat()
                })

            # timedelta 객체 처리
            if isinstance(data, timedelta):
                return json.dumps({
                    "_type": "timedelta",
                    "value": data.total_seconds()
                })

            # 일반 데이터
            return json.dumps(data, default=str)

        except (TypeError, ValueError) as e:
            raise ValueError(f"캐시 데이터 직렬화 실패: {e}")

    @staticmethod
    def deserialize(data_str: str) -> Any:
        """JSON 문자열을 원래 데이터로 역직렬화"""
        try:
            data = json.loads(data_str)

            # 특수 타입 복원
            if isinstance(data, dict) and "_type" in data:
                if data["_type"] == "datetime":
                    return datetime.fromisoformat(data["value"])
                elif data["_type"] == "timedelta":
                    return timedelta(seconds=data["value"])

            return data

        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"캐시 데이터 역직렬화 실패: {e}")


