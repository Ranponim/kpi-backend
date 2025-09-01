"""
공통 모델 및 유틸리티 클래스 정의

이 모듈은 Pydantic과 MongoDB ObjectId를 원활하게 통합하기 위한
커스텀 타입과 기본 모델들을 포함합니다.
"""

from bson import ObjectId
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema
from typing import Any


class PyObjectId(ObjectId):
    """
    MongoDB ObjectId를 Pydantic에서 사용할 수 있도록 하는 커스텀 타입
    
    Features:
    - JSON 직렬화 시 자동으로 문자열로 변환
    - 문자열 입력을 ObjectId로 자동 변환
    - Pydantic 모델에서 네이티브하게 사용 가능
    """
    
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Pydantic v2용 스키마 정의"""
        return core_schema.json_or_python_schema(
            json_schema=core_schema.str_schema(),
            python_schema=core_schema.union_schema(
                [
                    core_schema.is_instance_schema(ObjectId),
                    core_schema.chain_schema(
                        [
                            core_schema.str_schema(),
                            core_schema.no_info_plain_validator_function(cls.validate),
                        ]
                    ),
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: str(x)
            ),
        )

    @classmethod
    def validate(cls, v):
        """
        입력값을 ObjectId로 변환하는 검증 함수
        
        Args:
            v: 변환할 값 (문자열 또는 ObjectId)
            
        Returns:
            ObjectId: 변환된 ObjectId 객체
            
        Raises:
            ValueError: 유효하지 않은 ObjectId 형식인 경우
        """
        if not ObjectId.is_valid(v):
            raise ValueError(f"Invalid ObjectId: {v}")
        return ObjectId(v)

    def __repr__(self) -> str:
        return f"PyObjectId('{self}')"

