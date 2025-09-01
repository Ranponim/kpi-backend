"""
입력 검증 유틸리티

이 모듈은 API 입력 데이터의 검증을 위한 고급 유틸리티들을 제공합니다.
Pydantic 모델과 함께 사용하여 더욱 강력한 데이터 검증을 수행합니다.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, validator
from bson import ObjectId
import json

logger = logging.getLogger("app.validation")


class ValidationError(Exception):
    """검증 오류 예외"""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self.message)


class ValidationRule:
    """검증 규칙 클래스"""
    
    def __init__(self, 
                 validator_func: Callable[[Any], bool],
                 error_message: str,
                 field_name: str = None):
        self.validator_func = validator_func
        self.error_message = error_message
        self.field_name = field_name
    
    def validate(self, value: Any, field_name: str = None) -> None:
        """값을 검증하고 실패 시 예외를 발생시킵니다."""
        field = field_name or self.field_name or "unknown"
        
        try:
            if not self.validator_func(value):
                raise ValidationError(
                    message=self.error_message,
                    field=field,
                    value=value
                )
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(
                message=f"검증 중 오류 발생: {str(e)}",
                field=field,
                value=value
            )


# === 기본 검증 규칙들 ===

def is_valid_ne_id(value: str) -> bool:
    """NE ID 형식 검증 (예: eNB001, gNB123)"""
    if not isinstance(value, str):
        return False
    pattern = r'^[egn]NB\d{3,6}$'
    return bool(re.match(pattern, value, re.IGNORECASE))

def is_valid_cell_id(value: str) -> bool:
    """Cell ID 형식 검증 (예: CELL001, Cell_123)"""
    if not isinstance(value, str):
        return False
    pattern = r'^CELL[_]?\d{3,6}$'
    return bool(re.match(pattern, value, re.IGNORECASE))

def is_valid_object_id(value: str) -> bool:
    """MongoDB ObjectId 형식 검증"""
    if not isinstance(value, str):
        return False
    try:
        ObjectId(value)
        return True
    except:
        return False

def is_valid_date_range(start_date: datetime, end_date: datetime) -> bool:
    """날짜 범위 검증"""
    if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
        return False
    
    # 시작일이 종료일보다 늦으면 안됨
    if start_date >= end_date:
        return False
    
    # 범위가 너무 크면 안됨 (예: 1년 이상)
    if (end_date - start_date).days > 365:
        return False
    
    # 미래 날짜는 허용하지 않음
    if end_date > datetime.now():
        return False
    
    return True

def is_valid_kpi_value(value: float) -> bool:
    """KPI 값 검증 (0-100 범위 또는 합리적인 범위)"""
    if not isinstance(value, (int, float)):
        return False
    
    # 음수 불허
    if value < 0:
        return False
    
    # 너무 큰 값 불허 (10000 이상)
    if value > 10000:
        return False
    
    return True

def is_valid_json_string(value: str, max_size: int = 10 * 1024 * 1024) -> bool:
    """JSON 문자열 검증"""
    if not isinstance(value, str):
        return False
    
    # 크기 제한
    if len(value.encode('utf-8')) > max_size:
        return False
    
    try:
        json.loads(value)
        return True
    except (json.JSONDecodeError, ValueError):
        return False

def is_safe_string(value: str, max_length: int = 1000) -> bool:
    """안전한 문자열 검증 (SQL 인젝션, XSS 방지)"""
    if not isinstance(value, str):
        return False
    
    # 길이 제한
    if len(value) > max_length:
        return False
    
    # 위험한 패턴 확인
    dangerous_patterns = [
        r'<script[^>]*>',  # XSS
        r'javascript:',    # XSS
        r'on\w+\s*=',     # 이벤트 핸들러
        r'(union|select|insert|update|delete|drop|create|alter)\s+',  # SQL
        r'--\s*',         # SQL 주석
        r'/\*.*\*/',      # SQL 주석
        r';\s*(drop|delete|update|insert)',  # SQL 체이닝
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, value, re.IGNORECASE):
            return False
    
    return True


# === 사전 정의된 검증 규칙들 ===

NE_ID_RULE = ValidationRule(
    is_valid_ne_id,
    "NE ID는 'eNB' 또는 'gNB' 뒤에 3-6자리 숫자 형식이어야 합니다 (예: eNB001, gNB123)"
)

CELL_ID_RULE = ValidationRule(
    is_valid_cell_id,
    "Cell ID는 'CELL' 뒤에 3-6자리 숫자 형식이어야 합니다 (예: CELL001, Cell_123)"
)

OBJECT_ID_RULE = ValidationRule(
    is_valid_object_id,
    "올바른 MongoDB ObjectId 형식이어야 합니다"
)

KPI_VALUE_RULE = ValidationRule(
    is_valid_kpi_value,
    "KPI 값은 0 이상 10000 이하의 숫자여야 합니다"
)

SAFE_STRING_RULE = ValidationRule(
    lambda x: is_safe_string(x, 500),
    "안전하지 않은 문자열입니다. 스크립트, SQL 인젝션 시도가 감지되었습니다"
)


# === 검증 헬퍼 함수들 ===

def validate_analysis_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    분석 결과 필터 검증
    
    Args:
        filters: 필터 딕셔너리
        
    Returns:
        검증된 필터 딕셔너리
        
    Raises:
        ValidationError: 검증 실패 시
    """
    validated = {}
    
    # NE ID 검증
    if 'ne_id' in filters and filters['ne_id']:
        NE_ID_RULE.validate(filters['ne_id'], 'ne_id')
        validated['ne_id'] = filters['ne_id'].strip()
    
    # Cell ID 검증
    if 'cell_id' in filters and filters['cell_id']:
        CELL_ID_RULE.validate(filters['cell_id'], 'cell_id')
        validated['cell_id'] = filters['cell_id'].strip()
    
    # 상태 검증
    if 'status' in filters and filters['status']:
        status = filters['status'].strip().lower()
        if status not in ['success', 'warning', 'error', 'pending']:
            raise ValidationError(
                "상태는 'success', 'warning', 'error', 'pending' 중 하나여야 합니다",
                'status',
                filters['status']
            )
        validated['status'] = status
    
    # 날짜 범위 검증
    if 'date_from' in filters and 'date_to' in filters:
        if filters['date_from'] and filters['date_to']:
            if not is_valid_date_range(filters['date_from'], filters['date_to']):
                raise ValidationError(
                    "날짜 범위가 올바르지 않습니다. 시작일은 종료일보다 이전이어야 하고, 범위는 1년을 초과할 수 없습니다",
                    'date_range',
                    {'date_from': filters['date_from'], 'date_to': filters['date_to']}
                )
            validated['date_from'] = filters['date_from']
            validated['date_to'] = filters['date_to']
    
    return validated


def validate_pagination_params(page: int, size: int) -> tuple[int, int]:
    """
    페이지네이션 매개변수 검증
    
    Args:
        page: 페이지 번호
        size: 페이지 크기
        
    Returns:
        검증된 (page, size) 튜플
        
    Raises:
        ValidationError: 검증 실패 시
    """
    # 페이지 번호 검증
    if not isinstance(page, int) or page < 1:
        raise ValidationError(
            "페이지 번호는 1 이상의 정수여야 합니다",
            'page',
            page
        )
    
    # 페이지 크기 검증
    if not isinstance(size, int) or size < 1 or size > 100:
        raise ValidationError(
            "페이지 크기는 1 이상 100 이하의 정수여야 합니다",
            'size',
            size
        )
    
    return page, size


def validate_analysis_result_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    분석 결과 데이터 검증
    
    Args:
        data: 분석 결과 데이터
        
    Returns:
        검증된 데이터
        
    Raises:
        ValidationError: 검증 실패 시
    """
    validated = {}
    
    # 필수 필드 검증
    required_fields = ['ne_id', 'cell_id', 'analysis_date', 'status']
    for field in required_fields:
        if field not in data or not data[field]:
            raise ValidationError(
                f"필수 필드 '{field}'가 누락되었습니다",
                field,
                data.get(field)
            )
    
    # NE ID 검증
    NE_ID_RULE.validate(data['ne_id'], 'ne_id')
    validated['ne_id'] = data['ne_id'].strip()
    
    # Cell ID 검증
    CELL_ID_RULE.validate(data['cell_id'], 'cell_id')
    validated['cell_id'] = data['cell_id'].strip()
    
    # 분석 날짜 검증
    if not isinstance(data['analysis_date'], datetime):
        raise ValidationError(
            "분석 날짜는 올바른 datetime 형식이어야 합니다",
            'analysis_date',
            data['analysis_date']
        )
    validated['analysis_date'] = data['analysis_date']
    
    # 상태 검증
    status = data['status'].strip().lower()
    if status not in ['success', 'warning', 'error', 'pending']:
        raise ValidationError(
            "상태는 'success', 'warning', 'error', 'pending' 중 하나여야 합니다",
            'status',
            data['status']
        )
    validated['status'] = status
    
    # 선택적 필드들 검증
    if 'results' in data and data['results']:
        if not isinstance(data['results'], list):
            raise ValidationError(
                "results는 배열이어야 합니다",
                'results',
                data['results']
            )
        validated['results'] = data['results']
    
    if 'stats' in data and data['stats']:
        if not isinstance(data['stats'], list):
            raise ValidationError(
                "stats는 배열이어야 합니다",
                'stats',
                data['stats']
            )
        validated['stats'] = data['stats']
    
    # 문자열 필드 안전성 검증
    string_fields = ['analysis_type', 'report_path']
    for field in string_fields:
        if field in data and data[field]:
            SAFE_STRING_RULE.validate(data[field], field)
            validated[field] = data[field].strip()
    
    return validated


def log_validation_event(event_type: str, field: str, value: Any, error: str = None) -> None:
    """
    검증 이벤트를 로깅합니다.
    
    Args:
        event_type: 이벤트 타입 ('success', 'failure')
        field: 검증된 필드명
        value: 검증된 값
        error: 오류 메시지 (실패 시)
    """
    from .request_tracing import get_current_request_id
    
    request_id = get_current_request_id()
    
    log_data = {
        "event": "validation",
        "type": event_type,
        "field": field,
        "request_id": request_id,
        "validation": True
    }
    
    if event_type == "success":
        logger.debug(f"검증 성공: {field}", extra=log_data)
    else:
        log_data.update({
            "error": error,
            "invalid_value": str(value)[:100]  # 값 로깅 시 크기 제한
        })
        logger.warning(f"검증 실패: {field} - {error}", extra=log_data)


# === 데코레이터 ===

def validate_input(validation_func: Callable[[Any], Any]):
    """
    입력 검증 데코레이터
    
    Args:
        validation_func: 검증 함수
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                # 검증 함수 실행
                if 'request' in kwargs:
                    # Request 객체가 있는 경우
                    request = kwargs['request']
                    # 요청 데이터 검증 로직 추가 가능
                
                # 원본 함수 실행
                result = await func(*args, **kwargs)
                return result
                
            except ValidationError as e:
                log_validation_event("failure", e.field or "unknown", e.value, e.message)
                raise
            except Exception as e:
                logger.error(f"검증 중 예상치 못한 오류: {e}", exc_info=True)
                raise
        
        return wrapper
    return decorator
