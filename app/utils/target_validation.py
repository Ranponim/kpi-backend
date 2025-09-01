"""
타겟 검증 유틸리티 모듈

NE, Cell ID, Host 필터링에 대한 포괄적인 검증 기능을 제공합니다.
기존 analysis_llm.py의 검증 로직을 확장하고 모듈화합니다.
"""

import logging
import re
import ipaddress
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass

import validators
import dns.resolver
import psycopg2
import psycopg2.extras
from pydantic import BaseModel, Field

from ..exceptions import (
    TargetValidationException,
    HostValidationException, 
    RelationshipValidationException,
    FilterCombinationException,
    raise_host_validation_error,
    raise_relationship_validation_error,
    raise_filter_combination_error
)

# 로깅 설정
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """검증 결과를 담는 데이터 클래스"""
    is_valid: bool
    valid_items: List[str]
    invalid_items: List[str]
    validation_errors: Dict[str, str]
    metadata: Dict[str, Any]


class TargetFilters(BaseModel):
    """타겟 필터를 위한 Pydantic 모델"""
    ne_filters: Optional[List[str]] = Field(None, description="NE ID 필터 목록")
    cellid_filters: Optional[List[str]] = Field(None, description="Cell ID 필터 목록")
    host_filters: Optional[List[str]] = Field(None, description="Host ID 필터 목록")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ne_filters": ["nvgnb#10000", "nvgnb#20000"],
                "cellid_filters": ["2010", "2011", "8418"],
                "host_filters": ["host01", "192.168.1.10", "10.251.196.122"]
            }
        }


class NEIDValidator:
    """
    NE ID 검증 클래스
    
    NE ID 형식 검증 및 데이터베이스 존재 여부 확인을 담당합니다.
    """
    
    # NE ID 정규식 패턴들 (확장 가능)
    NE_ID_PATTERNS = {
        'nvgnb': r'^nvgnb#\d{5,}$',  # nvgnb#10000 형식
        'enb': r'^enb#\d{3,}$',      # enb#123 형식  
        'gnb': r'^gnb#\d{3,}$',      # gnb#123 형식
        'generic': r'^[a-zA-Z]+#\d+$'  # 일반적인 패턴
    }
    
    def __init__(self, db_connection=None):
        """
        NE ID 검증기 초기화
        
        Args:
            db_connection: PostgreSQL 연결 객체 (선택적)
        """
        self.db_connection = db_connection
        logger.info("NEIDValidator 초기화 완료")
    
    def validate_format(self, ne_id: str) -> Tuple[bool, str]:
        """
        NE ID 형식을 검증합니다.
        
        Args:
            ne_id: 검증할 NE ID
            
        Returns:
            Tuple[bool, str]: (검증 성공 여부, 오류 메시지)
        """
        if not isinstance(ne_id, str):
            return False, f"NE ID는 문자열이어야 합니다: {type(ne_id)}"
        
        ne_id = ne_id.strip()
        if not ne_id:
            return False, "NE ID는 빈 문자열일 수 없습니다"
        
        # 패턴별 검증
        for pattern_name, pattern in self.NE_ID_PATTERNS.items():
            if re.match(pattern, ne_id, re.IGNORECASE):
                logger.debug(f"NE ID '{ne_id}' 형식 검증 성공: {pattern_name} 패턴")
                return True, ""
        
        return False, f"유효하지 않은 NE ID 형식: '{ne_id}' (예상: nvgnb#10000, enb#123 등)"
    
    def validate_existence(self, ne_id: str, table: str = "summary", ne_column: str = "ne") -> Tuple[bool, str]:
        """
        데이터베이스에서 NE ID 존재 여부를 확인합니다.
        
        Args:
            ne_id: 확인할 NE ID
            table: 조회할 테이블명
            ne_column: NE ID 컬럼명
            
        Returns:
            Tuple[bool, str]: (존재 여부, 오류 메시지)
        """
        if not self.db_connection:
            logger.warning("DB 연결이 없어 NE ID 존재 여부를 확인할 수 없습니다")
            return True, "DB 연결 없음 - 존재 여부 확인 생략"
        
        try:
            sql = f"SELECT COUNT(*) FROM {table} WHERE {ne_column} = %s LIMIT 1"
            with self.db_connection.cursor() as cur:
                cur.execute(sql, (ne_id,))
                count = cur.fetchone()[0]
                
            exists = count > 0
            if exists:
                logger.debug(f"NE ID '{ne_id}' 존재 확인됨")
                return True, ""
            else:
                return False, f"NE ID '{ne_id}'가 데이터베이스에 존재하지 않습니다"
                
        except Exception as e:
            logger.error(f"NE ID 존재 여부 확인 실패: {e}")
            return False, f"DB 조회 오류: {str(e)}"
    
    def validate_multiple(self, ne_ids: List[str], **kwargs) -> ValidationResult:
        """
        여러 NE ID를 일괄 검증합니다.
        
        Args:
            ne_ids: 검증할 NE ID 목록
            **kwargs: validate_existence에 전달할 추가 파라미터
            
        Returns:
            ValidationResult: 검증 결과
        """
        logger.info(f"다중 NE ID 검증 시작: {len(ne_ids)}개")
        
        valid_items = []
        invalid_items = []
        validation_errors = {}
        
        for ne_id in ne_ids:
            # 1. 형식 검증
            format_valid, format_error = self.validate_format(ne_id)
            if not format_valid:
                invalid_items.append(ne_id)
                validation_errors[ne_id] = format_error
                continue
            
            # 2. 존재 여부 검증 (DB 연결이 있는 경우)
            if self.db_connection:
                exists_valid, exists_error = self.validate_existence(ne_id, **kwargs)
                if not exists_valid:
                    invalid_items.append(ne_id)
                    validation_errors[ne_id] = exists_error
                    continue
            
            # 모든 검증 통과
            valid_items.append(ne_id)
        
        is_valid = len(invalid_items) == 0
        metadata = {
            "total_count": len(ne_ids),
            "valid_count": len(valid_items),
            "invalid_count": len(invalid_items),
            "db_check_enabled": self.db_connection is not None
        }
        
        logger.info(f"NE ID 검증 완료: {len(valid_items)}/{len(ne_ids)} 유효")
        return ValidationResult(is_valid, valid_items, invalid_items, validation_errors, metadata)


class CellIDValidator:
    """
    Cell ID 검증 클래스
    
    Cell ID 형식 검증 및 데이터베이스 존재 여부 확인을 담당합니다.
    """
    
    def __init__(self, db_connection=None):
        """
        Cell ID 검증기 초기화
        
        Args:
            db_connection: PostgreSQL 연결 객체 (선택적)
        """
        self.db_connection = db_connection
        logger.info("CellIDValidator 초기화 완료")
    
    def validate_format(self, cell_id: Union[str, int]) -> Tuple[bool, str]:
        """
        Cell ID 형식을 검증합니다.
        
        Args:
            cell_id: 검증할 Cell ID
            
        Returns:
            Tuple[bool, str]: (검증 성공 여부, 오류 메시지)
        """
        # 숫자로 변환 시도
        try:
            if isinstance(cell_id, str):
                cell_id = cell_id.strip()
                if not cell_id:
                    return False, "Cell ID는 빈 문자열일 수 없습니다"
                cell_id_int = int(cell_id)
            elif isinstance(cell_id, int):
                cell_id_int = cell_id
            else:
                return False, f"Cell ID는 숫자 또는 숫자 문자열이어야 합니다: {type(cell_id)}"
            
            # 범위 검증 (일반적인 Cell ID 범위)
            if cell_id_int < 0:
                return False, f"Cell ID는 0 이상이어야 합니다: {cell_id_int}"
            if cell_id_int > 268435455:  # 2^28 - 1 (LTE 최대값)
                return False, f"Cell ID 범위를 초과했습니다: {cell_id_int} (최대: 268435455)"
            
            logger.debug(f"Cell ID '{cell_id}' 형식 검증 성공")
            return True, ""
            
        except ValueError:
            return False, f"Cell ID를 숫자로 변환할 수 없습니다: '{cell_id}'"
    
    def validate_existence(self, cell_id: Union[str, int], table: str = "summary", 
                          cell_column: str = "cellid") -> Tuple[bool, str]:
        """
        데이터베이스에서 Cell ID 존재 여부를 확인합니다.
        
        Args:
            cell_id: 확인할 Cell ID
            table: 조회할 테이블명  
            cell_column: Cell ID 컬럼명
            
        Returns:
            Tuple[bool, str]: (존재 여부, 오류 메시지)
        """
        if not self.db_connection:
            logger.warning("DB 연결이 없어 Cell ID 존재 여부를 확인할 수 없습니다")
            return True, "DB 연결 없음 - 존재 여부 확인 생략"
        
        try:
            # Cell ID를 정수로 정규화
            cell_id_int = int(cell_id)
            
            sql = f"SELECT COUNT(*) FROM {table} WHERE {cell_column} = %s LIMIT 1"
            with self.db_connection.cursor() as cur:
                cur.execute(sql, (cell_id_int,))
                count = cur.fetchone()[0]
                
            exists = count > 0
            if exists:
                logger.debug(f"Cell ID '{cell_id}' 존재 확인됨")
                return True, ""
            else:
                return False, f"Cell ID '{cell_id}'가 데이터베이스에 존재하지 않습니다"
                
        except Exception as e:
            logger.error(f"Cell ID 존재 여부 확인 실패: {e}")
            return False, f"DB 조회 오류: {str(e)}"
    
    def validate_multiple(self, cell_ids: List[Union[str, int]], **kwargs) -> ValidationResult:
        """
        여러 Cell ID를 일괄 검증합니다.
        
        Args:
            cell_ids: 검증할 Cell ID 목록
            **kwargs: validate_existence에 전달할 추가 파라미터
            
        Returns:
            ValidationResult: 검증 결과
        """
        logger.info(f"다중 Cell ID 검증 시작: {len(cell_ids)}개")
        
        valid_items = []
        invalid_items = []
        validation_errors = {}
        
        for cell_id in cell_ids:
            cell_id_str = str(cell_id)
            
            # 1. 형식 검증
            format_valid, format_error = self.validate_format(cell_id)
            if not format_valid:
                invalid_items.append(cell_id_str)
                validation_errors[cell_id_str] = format_error
                continue
            
            # 2. 존재 여부 검증 (DB 연결이 있는 경우)
            if self.db_connection:
                exists_valid, exists_error = self.validate_existence(cell_id, **kwargs)
                if not exists_valid:
                    invalid_items.append(cell_id_str)
                    validation_errors[cell_id_str] = exists_error
                    continue
            
            # 모든 검증 통과
            valid_items.append(cell_id_str)
        
        is_valid = len(invalid_items) == 0
        metadata = {
            "total_count": len(cell_ids),
            "valid_count": len(valid_items),
            "invalid_count": len(invalid_items),
            "db_check_enabled": self.db_connection is not None
        }
        
        logger.info(f"Cell ID 검증 완료: {len(valid_items)}/{len(cell_ids)} 유효")
        return ValidationResult(is_valid, valid_items, invalid_items, validation_errors, metadata)


class HostValidator:
    """
    Host 검증 클래스
    
    Host ID (IP 주소, 호스트명) 형식 검증 및 DNS 확인을 담당합니다.
    """
    
    def __init__(self, db_connection=None, enable_dns_check: bool = False):
        """
        Host 검증기 초기화
        
        Args:
            db_connection: PostgreSQL 연결 객체 (선택적)
            enable_dns_check: DNS 조회 활성화 여부
        """
        self.db_connection = db_connection
        self.enable_dns_check = enable_dns_check
        logger.info(f"HostValidator 초기화 완료 (DNS 확인: {enable_dns_check})")
    
    def validate_format(self, host_id: str) -> Tuple[bool, str]:
        """
        Host ID 형식을 검증합니다.
        
        Args:
            host_id: 검증할 Host ID (IP 주소 또는 호스트명)
            
        Returns:
            Tuple[bool, str]: (검증 성공 여부, 오류 메시지)
        """
        if not isinstance(host_id, str):
            return False, f"Host ID는 문자열이어야 합니다: {type(host_id)}"
        
        host_id = host_id.strip()
        if not host_id:
            return False, "Host ID는 빈 문자열일 수 없습니다"
        
        # 1. IP 주소 검증 시도 (IPv4/IPv6)
        try:
            ipaddress.ip_address(host_id)
            logger.debug(f"Host ID '{host_id}' IP 주소 형식 검증 성공")
            return True, ""
        except ValueError:
            pass  # IP 주소가 아니므로 호스트명으로 검증 계속
        
        # 2. 호스트명 검증
        if validators.domain(host_id):
            logger.debug(f"Host ID '{host_id}' 도메인 형식 검증 성공")
            return True, ""
        
        # 3. 단순 호스트명 패턴 검증 (alphanumeric + hyphen/underscore)
        hostname_pattern = r'^[a-zA-Z0-9][a-zA-Z0-9\-_]*[a-zA-Z0-9]$|^[a-zA-Z0-9]$'
        if re.match(hostname_pattern, host_id):
            logger.debug(f"Host ID '{host_id}' 호스트명 패턴 검증 성공")
            return True, ""
        
        return False, f"유효하지 않은 Host ID 형식: '{host_id}' (IP 주소, 도메인명, 또는 호스트명이어야 함)"
    
    def validate_dns_resolution(self, host_id: str) -> Tuple[bool, str]:
        """
        DNS 조회를 통해 호스트 해석 가능 여부를 확인합니다.
        
        Args:
            host_id: DNS 조회할 Host ID
            
        Returns:
            Tuple[bool, str]: (해석 가능 여부, 오류 메시지)
        """
        if not self.enable_dns_check:
            return True, "DNS 확인 비활성화됨"
        
        # IP 주소인 경우 DNS 조회 생략
        try:
            ipaddress.ip_address(host_id)
            return True, "IP 주소 - DNS 조회 생략"
        except ValueError:
            pass
        
        try:
            # DNS A 레코드 조회
            dns.resolver.resolve(host_id, 'A')
            logger.debug(f"Host ID '{host_id}' DNS 조회 성공")
            return True, ""
        except dns.resolver.NXDOMAIN:
            return False, f"Host '{host_id}'를 DNS에서 찾을 수 없습니다"
        except dns.resolver.Timeout:
            return False, f"Host '{host_id}' DNS 조회 시간 초과"
        except Exception as e:
            return False, f"Host '{host_id}' DNS 조회 오류: {str(e)}"
    
    def validate_existence(self, host_id: str, table: str = "summary", 
                          host_column: str = "host") -> Tuple[bool, str]:
        """
        데이터베이스에서 Host ID 존재 여부를 확인합니다.
        
        Args:
            host_id: 확인할 Host ID
            table: 조회할 테이블명
            host_column: Host ID 컬럼명
            
        Returns:
            Tuple[bool, str]: (존재 여부, 오류 메시지)
        """
        if not self.db_connection:
            logger.warning("DB 연결이 없어 Host ID 존재 여부를 확인할 수 없습니다")
            return True, "DB 연결 없음 - 존재 여부 확인 생략"
        
        try:
            sql = f"SELECT COUNT(*) FROM {table} WHERE {host_column} = %s LIMIT 1"
            with self.db_connection.cursor() as cur:
                cur.execute(sql, (host_id,))
                count = cur.fetchone()[0]
                
            exists = count > 0
            if exists:
                logger.debug(f"Host ID '{host_id}' 존재 확인됨")
                return True, ""
            else:
                return False, f"Host ID '{host_id}'가 데이터베이스에 존재하지 않습니다"
                
        except Exception as e:
            logger.error(f"Host ID 존재 여부 확인 실패: {e}")
            return False, f"DB 조회 오류: {str(e)}"
    
    def validate_multiple(self, host_ids: List[str], **kwargs) -> ValidationResult:
        """
        여러 Host ID를 일괄 검증합니다.
        
        Args:
            host_ids: 검증할 Host ID 목록
            **kwargs: validate_existence에 전달할 추가 파라미터
            
        Returns:
            ValidationResult: 검증 결과
        """
        logger.info(f"다중 Host ID 검증 시작: {len(host_ids)}개")
        
        valid_items = []
        invalid_items = []
        validation_errors = {}
        
        for host_id in host_ids:
            # 1. 형식 검증
            format_valid, format_error = self.validate_format(host_id)
            if not format_valid:
                invalid_items.append(host_id)
                validation_errors[host_id] = format_error
                continue
            
            # 2. DNS 조회 (활성화된 경우)
            if self.enable_dns_check:
                dns_valid, dns_error = self.validate_dns_resolution(host_id)
                if not dns_valid:
                    invalid_items.append(host_id)
                    validation_errors[host_id] = dns_error
                    continue
            
            # 3. 존재 여부 검증 (DB 연결이 있는 경우)
            if self.db_connection:
                exists_valid, exists_error = self.validate_existence(host_id, **kwargs)
                if not exists_valid:
                    invalid_items.append(host_id)
                    validation_errors[host_id] = exists_error
                    continue
            
            # 모든 검증 통과
            valid_items.append(host_id)
        
        is_valid = len(invalid_items) == 0
        metadata = {
            "total_count": len(host_ids),
            "valid_count": len(valid_items),
            "invalid_count": len(invalid_items),
            "db_check_enabled": self.db_connection is not None,
            "dns_check_enabled": self.enable_dns_check
        }
        
        logger.info(f"Host ID 검증 완료: {len(valid_items)}/{len(host_ids)} 유효")
        return ValidationResult(is_valid, valid_items, invalid_items, validation_errors, metadata)


def to_list(raw_input: Union[str, List[str], None]) -> List[str]:
    """
    다양한 입력 형식을 일관된 문자열 리스트로 변환합니다.
    
    기존 analysis_llm.py의 to_list 함수를 개선한 버전입니다.
    
    Args:
        raw_input: 변환할 입력 (문자열, 리스트, 또는 None)
        
    Returns:
        List[str]: 정규화된 문자열 리스트
    """
    if raw_input is None:
        return []
    
    if isinstance(raw_input, str):
        # 쉼표로 구분된 문자열을 분리
        return [item.strip() for item in raw_input.split(',') if item.strip()]
    
    if isinstance(raw_input, list):
        # 리스트의 각 항목을 문자열로 변환하고 공백 제거
        return [str(item).strip() for item in raw_input if str(item).strip()]
    
    # 기타 타입은 문자열로 변환 후 리스트로 감싸기
    item_str = str(raw_input).strip()
    return [item_str] if item_str else []


def validate_ne_cell_host_filters(
    request: Dict[str, Any],
    db_connection=None,
    enable_dns_check: bool = False,
    **db_params
) -> Tuple[TargetFilters, Dict[str, ValidationResult]]:
    """
    통합 NE/Cell/Host 필터 검증 함수
    
    기존 analysis_llm.py의 필터 처리 로직을 확장하여
    포괄적인 검증을 수행합니다.
    
    Args:
        request: 요청 딕셔너리 (ne, cell/cellid, host 키 포함)
        db_connection: PostgreSQL 연결 객체
        enable_dns_check: Host DNS 조회 활성화 여부
        **db_params: 데이터베이스 테이블/컬럼 설정
        
    Returns:
        Tuple[TargetFilters, Dict[str, ValidationResult]]: 
        (정규화된 필터, 검증 결과)
        
    Raises:
        TargetValidationException: 검증 실패 시
        HostValidationException: Host 검증 실패 시
        RelationshipValidationException: 관계 검증 실패 시
    """
    logger.info("통합 타겟 필터 검증 시작")
    
    # 입력 파라미터 추출 및 정규화
    ne_raw = request.get('ne')
    cell_raw = request.get('cellid') or request.get('cell')
    host_raw = request.get('host')
    
    ne_filters = to_list(ne_raw)
    cellid_filters = to_list(cell_raw)
    host_filters = to_list(host_raw)
    
    logger.info(f"입력 필터 - NE: {len(ne_filters)}, Cell: {len(cellid_filters)}, Host: {len(host_filters)}")
    
    # 검증기 초기화
    table = db_params.get('table', 'summary')
    ne_validator = NEIDValidator(db_connection)
    cell_validator = CellIDValidator(db_connection)
    host_validator = HostValidator(db_connection, enable_dns_check)
    
    # 개별 검증 수행
    validation_results = {}
    
    # 1. NE ID 검증
    if ne_filters:
        ne_result = ne_validator.validate_multiple(
            ne_filters, 
            table=table, 
            ne_column=db_params.get('ne_column', 'ne')
        )
        validation_results['ne'] = ne_result
        
        if not ne_result.is_valid:
            raise_host_validation_error(
                ne_result.invalid_items,
                ne_result.validation_errors,
                f"NE ID 검증 실패: {len(ne_result.invalid_items)}개 항목"
            )
    
    # 2. Cell ID 검증
    if cellid_filters:
        cell_result = cell_validator.validate_multiple(
            cellid_filters,
            table=table,
            cell_column=db_params.get('cell_column', 'cellid')
        )
        validation_results['cell'] = cell_result
        
        if not cell_result.is_valid:
            raise_host_validation_error(
                cell_result.invalid_items,
                cell_result.validation_errors,
                f"Cell ID 검증 실패: {len(cell_result.invalid_items)}개 항목"
            )
    
    # 3. Host ID 검증
    if host_filters:
        host_result = host_validator.validate_multiple(
            host_filters,
            table=table,
            host_column=db_params.get('host_column', 'host')
        )
        validation_results['host'] = host_result
        
        if not host_result.is_valid:
            raise HostValidationException(
                message=f"Host 검증 실패: {len(host_result.invalid_items)}개 항목",
                invalid_hosts=host_result.invalid_items,
                validation_errors=host_result.validation_errors
            )
    
    # 4. 필터 조합 검증
    _validate_filter_combinations(ne_filters, cellid_filters, host_filters)
    
    # 5. 관계 검증 (DB 연결이 있는 경우)
    relationship_mapping = None
    coverage_analysis = None
    if db_connection and any([ne_filters, cellid_filters, host_filters]):
        try:
            # 강화된 관계 검증 사용
            from .relationship_validator import NetworkRelationshipValidator
            
            relationship_validator = NetworkRelationshipValidator(db_connection, table)
            relationship_mapping, coverage_analysis = relationship_validator.validate_comprehensive_relationships(
                ne_filters, cellid_filters, host_filters,
                db_params.get('ne_column', 'ne'),
                db_params.get('cell_column', 'cellid'),
                db_params.get('host_column', 'host')
            )
            logger.info(f"강화된 관계 검증 완료: 커버리지 {coverage_analysis.coverage_ratio:.1%}")
            
        except ImportError:
            logger.warning("강화된 관계 검증 모듈을 로드할 수 없습니다. 기본 검증을 사용합니다.")
            _validate_target_relationships(
                db_connection, ne_filters, cellid_filters, host_filters, **db_params
            )
        except Exception as e:
            logger.error(f"강화된 관계 검증 실패: {e}")
            # 기본 검증으로 폴백
            _validate_target_relationships(
                db_connection, ne_filters, cellid_filters, host_filters, **db_params
            )
    
    # 정규화된 필터 생성
    target_filters = TargetFilters(
        ne_filters=ne_filters if ne_filters else None,
        cellid_filters=cellid_filters if cellid_filters else None,
        host_filters=host_filters if host_filters else None
    )
    
    # 관계 검증 결과를 validation_results에 추가
    if relationship_mapping and coverage_analysis:
        from .relationship_validator import get_relationship_validation_summary
        
        relationship_summary = get_relationship_validation_summary(
            relationship_mapping, coverage_analysis
        )
        validation_results['relationships'] = ValidationResult(
            is_valid=True,  # 관계 검증은 예외 발생하지 않으면 성공
            valid_items=[],
            invalid_items=[],
            validation_errors={},
            metadata=relationship_summary
        )
    
    logger.info("통합 타겟 필터 검증 완료")
    return target_filters, validation_results


def _validate_filter_combinations(
    ne_filters: List[str], 
    cellid_filters: List[str], 
    host_filters: List[str]
) -> None:
    """
    필터 조합의 논리적 유효성을 검증합니다.
    
    Args:
        ne_filters: NE ID 필터 목록
        cellid_filters: Cell ID 필터 목록  
        host_filters: Host ID 필터 목록
        
    Raises:
        FilterCombinationException: 유효하지 않은 조합인 경우
    """
    logger.debug("필터 조합 검증 시작")
    
    # 1. 모든 필터가 비어있는 경우
    if not any([ne_filters, cellid_filters, host_filters]):
        logger.warning("모든 필터가 비어있음 - 전체 데이터 대상 분석")
        return
    
    # 2. 과도한 필터 조합 (성능 고려)
    total_combinations = len(ne_filters or [1]) * len(cellid_filters or [1]) * len(host_filters or [1])
    if total_combinations > 1000:  # 임계값 설정
        conflicting_filters = {
            "ne_count": len(ne_filters),
            "cell_count": len(cellid_filters),
            "host_count": len(host_filters),
            "total_combinations": total_combinations
        }
        raise_filter_combination_error(
            conflicting_filters,
            f"필터 조합이 너무 많습니다: {total_combinations}개 (최대: 1000개)"
        )
    
    # 3. 중복 제거 권장사항 로깅
    for filter_type, filters in [("NE", ne_filters), ("Cell", cellid_filters), ("Host", host_filters)]:
        if filters and len(filters) != len(set(filters)):
            logger.warning(f"{filter_type} 필터에 중복 항목이 있습니다: {filters}")
    
    logger.debug("필터 조합 검증 완료")


def _validate_target_relationships(
    db_connection,
    ne_filters: List[str],
    cellid_filters: List[str], 
    host_filters: List[str],
    **db_params
) -> None:
    """
    데이터베이스에서 NE-Cell-Host 간의 실제 관계를 검증합니다.
    
    Args:
        db_connection: PostgreSQL 연결 객체
        ne_filters: NE ID 필터 목록
        cellid_filters: Cell ID 필터 목록
        host_filters: Host ID 필터 목록
        **db_params: 데이터베이스 테이블/컬럼 설정
        
    Raises:
        RelationshipValidationException: 관계 검증 실패 시
    """
    logger.debug("타겟 관계 검증 시작")
    
    try:
        table = db_params.get('table', 'summary')
        ne_col = db_params.get('ne_column', 'ne')
        cell_col = db_params.get('cell_column', 'cellid')
        host_col = db_params.get('host_column', 'host')
        
        # 존재하는 조합 조회
        conditions = []
        params = []
        
        if ne_filters:
            placeholders = ','.join(['%s'] * len(ne_filters))
            conditions.append(f"{ne_col} IN ({placeholders})")
            params.extend(ne_filters)
        
        if cellid_filters:
            placeholders = ','.join(['%s'] * len(cellid_filters))
            conditions.append(f"{cell_col} IN ({placeholders})")
            params.extend(cellid_filters)
        
        if host_filters:
            placeholders = ','.join(['%s'] * len(host_filters))
            conditions.append(f"{host_col} IN ({placeholders})")
            params.extend(host_filters)
        
        if not conditions:
            logger.debug("관계 검증을 위한 필터가 없음")
            return
        
        sql = f"""
        SELECT DISTINCT {ne_col}, {cell_col}, {host_col}
        FROM {table}
        WHERE {' AND '.join(conditions)}
        LIMIT 1000
        """
        
        with db_connection.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(sql, params)
            existing_combinations = cur.fetchall()
        
        if not existing_combinations:
            missing_combinations = []
            for ne in (ne_filters or [None]):
                for cell in (cellid_filters or [None]):
                    for host in (host_filters or [None]):
                        if ne or cell or host:  # 최소 하나의 필터는 있어야 함
                            combo = {}
                            if ne: combo['ne'] = ne
                            if cell: combo['cell'] = cell
                            if host: combo['host'] = host
                            missing_combinations.append(combo)
            
            raise_relationship_validation_error(
                missing_combinations[:10],  # 최대 10개만 표시
                "지정된 NE-Cell-Host 조합이 데이터베이스에 존재하지 않습니다"
            )
        
        logger.debug(f"타겟 관계 검증 완료: {len(existing_combinations)}개 조합 확인")
        
    except Exception as e:
        if isinstance(e, RelationshipValidationException):
            raise
        logger.error(f"타겟 관계 검증 중 오류 발생: {e}")
        # 관계 검증 실패는 치명적이지 않으므로 경고만 출력
        logger.warning("타겟 관계 검증을 건너뜁니다")


# 편의 함수들

def create_target_validators(db_connection=None, enable_dns_check: bool = False) -> Tuple[NEIDValidator, CellIDValidator, HostValidator]:
    """
    타겟 검증기들을 일괄 생성하는 편의 함수
    
    Args:
        db_connection: PostgreSQL 연결 객체
        enable_dns_check: Host DNS 조회 활성화 여부
        
    Returns:
        Tuple[NEIDValidator, CellIDValidator, HostValidator]: 검증기 튜플
    """
    return (
        NEIDValidator(db_connection),
        CellIDValidator(db_connection), 
        HostValidator(db_connection, enable_dns_check)
    )


def get_validation_summary(validation_results: Dict[str, ValidationResult]) -> Dict[str, Any]:
    """
    검증 결과를 요약하는 편의 함수
    
    Args:
        validation_results: 검증 결과 딕셔너리
        
    Returns:
        Dict[str, Any]: 요약 정보
    """
    summary = {
        "total_validated_types": len(validation_results),
        "all_valid": all(result.is_valid for result in validation_results.values()),
        "validation_details": {}
    }
    
    for target_type, result in validation_results.items():
        summary["validation_details"][target_type] = {
            "is_valid": result.is_valid,
            "valid_count": len(result.valid_items),
            "invalid_count": len(result.invalid_items),
            "total_count": result.metadata.get("total_count", 0)
        }
    
    return summary


if __name__ == "__main__":
    # 단독 실행 시 기본 테스트
    print("Target Validation 유틸리티 모듈 테스트")
    
    # NE ID 검증 테스트
    ne_validator = NEIDValidator()
    test_ne_ids = ["nvgnb#10000", "invalid_ne", "enb#123"]
    
    for ne_id in test_ne_ids:
        is_valid, error = ne_validator.validate_format(ne_id)
        print(f"NE ID '{ne_id}': {'유효' if is_valid else '무효'} - {error}")
    
    # Cell ID 검증 테스트  
    cell_validator = CellIDValidator()
    test_cell_ids = ["2010", 8418, "invalid", -1]
    
    for cell_id in test_cell_ids:
        is_valid, error = cell_validator.validate_format(cell_id)
        print(f"Cell ID '{cell_id}': {'유효' if is_valid else '무효'} - {error}")
    
    # Host ID 검증 테스트
    host_validator = HostValidator()
    test_host_ids = ["192.168.1.1", "host01", "example.com", "invalid@host"]
    
    for host_id in test_host_ids:
        is_valid, error = host_validator.validate_format(host_id)
        print(f"Host ID '{host_id}': {'유효' if is_valid else '무효'} - {error}")
    
    print("테스트 완료!")
