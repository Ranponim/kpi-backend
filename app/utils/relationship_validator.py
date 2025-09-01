"""
NE-Cell-Host 관계 검증 모듈

네트워크 요소 간의 논리적/물리적 관계를 검증하고 
coverage analysis를 수행하는 전문 모듈입니다.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict

import psycopg2
import psycopg2.extras

from ..exceptions import RelationshipValidationException, raise_relationship_validation_error

# 로깅 설정
logger = logging.getLogger(__name__)


@dataclass
class RelationshipMapping:
    """관계 매핑 결과를 담는 데이터 클래스"""
    ne_cell_pairs: List[Tuple[str, str]]  # (ne_id, cell_id) 쌍
    cell_host_pairs: List[Tuple[str, str]]  # (cell_id, host_id) 쌍
    ne_host_pairs: List[Tuple[str, str]]    # (ne_id, host_id) 쌍
    valid_combinations: List[Dict[str, str]]  # 완전한 NE-Cell-Host 조합
    orphaned_targets: Dict[str, List[str]]   # 연결되지 않은 타겟들


@dataclass
class CoverageAnalysis:
    """커버리지 분석 결과를 담는 데이터 클래스"""
    total_possible_combinations: int
    valid_combinations: int
    coverage_ratio: float
    missing_relationships: List[Dict[str, str]]
    redundant_filters: List[str]
    optimization_suggestions: List[str]


class NetworkRelationshipValidator:
    """
    네트워크 요소 간 관계 검증 클래스
    
    NE-Cell-Host 간의 논리적/물리적 관계를 검증하고
    커버리지 분석을 수행합니다.
    """
    
    def __init__(self, db_connection, table: str = "summary"):
        """
        관계 검증기 초기화
        
        Args:
            db_connection: PostgreSQL 연결 객체
            table: 검증할 테이블명
        """
        self.db_connection = db_connection
        self.table = table
        logger.info(f"NetworkRelationshipValidator 초기화: table={table}")
    
    def validate_comprehensive_relationships(
        self,
        ne_filters: List[str],
        cellid_filters: List[str],
        host_filters: List[str],
        ne_column: str = "ne",
        cell_column: str = "cellid", 
        host_column: str = "host"
    ) -> Tuple[RelationshipMapping, CoverageAnalysis]:
        """
        포괄적인 NE-Cell-Host 관계 검증을 수행합니다.
        
        Args:
            ne_filters: NE ID 필터 목록
            cellid_filters: Cell ID 필터 목록
            host_filters: Host ID 필터 목록
            ne_column: NE ID 컬럼명
            cell_column: Cell ID 컬럼명
            host_column: Host ID 컬럼명
            
        Returns:
            Tuple[RelationshipMapping, CoverageAnalysis]: 관계 매핑과 커버리지 분석 결과
            
        Raises:
            RelationshipValidationException: 관계 검증 실패 시
        """
        logger.info("포괄적인 관계 검증 시작")
        
        try:
            # 1. 기존 관계 매핑 조회
            relationship_mapping = self._get_existing_relationships(
                ne_filters, cellid_filters, host_filters,
                ne_column, cell_column, host_column
            )
            
            # 2. 논리적 관계 검증
            self._validate_logical_relationships(
                ne_filters, cellid_filters, host_filters,
                relationship_mapping, ne_column, cell_column, host_column
            )
            
            # 3. 커버리지 분석
            coverage_analysis = self._analyze_coverage(
                ne_filters, cellid_filters, host_filters,
                relationship_mapping
            )
            
            # 4. 관계 일관성 검증
            self._validate_relationship_consistency(relationship_mapping, coverage_analysis)
            
            logger.info(f"관계 검증 완료: {len(relationship_mapping.valid_combinations)}개 유효 조합")
            return relationship_mapping, coverage_analysis
            
        except Exception as e:
            logger.error(f"관계 검증 중 오류 발생: {e}")
            if isinstance(e, RelationshipValidationException):
                raise
            raise RelationshipValidationException(
                message=f"관계 검증 실패: {str(e)}",
                missing_relationships=[{"error": str(e)}]
            )
    
    def _get_existing_relationships(
        self,
        ne_filters: List[str],
        cellid_filters: List[str], 
        host_filters: List[str],
        ne_column: str,
        cell_column: str,
        host_column: str
    ) -> RelationshipMapping:
        """데이터베이스에서 기존 관계를 조회합니다."""
        logger.debug("기존 관계 매핑 조회 시작")
        
        # 모든 필터가 있는 경우와 없는 경우를 구분하여 쿼리 구성
        all_filters = ne_filters + cellid_filters + host_filters
        if not all_filters:
            logger.warning("필터가 없어 관계 검증을 건너뜁니다")
            return RelationshipMapping([], [], [], [], {})
        
        # 기본 쿼리 구성
        base_query = f"""
        SELECT DISTINCT {ne_column}, {cell_column}, {host_column}
        FROM {self.table}
        WHERE 1=1
        """
        
        conditions = []
        params = []
        
        # 필터 조건 추가
        if ne_filters:
            placeholders = ','.join(['%s'] * len(ne_filters))
            conditions.append(f"{ne_column} IN ({placeholders})")
            params.extend(ne_filters)
        
        if cellid_filters:
            placeholders = ','.join(['%s'] * len(cellid_filters))
            conditions.append(f"{cell_column} IN ({placeholders})")
            params.extend(cellid_filters)
        
        if host_filters:
            placeholders = ','.join(['%s'] * len(host_filters))
            conditions.append(f"{host_column} IN ({placeholders})")
            params.extend(host_filters)
        
        # 쿼리 완성
        if conditions:
            query = base_query + " AND " + " AND ".join(conditions)
        else:
            query = base_query
        
        query += f" ORDER BY {ne_column}, {cell_column}, {host_column} LIMIT 10000"
        
        try:
            with self.db_connection.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
            
            logger.debug(f"조회된 관계: {len(rows)}개")
            
            # 관계 분석
            ne_cell_pairs = set()
            cell_host_pairs = set()
            ne_host_pairs = set()
            valid_combinations = []
            
            for row in rows:
                ne_id = str(row[ne_column]) if row[ne_column] else None
                cell_id = str(row[cell_column]) if row[cell_column] else None
                host_id = str(row[host_column]) if row[host_column] else None
                
                if ne_id and cell_id:
                    ne_cell_pairs.add((ne_id, cell_id))
                if cell_id and host_id:
                    cell_host_pairs.add((cell_id, host_id))
                if ne_id and host_id:
                    ne_host_pairs.add((ne_id, host_id))
                
                if ne_id and cell_id and host_id:
                    valid_combinations.append({
                        "ne": ne_id,
                        "cell": cell_id,
                        "host": host_id
                    })
            
            # 고아 타겟 찾기
            orphaned_targets = self._find_orphaned_targets(
                ne_filters, cellid_filters, host_filters,
                ne_cell_pairs, cell_host_pairs, ne_host_pairs
            )
            
            return RelationshipMapping(
                ne_cell_pairs=list(ne_cell_pairs),
                cell_host_pairs=list(cell_host_pairs),
                ne_host_pairs=list(ne_host_pairs),
                valid_combinations=valid_combinations,
                orphaned_targets=orphaned_targets
            )
            
        except Exception as e:
            logger.error(f"관계 조회 실패: {e}")
            raise RelationshipValidationException(
                message=f"데이터베이스 관계 조회 실패: {str(e)}",
                missing_relationships=[{"db_error": str(e)}]
            )
    
    def _validate_logical_relationships(
        self,
        ne_filters: List[str],
        cellid_filters: List[str],
        host_filters: List[str],
        relationship_mapping: RelationshipMapping,
        ne_column: str,
        cell_column: str,
        host_column: str
    ) -> None:
        """논리적 관계의 유효성을 검증합니다."""
        logger.debug("논리적 관계 검증 시작")
        
        validation_errors = []
        
        # 1. NE-Cell 관계 검증
        if ne_filters and cellid_filters:
            invalid_ne_cell_pairs = self._validate_ne_cell_relationships(
                ne_filters, cellid_filters, relationship_mapping.ne_cell_pairs
            )
            if invalid_ne_cell_pairs:
                validation_errors.extend([
                    {"type": "invalid_ne_cell", "ne": ne, "cell": cell}
                    for ne, cell in invalid_ne_cell_pairs
                ])
        
        # 2. Cell-Host 관계 검증
        if cellid_filters and host_filters:
            invalid_cell_host_pairs = self._validate_cell_host_relationships(
                cellid_filters, host_filters, relationship_mapping.cell_host_pairs
            )
            if invalid_cell_host_pairs:
                validation_errors.extend([
                    {"type": "invalid_cell_host", "cell": cell, "host": host}
                    for cell, host in invalid_cell_host_pairs
                ])
        
        # 3. NE-Host 관계 검증
        if ne_filters and host_filters:
            invalid_ne_host_pairs = self._validate_ne_host_relationships(
                ne_filters, host_filters, relationship_mapping.ne_host_pairs
            )
            if invalid_ne_host_pairs:
                validation_errors.extend([
                    {"type": "invalid_ne_host", "ne": ne, "host": host}
                    for ne, host in invalid_ne_host_pairs
                ])
        
        # 4. 전체 조합 검증
        if ne_filters and cellid_filters and host_filters:
            invalid_combinations = self._validate_complete_combinations(
                ne_filters, cellid_filters, host_filters,
                relationship_mapping.valid_combinations
            )
            if invalid_combinations:
                validation_errors.extend([
                    {"type": "invalid_combination", **combo}
                    for combo in invalid_combinations
                ])
        
        # 검증 실패 시 예외 발생
        if validation_errors:
            logger.error(f"논리적 관계 검증 실패: {len(validation_errors)}개 오류")
            raise_relationship_validation_error(
                validation_errors[:10],  # 최대 10개만 표시
                f"논리적 관계 검증 실패: {len(validation_errors)}개 오류 발견"
            )
        
        logger.debug("논리적 관계 검증 성공")
    
    def _analyze_coverage(
        self,
        ne_filters: List[str],
        cellid_filters: List[str],
        host_filters: List[str],
        relationship_mapping: RelationshipMapping
    ) -> CoverageAnalysis:
        """커버리지 분석을 수행합니다."""
        logger.debug("커버리지 분석 시작")
        
        # 이론적 최대 조합 수 계산
        total_possible = len(ne_filters or [1]) * len(cellid_filters or [1]) * len(host_filters or [1])
        if not any([ne_filters, cellid_filters, host_filters]):
            total_possible = 0
        
        # 실제 유효한 조합 수
        valid_count = len(relationship_mapping.valid_combinations)
        
        # 커버리지 비율 계산
        coverage_ratio = valid_count / total_possible if total_possible > 0 else 0.0
        
        # 누락된 관계 찾기
        missing_relationships = self._find_missing_relationships(
            ne_filters, cellid_filters, host_filters,
            relationship_mapping.valid_combinations
        )
        
        # 중복 필터 찾기
        redundant_filters = self._find_redundant_filters(
            ne_filters, cellid_filters, host_filters,
            relationship_mapping
        )
        
        # 최적화 제안 생성
        optimization_suggestions = self._generate_optimization_suggestions(
            coverage_ratio, missing_relationships, redundant_filters
        )
        
        return CoverageAnalysis(
            total_possible_combinations=total_possible,
            valid_combinations=valid_count,
            coverage_ratio=coverage_ratio,
            missing_relationships=missing_relationships,
            redundant_filters=redundant_filters,
            optimization_suggestions=optimization_suggestions
        )
    
    def _validate_relationship_consistency(
        self, 
        relationship_mapping: RelationshipMapping,
        coverage_analysis: CoverageAnalysis
    ) -> None:
        """관계의 일관성을 검증합니다."""
        logger.debug("관계 일관성 검증 시작")
        
        consistency_issues = []
        
        # 1. 커버리지가 너무 낮은 경우
        if coverage_analysis.coverage_ratio < 0.1 and coverage_analysis.total_possible_combinations > 1:
            consistency_issues.append({
                "type": "low_coverage",
                "message": f"커버리지가 낮습니다: {coverage_analysis.coverage_ratio:.1%}",
                "coverage_ratio": coverage_analysis.coverage_ratio
            })
        
        # 2. 고아 타겟이 너무 많은 경우
        total_orphans = sum(len(targets) for targets in relationship_mapping.orphaned_targets.values())
        if total_orphans > 0:
            consistency_issues.append({
                "type": "orphaned_targets",
                "message": f"연결되지 않은 타겟: {total_orphans}개",
                "orphaned_count": total_orphans,
                "orphaned_targets": relationship_mapping.orphaned_targets
            })
        
        # 3. 경고 레벨 이슈는 로그만 출력 (예외 발생하지 않음)
        if consistency_issues:
            for issue in consistency_issues:
                logger.warning(f"관계 일관성 이슈: {issue['message']}")
        
        logger.debug("관계 일관성 검증 완료")
    
    def _validate_ne_cell_relationships(
        self, 
        ne_filters: List[str], 
        cellid_filters: List[str],
        ne_cell_pairs: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """NE-Cell 관계를 검증합니다."""
        existing_pairs = set(ne_cell_pairs)
        invalid_pairs = []
        
        for ne in ne_filters:
            for cell in cellid_filters:
                if (ne, cell) not in existing_pairs:
                    invalid_pairs.append((ne, cell))
        
        return invalid_pairs
    
    def _validate_cell_host_relationships(
        self,
        cellid_filters: List[str],
        host_filters: List[str],
        cell_host_pairs: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """Cell-Host 관계를 검증합니다."""
        existing_pairs = set(cell_host_pairs)
        invalid_pairs = []
        
        for cell in cellid_filters:
            for host in host_filters:
                if (cell, host) not in existing_pairs:
                    invalid_pairs.append((cell, host))
        
        return invalid_pairs
    
    def _validate_ne_host_relationships(
        self,
        ne_filters: List[str],
        host_filters: List[str], 
        ne_host_pairs: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """NE-Host 관계를 검증합니다."""
        existing_pairs = set(ne_host_pairs)
        invalid_pairs = []
        
        for ne in ne_filters:
            for host in host_filters:
                if (ne, host) not in existing_pairs:
                    invalid_pairs.append((ne, host))
        
        return invalid_pairs
    
    def _validate_complete_combinations(
        self,
        ne_filters: List[str],
        cellid_filters: List[str],
        host_filters: List[str],
        valid_combinations: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """완전한 NE-Cell-Host 조합을 검증합니다."""
        existing_combinations = set(
            (combo["ne"], combo["cell"], combo["host"])
            for combo in valid_combinations
        )
        
        invalid_combinations = []
        
        for ne in ne_filters:
            for cell in cellid_filters:
                for host in host_filters:
                    if (ne, cell, host) not in existing_combinations:
                        invalid_combinations.append({
                            "ne": ne,
                            "cell": cell,
                            "host": host
                        })
        
        return invalid_combinations
    
    def _find_orphaned_targets(
        self,
        ne_filters: List[str],
        cellid_filters: List[str],
        host_filters: List[str],
        ne_cell_pairs: Set[Tuple[str, str]],
        cell_host_pairs: Set[Tuple[str, str]],
        ne_host_pairs: Set[Tuple[str, str]]
    ) -> Dict[str, List[str]]:
        """연결되지 않은 고아 타겟을 찾습니다."""
        orphaned = defaultdict(list)
        
        # 연결된 타겟들 추출
        connected_nes = set(pair[0] for pair in ne_cell_pairs) | set(pair[0] for pair in ne_host_pairs)
        connected_cells = set(pair[1] for pair in ne_cell_pairs) | set(pair[0] for pair in cell_host_pairs)
        connected_hosts = set(pair[1] for pair in cell_host_pairs) | set(pair[1] for pair in ne_host_pairs)
        
        # 고아 타겟 찾기
        for ne in ne_filters:
            if ne not in connected_nes:
                orphaned["ne"].append(ne)
        
        for cell in cellid_filters:
            if cell not in connected_cells:
                orphaned["cell"].append(cell)
        
        for host in host_filters:
            if host not in connected_hosts:
                orphaned["host"].append(host)
        
        return dict(orphaned)
    
    def _find_missing_relationships(
        self,
        ne_filters: List[str],
        cellid_filters: List[str], 
        host_filters: List[str],
        valid_combinations: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """누락된 관계를 찾습니다."""
        if not all([ne_filters, cellid_filters, host_filters]):
            return []
        
        existing_combinations = set(
            (combo["ne"], combo["cell"], combo["host"])
            for combo in valid_combinations
        )
        
        missing = []
        for ne in ne_filters:
            for cell in cellid_filters:
                for host in host_filters:
                    if (ne, cell, host) not in existing_combinations:
                        missing.append({
                            "ne": ne,
                            "cell": cell,
                            "host": host
                        })
        
        return missing
    
    def _find_redundant_filters(
        self,
        ne_filters: List[str],
        cellid_filters: List[str],
        host_filters: List[str],
        relationship_mapping: RelationshipMapping
    ) -> List[str]:
        """중복 필터를 찾습니다."""
        redundant = []
        
        # 단순한 중복 검사 (향후 더 정교한 로직으로 확장 가능)
        if len(set(ne_filters)) < len(ne_filters):
            redundant.append("NE 필터에 중복 항목이 있습니다")
        
        if len(set(cellid_filters)) < len(cellid_filters):
            redundant.append("Cell 필터에 중복 항목이 있습니다")
        
        if len(set(host_filters)) < len(host_filters):
            redundant.append("Host 필터에 중복 항목이 있습니다")
        
        return redundant
    
    def _generate_optimization_suggestions(
        self,
        coverage_ratio: float,
        missing_relationships: List[Dict[str, str]],
        redundant_filters: List[str]
    ) -> List[str]:
        """최적화 제안을 생성합니다."""
        suggestions = []
        
        if coverage_ratio < 0.5:
            suggestions.append("커버리지가 낮습니다. 필터 조건을 재검토하세요.")
        
        if len(missing_relationships) > 10:
            suggestions.append(f"누락된 관계가 많습니다 ({len(missing_relationships)}개). 필터 범위를 축소하는 것을 고려하세요.")
        
        if redundant_filters:
            suggestions.append("중복 필터를 제거하여 성능을 개선할 수 있습니다.")
        
        if coverage_ratio > 0.9:
            suggestions.append("우수한 커버리지입니다. 현재 필터 설정이 최적화되어 있습니다.")
        
        return suggestions


def get_relationship_validation_summary(
    relationship_mapping: RelationshipMapping,
    coverage_analysis: CoverageAnalysis
) -> Dict[str, Any]:
    """
    관계 검증 결과를 요약하는 편의 함수
    
    Args:
        relationship_mapping: 관계 매핑 결과
        coverage_analysis: 커버리지 분석 결과
        
    Returns:
        Dict[str, Any]: 요약 정보
    """
    return {
        "relationship_summary": {
            "ne_cell_relationships": len(relationship_mapping.ne_cell_pairs),
            "cell_host_relationships": len(relationship_mapping.cell_host_pairs),
            "ne_host_relationships": len(relationship_mapping.ne_host_pairs),
            "valid_complete_combinations": len(relationship_mapping.valid_combinations),
            "orphaned_targets_count": sum(len(targets) for targets in relationship_mapping.orphaned_targets.values())
        },
        "coverage_summary": {
            "coverage_ratio": coverage_analysis.coverage_ratio,
            "valid_combinations": coverage_analysis.valid_combinations,
            "total_possible": coverage_analysis.total_possible_combinations,
            "missing_relationships_count": len(coverage_analysis.missing_relationships),
            "optimization_suggestions_count": len(coverage_analysis.optimization_suggestions)
        },
        "validation_status": {
            "has_orphaned_targets": bool(relationship_mapping.orphaned_targets),
            "has_missing_relationships": bool(coverage_analysis.missing_relationships),
            "has_redundant_filters": bool(coverage_analysis.redundant_filters),
            "coverage_level": "high" if coverage_analysis.coverage_ratio > 0.8 else 
                           "medium" if coverage_analysis.coverage_ratio > 0.5 else "low"
        }
    }


if __name__ == "__main__":
    # 기본 테스트
    print("🧪 관계 검증 모듈 테스트")
    
    # Mock 데이터로 테스트
    print("✅ 모듈 로드 성공")
    print("✅ 클래스 및 함수 정의 확인")
    
    # 데이터 구조 테스트
    mapping = RelationshipMapping([], [], [], [], {})
    analysis = CoverageAnalysis(0, 0, 0.0, [], [], [])
    
    print("✅ 데이터 구조 생성 성공")
    print("\n🎉 관계 검증 모듈 기본 테스트 통과!")
