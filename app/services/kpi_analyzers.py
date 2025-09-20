"""
KPI 분석 규칙 구현 (5장)

이 모듈은 SOLID 원칙을 완벽히 준수하여 각 KPI 분석 규칙을
독립적인 클래스로 구현합니다.

SOLID 원칙 적용:
- Single Responsibility: 각 분석기는 하나의 판정 규칙만 담당
- Open/Closed: 새로운 판정 규칙 추가 시 기존 코드 수정 없음
- Liskov Substitution: 모든 분석기는 동일한 인터페이스 구현
- Interface Segregation: 각 분석기는 필요한 메서드만 구현
- Dependency Inversion: 추상화에 의존, 구체 클래스에 의존하지 않음

PRD 참조: 섹션 2.3 (통계 분석 알고리즘)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Protocol, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import numpy as np

from ..models.judgement import (
    PegSampleSeries,
    PegPeriodStats,
    PegCompareMetrics,
    PegCompareDecision,
    JudgementType,
    CompareDetail,
    KPIPositivity
)
from ..utils.logging_decorators import log_analyzer_execution
from ..exceptions import KPIAnalysisError, InsufficientDataError

logger = logging.getLogger(__name__)


# =============================================================================
# KPI 분석 결과 데이터 클래스
# =============================================================================

@dataclass(frozen=True)
class KPIAnalysisResult:
    """
    KPI 분석 결과 (불변 객체)
    
    각 분석기의 결과를 표현하는 불변 데이터 클래스
    """
    judgement_type: JudgementType
    compare_detail: CompareDetail
    reasoning: str
    confidence: float
    metrics: Dict[str, Any]
    thresholds_used: Dict[str, float]
    
    def __post_init__(self):
        """후처리 검증"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")


class AnalysisRulePriority(Enum):
    """
    분석 규칙 우선순위 (5장)
    
    높은 값이 높은 우선순위
    """
    CANT_JUDGE = 100
    HIGH_VARIATION = 90
    IMPROVE = 80
    DEGRADE = 80
    HIGH_DELTA = 70
    MEDIUM_DELTA = 60
    LOW_DELTA = 50
    SIMILAR = 40


# =============================================================================
# KPI 분석기 인터페이스 (Protocol 사용)
# =============================================================================

class KPIAnalyzer(Protocol):
    """
    KPI 분석기 프로토콜
    
    Protocol을 사용하여 덕 타이핑 지원 및 더 유연한 인터페이스 제공
    """
    
    def analyze(self, 
                pre_stats: PegPeriodStats,
                post_stats: PegPeriodStats,
                compare_metrics: PegCompareMetrics,
                config: Dict[str, Any]) -> Optional[KPIAnalysisResult]:
        """
        KPI 분석 실행
        
        Args:
            pre_stats: Pre 기간 통계
            post_stats: Post 기간 통계
            compare_metrics: 비교 지표
            config: 분석 설정
            
        Returns:
            Optional[KPIAnalysisResult]: 분석 결과 (해당 없으면 None)
        """
        ...
    
    def get_priority(self) -> int:
        """분석 규칙 우선순위 반환"""
        ...
    
    def get_analyzer_info(self) -> Dict[str, Any]:
        """분석기 정보 반환"""
        ...


# =============================================================================
# 추상 기본 분석기 클래스
# =============================================================================

class BaseKPIAnalyzer(ABC):
    """
    KPI 분석기 기본 추상 클래스
    
    공통 기능과 템플릿 메서드 패턴을 제공합니다.
    """
    
    def __init__(self, analyzer_name: str, priority: AnalysisRulePriority, version: str = "1.0.0"):
        """
        기본 분석기 초기화
        
        Args:
            analyzer_name: 분석기 이름
            priority: 규칙 우선순위
            version: 버전
        """
        self.analyzer_name = analyzer_name
        self.priority = priority
        self.version = version
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.logger.info(f"KPI analyzer '{analyzer_name}' v{version} initialized (priority: {priority.value})")
    
    @log_analyzer_execution()
    def analyze(self, 
                pre_stats: PegPeriodStats,
                post_stats: PegPeriodStats,
                compare_metrics: PegCompareMetrics,
                config: Dict[str, Any]) -> Optional[KPIAnalysisResult]:
        """
        템플릿 메서드: KPI 분석 실행
        
        Args:
            pre_stats: Pre 기간 통계
            post_stats: Post 기간 통계
            compare_metrics: 비교 지표
            config: 분석 설정
            
        Returns:
            Optional[KPIAnalysisResult]: 분석 결과
        """
        try:
            # 1. 설정 검증
            if not self._validate_config(config):
                raise ValueError(f"Invalid config for {self.analyzer_name}")
            
            # 2. 입력 데이터 검증
            if not self._validate_input_data(pre_stats, post_stats, compare_metrics):
                self.logger.debug(f"{self.analyzer_name}: Input validation failed, skipping analysis")
                return None
            
            # 3. 규칙 적용 가능성 확인
            if not self._is_rule_applicable(pre_stats, post_stats, compare_metrics, config):
                self.logger.debug(f"{self.analyzer_name}: Rule not applicable, skipping")
                return None
            
            # 4. 실제 분석 로직 실행 (하위 클래스에서 구현)
            self.logger.debug(f"Executing {self.analyzer_name} analysis")
            analysis_result = self._execute_analysis(pre_stats, post_stats, compare_metrics, config)
            
            # 5. 결과 검증
            if analysis_result and not self._validate_result(analysis_result):
                raise RuntimeError(f"Invalid analysis result from {self.analyzer_name}")
            
            if analysis_result:
                self.logger.info(f"{self.analyzer_name} analysis completed: "
                               f"{analysis_result.judgement_type} ({analysis_result.compare_detail})")
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error in {self.analyzer_name} analysis: {e}")
            raise
    
    @abstractmethod
    def _execute_analysis(self, 
                         pre_stats: PegPeriodStats,
                         post_stats: PegPeriodStats,
                         compare_metrics: PegCompareMetrics,
                         config: Dict[str, Any]) -> Optional[KPIAnalysisResult]:
        """
        실제 분석 로직 구현 (하위 클래스에서 구현)
        
        Args:
            pre_stats: Pre 기간 통계
            post_stats: Post 기간 통계
            compare_metrics: 비교 지표
            config: 분석 설정
            
        Returns:
            Optional[KPIAnalysisResult]: 분석 결과
        """
        pass
    
    @abstractmethod
    def _is_rule_applicable(self, 
                           pre_stats: PegPeriodStats,
                           post_stats: PegPeriodStats,
                           compare_metrics: PegCompareMetrics,
                           config: Dict[str, Any]) -> bool:
        """
        규칙 적용 가능성 확인 (하위 클래스에서 구현)
        
        Args:
            pre_stats: Pre 기간 통계
            post_stats: Post 기간 통계
            compare_metrics: 비교 지표
            config: 분석 설정
            
        Returns:
            bool: 규칙 적용 가능 여부
        """
        pass
    
    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """
        기본 설정 검증
        
        Args:
            config: 검증할 설정
            
        Returns:
            bool: 유효성 여부
        """
        try:
            if not isinstance(config, dict):
                self.logger.error("Config must be a dictionary")
                return False
            
            # 하위 클래스에서 추가 검증 수행
            return self._validate_specific_config(config)
            
        except Exception as e:
            self.logger.error(f"Config validation error: {e}")
            return False
    
    @abstractmethod
    def _validate_specific_config(self, config: Dict[str, Any]) -> bool:
        """하위 클래스별 특화 설정 검증"""
        pass
    
    def _validate_input_data(self, 
                            pre_stats: PegPeriodStats,
                            post_stats: PegPeriodStats,
                            compare_metrics: PegCompareMetrics) -> bool:
        """
        입력 데이터 검증
        
        Args:
            pre_stats: Pre 기간 통계
            post_stats: Post 기간 통계
            compare_metrics: 비교 지표
            
        Returns:
            bool: 유효성 여부
        """
        try:
            if not isinstance(pre_stats, PegPeriodStats):
                self.logger.error("pre_stats must be PegPeriodStats instance")
                return False
            
            if not isinstance(post_stats, PegPeriodStats):
                self.logger.error("post_stats must be PegPeriodStats instance")
                return False
            
            if not isinstance(compare_metrics, PegCompareMetrics):
                self.logger.error("compare_metrics must be PegCompareMetrics instance")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Input data validation error: {e}")
            return False
    
    def _validate_result(self, result: KPIAnalysisResult) -> bool:
        """
        결과 검증
        
        Args:
            result: 검증할 결과
            
        Returns:
            bool: 유효성 여부
        """
        try:
            if not isinstance(result, KPIAnalysisResult):
                self.logger.error("Result must be KPIAnalysisResult instance")
                return False
            
            if not result.reasoning:
                self.logger.error("Reasoning cannot be empty")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Result validation error: {e}")
            return False
    
    def get_priority(self) -> int:
        """분석 규칙 우선순위 반환"""
        return self.priority.value
    
    def get_analyzer_info(self) -> Dict[str, Any]:
        """
        분석기 정보 반환
        
        Returns:
            Dict[str, Any]: 분석기 메타데이터
        """
        return {
            "name": self.analyzer_name,
            "version": self.version,
            "priority": self.priority.value,
            "type": "kpi_analyzer",
            "description": f"KPI analyzer implementation: {self.analyzer_name}"
        }


# =============================================================================
# Can't Judge 분석기 (최고 우선순위)
# =============================================================================

class CantJudgeAnalyzer(BaseKPIAnalyzer):
    """
    Can't Judge 분석기
    
    pre 또는 post 데이터에 ND가 포함된 경우 판정 불가로 처리합니다.
    
    Single Responsibility: Can't Judge 조건 검사만 담당
    """
    
    def __init__(self):
        """Can't Judge 분석기 초기화"""
        super().__init__("CantJudgeAnalyzer", AnalysisRulePriority.CANT_JUDGE)
    
    def _is_rule_applicable(self, 
                           pre_stats: PegPeriodStats,
                           post_stats: PegPeriodStats,
                           compare_metrics: PegCompareMetrics,
                           config: Dict[str, Any]) -> bool:
        """
        Can't Judge 규칙 적용 가능성 확인
        
        ND가 포함된 경우에만 적용 가능
        """
        return compare_metrics.has_nd or pre_stats.nd_ratio > 0 or post_stats.nd_ratio > 0
    
    def _execute_analysis(self, 
                         pre_stats: PegPeriodStats,
                         post_stats: PegPeriodStats,
                         compare_metrics: PegCompareMetrics,
                         config: Dict[str, Any]) -> Optional[KPIAnalysisResult]:
        """
        Can't Judge 분석 실행
        
        Args:
            pre_stats: Pre 기간 통계
            post_stats: Post 기간 통계
            compare_metrics: 비교 지표
            config: 분석 설정
            
        Returns:
            Optional[KPIAnalysisResult]: Can't Judge 분석 결과
        """
        try:
            # ND 상세 분석
            nd_analysis = self._analyze_nd_details(pre_stats, post_stats)
            
            reasoning = self._generate_cant_judge_reasoning(nd_analysis)
            
            return KPIAnalysisResult(
                judgement_type=JudgementType.CANT_JUDGE,
                compare_detail=CompareDetail.CANT_JUDGE,
                reasoning=reasoning,
                confidence=1.0,  # ND 존재는 확실한 조건
                metrics=nd_analysis,
                thresholds_used={}
            )
            
        except Exception as e:
            self.logger.error(f"Can't Judge analysis error: {e}")
            raise
    
    def _analyze_nd_details(self, pre_stats: PegPeriodStats, post_stats: PegPeriodStats) -> Dict[str, Any]:
        """
        ND 상세 분석
        
        Args:
            pre_stats: Pre 기간 통계
            post_stats: Post 기간 통계
            
        Returns:
            Dict[str, Any]: ND 분석 세부 정보
        """
        return {
            "pre_nd_ratio": pre_stats.nd_ratio,
            "post_nd_ratio": post_stats.nd_ratio,
            "pre_sample_count": pre_stats.sample_count,
            "post_sample_count": post_stats.sample_count,
            "nd_pattern": self._determine_nd_pattern(pre_stats.nd_ratio, post_stats.nd_ratio)
        }
    
    def _determine_nd_pattern(self, pre_nd_ratio: float, post_nd_ratio: float) -> str:
        """ND 패턴 결정"""
        if pre_nd_ratio > 0 and post_nd_ratio > 0:
            return "both_periods"
        elif pre_nd_ratio > 0:
            return "pre_only"
        elif post_nd_ratio > 0:
            return "post_only"
        else:
            return "none"
    
    def _generate_cant_judge_reasoning(self, nd_analysis: Dict[str, Any]) -> str:
        """Can't Judge 판정 근거 생성"""
        pattern = nd_analysis["nd_pattern"]
        pre_ratio = nd_analysis["pre_nd_ratio"]
        post_ratio = nd_analysis["post_nd_ratio"]
        
        if pattern == "both_periods":
            return f"판정 불가: Pre({pre_ratio:.1%}) 및 Post({post_ratio:.1%}) 기간 모두 ND 포함"
        elif pattern == "pre_only":
            return f"판정 불가: Pre 기간에 ND 포함 ({pre_ratio:.1%})"
        elif pattern == "post_only":
            return f"판정 불가: Post 기간에 ND 포함 ({post_ratio:.1%})"
        else:
            return "판정 불가: 데이터 품질 문제"
    
    def _validate_specific_config(self, config: Dict[str, Any]) -> bool:
        """Can't Judge 분석기 특화 설정 검증"""
        # Can't Judge는 추가 설정 불필요
        return True


# =============================================================================
# High Variation 분석기
# =============================================================================

class HighVariationAnalyzer(BaseKPIAnalyzer):
    """
    High Variation 분석기
    
    CV(pre) > β4 또는 CV(post) > β4 조건을 검사합니다.
    
    Single Responsibility: High Variation 조건 검사만 담당
    """
    
    def __init__(self):
        """High Variation 분석기 초기화"""
        super().__init__("HighVariationAnalyzer", AnalysisRulePriority.HIGH_VARIATION)
    
    def _is_rule_applicable(self, 
                           pre_stats: PegPeriodStats,
                           post_stats: PegPeriodStats,
                           compare_metrics: PegCompareMetrics,
                           config: Dict[str, Any]) -> bool:
        """
        High Variation 규칙 적용 가능성 확인
        
        CV 계산이 가능한 경우에만 적용 가능
        """
        return (pre_stats.cv is not None or post_stats.cv is not None) and not compare_metrics.has_nd
    
    def _execute_analysis(self, 
                         pre_stats: PegPeriodStats,
                         post_stats: PegPeriodStats,
                         compare_metrics: PegCompareMetrics,
                         config: Dict[str, Any]) -> Optional[KPIAnalysisResult]:
        """
        High Variation 분석 실행
        
        Args:
            pre_stats: Pre 기간 통계
            post_stats: Post 기간 통계
            compare_metrics: 비교 지표
            config: 분석 설정
            
        Returns:
            Optional[KPIAnalysisResult]: High Variation 분석 결과
        """
        try:
            beta_4 = config.get("beta_4", 10.0)
            
            # CV 분석
            cv_analysis = self._analyze_coefficient_of_variation(pre_stats, post_stats, beta_4)
            
            if cv_analysis["is_high_variation"]:
                reasoning = self._generate_high_variation_reasoning(cv_analysis, beta_4)
                
                return KPIAnalysisResult(
                    judgement_type=JudgementType.CANT_JUDGE,  # High Variation은 판정 불가로 처리
                    compare_detail=CompareDetail.HIGH_VARIATION,
                    reasoning=reasoning,
                    confidence=1.0,  # CV 계산은 확실함
                    metrics=cv_analysis,
                    thresholds_used={"beta_4": beta_4}
                )
            
            return None  # High Variation이 아님
            
        except Exception as e:
            self.logger.error(f"High Variation analysis error: {e}")
            raise
    
    def _analyze_coefficient_of_variation(self, 
                                        pre_stats: PegPeriodStats, 
                                        post_stats: PegPeriodStats, 
                                        beta_4: float) -> Dict[str, Any]:
        """
        변동계수 분석
        
        Args:
            pre_stats: Pre 기간 통계
            post_stats: Post 기간 통계
            beta_4: CV 임계값
            
        Returns:
            Dict[str, Any]: CV 분석 결과
        """
        try:
            # Pre 기간 CV 분석
            pre_cv = pre_stats.cv if pre_stats.cv is not None else 0.0
            pre_exceeds = pre_cv > beta_4
            
            # Post 기간 CV 분석
            post_cv = post_stats.cv if post_stats.cv is not None else 0.0
            post_exceeds = post_cv > beta_4
            
            # High Variation 판정
            is_high_variation = pre_exceeds or post_exceeds
            
            # 특수 케이스 처리 (5.2 High Variation 2-6항)
            special_cases = self._check_special_variation_cases(pre_stats, post_stats)
            
            return {
                "pre_cv": pre_cv,
                "post_cv": post_cv,
                "beta_4_threshold": beta_4,
                "pre_exceeds_threshold": pre_exceeds,
                "post_exceeds_threshold": post_exceeds,
                "is_high_variation": is_high_variation or special_cases["has_special_case"],
                "special_cases": special_cases,
                "max_cv": max(pre_cv, post_cv),
                "cv_ratio": max(pre_cv, post_cv) / beta_4 if beta_4 > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"CV analysis error: {e}")
            return {"is_high_variation": False, "error": str(e)}
    
    def _check_special_variation_cases(self, pre_stats: PegPeriodStats, post_stats: PegPeriodStats) -> Dict[str, Any]:
        """
        High Variation 특수 케이스 검사 (5.2 항목 2-6)
        
        Args:
            pre_stats: Pre 기간 통계
            post_stats: Post 기간 통계
            
        Returns:
            Dict[str, Any]: 특수 케이스 분석 결과
        """
        special_cases = []
        
        # 2. Pre가 ND가 아니고 Post가 ND인 경우 δ = -100%
        if pre_stats.nd_ratio == 0 and post_stats.nd_ratio > 0:
            special_cases.append("pre_valid_post_nd")
        
        # 3. Pre가 ND이고 Post가 ND가 아닌 경우 δ = 100%
        if pre_stats.nd_ratio > 0 and post_stats.nd_ratio == 0:
            special_cases.append("pre_nd_post_valid")
        
        # 4-5. Zero 관련 특수 케이스 (% 통계가 아닌 경우)
        if pre_stats.zero_ratio == 0 and post_stats.zero_ratio > 0:
            special_cases.append("pre_nonzero_post_zero")
        
        if pre_stats.zero_ratio > 0 and post_stats.zero_ratio == 0:
            special_cases.append("pre_zero_post_nonzero")
        
        # 6. 일부 샘플이 ND인 경우
        if 0 < pre_stats.nd_ratio < 1 or 0 < post_stats.nd_ratio < 1:
            special_cases.append("partial_nd_samples")
        
        return {
            "has_special_case": len(special_cases) > 0,
            "cases": special_cases,
            "case_count": len(special_cases)
        }
    
    def _generate_high_variation_reasoning(self, cv_analysis: Dict[str, Any], beta_4: float) -> str:
        """High Variation 판정 근거 생성"""
        pre_cv = cv_analysis["pre_cv"]
        post_cv = cv_analysis["post_cv"]
        special_cases = cv_analysis["special_cases"]
        
        if special_cases["has_special_case"]:
            case_descriptions = {
                "pre_valid_post_nd": "Post 기간이 ND (δ = -100%)",
                "pre_nd_post_valid": "Pre 기간이 ND (δ = 100%)", 
                "pre_nonzero_post_zero": "Post 기간이 0 (δ = -100%)",
                "pre_zero_post_nonzero": "Pre 기간이 0 (δ = 100%)",
                "partial_nd_samples": "일부 샘플이 ND"
            }
            
            case_strs = [case_descriptions.get(case, case) for case in special_cases["cases"]]
            return f"High Variation: {', '.join(case_strs)}"
        
        if cv_analysis["pre_exceeds_threshold"] and cv_analysis["post_exceeds_threshold"]:
            return f"High Variation: Pre CV({pre_cv:.1f}) 및 Post CV({post_cv:.1f}) > β4({beta_4})"
        elif cv_analysis["pre_exceeds_threshold"]:
            return f"High Variation: Pre CV({pre_cv:.1f}) > β4({beta_4})"
        elif cv_analysis["post_exceeds_threshold"]:
            return f"High Variation: Post CV({post_cv:.1f}) > β4({beta_4})"
        else:
            return f"High Variation: 특수 조건 만족"
    
    def _validate_specific_config(self, config: Dict[str, Any]) -> bool:
        """Can't Judge 분석기 특화 설정 검증"""
        beta_4 = config.get("beta_4")
        if beta_4 is None:
            self.logger.error("beta_4 threshold is required for High Variation detection")
            return False
        
        if not isinstance(beta_4, (int, float)) or beta_4 <= 0:
            self.logger.error(f"beta_4 must be a positive number, got {beta_4}")
            return False
        
        return True


# =============================================================================
# Improve/Degrade 분석기
# =============================================================================

class ImproveAnalyzer(BaseKPIAnalyzer):
    """
    Improve 분석기
    
    KPI 극성에 따라 성능 개선을 탐지합니다.
    - Positive KPI: max(pre) < min(post)
    - Negative KPI: min(pre) > max(post)
    
    Single Responsibility: 성능 개선 검사만 담당
    """
    
    def __init__(self):
        """Improve 분석기 초기화"""
        super().__init__("ImproveAnalyzer", AnalysisRulePriority.IMPROVE)
    
    def _is_rule_applicable(self, 
                           pre_stats: PegPeriodStats,
                           post_stats: PegPeriodStats,
                           compare_metrics: PegCompareMetrics,
                           config: Dict[str, Any]) -> bool:
        """
        Improve 규칙 적용 가능성 확인
        
        min/max 값이 있고 ND가 없는 경우에만 적용 가능
        """
        return (not compare_metrics.has_nd and 
                pre_stats.min is not None and pre_stats.max is not None and
                post_stats.min is not None and post_stats.max is not None)
    
    def _execute_analysis(self, 
                         pre_stats: PegPeriodStats,
                         post_stats: PegPeriodStats,
                         compare_metrics: PegCompareMetrics,
                         config: Dict[str, Any]) -> Optional[KPIAnalysisResult]:
        """
        Improve 분석 실행
        
        Args:
            pre_stats: Pre 기간 통계
            post_stats: Post 기간 통계
            compare_metrics: 비교 지표
            config: 분석 설정
            
        Returns:
            Optional[KPIAnalysisResult]: Improve 분석 결과
        """
        try:
            # KPI 극성 확인 (설정에서 가져와야 함, 현재는 기본값)
            kpi_positivity = config.get("kpi_positivity", "positive")
            
            # 분포 비교 분석
            distribution_analysis = self._analyze_distribution_separation(
                pre_stats, post_stats, kpi_positivity
            )
            
            if distribution_analysis["is_improve"]:
                reasoning = self._generate_improve_reasoning(
                    distribution_analysis, kpi_positivity
                )
                
                return KPIAnalysisResult(
                    judgement_type=JudgementType.NOK,  # Improve는 NOK로 분류 (PLM 검증 필요)
                    compare_detail=CompareDetail.IMPROVE,
                    reasoning=reasoning,
                    confidence=0.9,  # 분포 기반 분석의 높은 신뢰도
                    metrics=distribution_analysis,
                    thresholds_used={}
                )
            
            return None  # Improve가 아님
            
        except Exception as e:
            self.logger.error(f"Improve analysis error: {e}")
            raise
    
    def _analyze_distribution_separation(self, 
                                       pre_stats: PegPeriodStats, 
                                       post_stats: PegPeriodStats, 
                                       kpi_positivity: str) -> Dict[str, Any]:
        """
        분포 분리 분석
        
        Args:
            pre_stats: Pre 기간 통계
            post_stats: Post 기간 통계
            kpi_positivity: KPI 극성 ("positive" 또는 "negative")
            
        Returns:
            Dict[str, Any]: 분포 분석 결과
        """
        try:
            pre_min, pre_max = pre_stats.min, pre_stats.max
            post_min, post_max = post_stats.min, post_stats.max
            
            if kpi_positivity == "positive":
                # Positive KPI: max(pre) < min(post) → Improve
                is_improve = pre_max < post_min
                comparison_type = "max_pre_vs_min_post"
                comparison_values = {"pre_max": pre_max, "post_min": post_min}
            else:
                # Negative KPI: min(pre) > max(post) → Improve  
                is_improve = pre_min > post_max
                comparison_type = "min_pre_vs_max_post"
                comparison_values = {"pre_min": pre_min, "post_max": post_max}
            
            # 분포 겹침 정도 계산
            overlap_analysis = self._calculate_distribution_overlap(pre_stats, post_stats)
            
            return {
                "is_improve": is_improve,
                "kpi_positivity": kpi_positivity,
                "comparison_type": comparison_type,
                "comparison_values": comparison_values,
                "distribution_overlap": overlap_analysis,
                "separation_clear": not overlap_analysis["has_overlap"]
            }
            
        except Exception as e:
            self.logger.error(f"Distribution separation analysis error: {e}")
            return {"is_improve": False, "error": str(e)}
    
    def _calculate_distribution_overlap(self, pre_stats: PegPeriodStats, post_stats: PegPeriodStats) -> Dict[str, Any]:
        """
        분포 겹침 계산
        
        Args:
            pre_stats: Pre 기간 통계
            post_stats: Post 기간 통계
            
        Returns:
            Dict[str, Any]: 겹침 분석 결과
        """
        try:
            # 분포 범위 계산
            pre_range = [pre_stats.min, pre_stats.max]
            post_range = [post_stats.min, post_stats.max]
            
            # 겹침 구간 계산
            overlap_start = max(pre_range[0], post_range[0])
            overlap_end = min(pre_range[1], post_range[1])
            
            has_overlap = overlap_start <= overlap_end
            overlap_size = max(0, overlap_end - overlap_start) if has_overlap else 0
            
            # 전체 범위 대비 겹침 비율
            total_range = max(pre_range[1], post_range[1]) - min(pre_range[0], post_range[0])
            overlap_ratio = overlap_size / total_range if total_range > 0 else 0
            
            return {
                "has_overlap": has_overlap,
                "overlap_size": overlap_size,
                "overlap_ratio": overlap_ratio,
                "pre_range": pre_range,
                "post_range": post_range,
                "separation_distance": overlap_start - overlap_end if not has_overlap else 0
            }
            
        except Exception as e:
            self.logger.error(f"Distribution overlap calculation error: {e}")
            return {"has_overlap": True, "error": str(e)}
    
    def _generate_improve_reasoning(self, distribution_analysis: Dict[str, Any], kpi_positivity: str) -> str:
        """Improve 판정 근거 생성"""
        comparison_values = distribution_analysis["comparison_values"]
        overlap_info = distribution_analysis["distribution_overlap"]
        
        if kpi_positivity == "positive":
            pre_max = comparison_values["pre_max"]
            post_min = comparison_values["post_min"]
            return (f"성능 개선: Positive KPI에서 Pre 최댓값({pre_max:.1f}) < Post 최솟값({post_min:.1f}), "
                   f"분포 분리됨 (겹침 비율: {overlap_info['overlap_ratio']:.1%})")
        else:
            pre_min = comparison_values["pre_min"]
            post_max = comparison_values["post_max"]
            return (f"성능 개선: Negative KPI에서 Pre 최솟값({pre_min:.1f}) > Post 최댓값({post_max:.1f}), "
                   f"분포 분리됨 (겹침 비율: {overlap_info['overlap_ratio']:.1%})")
    
    def _validate_specific_config(self, config: Dict[str, Any]) -> bool:
        """Improve 분석기 특화 설정 검증"""
        # KPI 극성 정보가 있으면 검증
        kpi_positivity = config.get("kpi_positivity")
        if kpi_positivity and kpi_positivity not in ["positive", "negative"]:
            self.logger.error(f"Invalid kpi_positivity: {kpi_positivity}")
            return False
        
        return True


class DegradeAnalyzer(BaseKPIAnalyzer):
    """
    Degrade 분석기
    
    KPI 극성에 따라 성능 저하를 탐지합니다.
    - Positive KPI: min(pre) > max(post)
    - Negative KPI: max(pre) < min(post)
    
    Single Responsibility: 성능 저하 검사만 담당
    """
    
    def __init__(self):
        """Degrade 분석기 초기화"""
        super().__init__("DegradeAnalyzer", AnalysisRulePriority.DEGRADE)
    
    def _is_rule_applicable(self, 
                           pre_stats: PegPeriodStats,
                           post_stats: PegPeriodStats,
                           compare_metrics: PegCompareMetrics,
                           config: Dict[str, Any]) -> bool:
        """
        Degrade 규칙 적용 가능성 확인
        
        min/max 값이 있고 ND가 없는 경우에만 적용 가능
        """
        return (not compare_metrics.has_nd and 
                pre_stats.min is not None and pre_stats.max is not None and
                post_stats.min is not None and post_stats.max is not None)
    
    def _execute_analysis(self, 
                         pre_stats: PegPeriodStats,
                         post_stats: PegPeriodStats,
                         compare_metrics: PegCompareMetrics,
                         config: Dict[str, Any]) -> Optional[KPIAnalysisResult]:
        """
        Degrade 분석 실행
        
        Args:
            pre_stats: Pre 기간 통계
            post_stats: Post 기간 통계
            compare_metrics: 비교 지표
            config: 분석 설정
            
        Returns:
            Optional[KPIAnalysisResult]: Degrade 분석 결과
        """
        try:
            # KPI 극성 확인
            kpi_positivity = config.get("kpi_positivity", "positive")
            
            # 분포 비교 분석 (Improve와 반대 조건)
            distribution_analysis = self._analyze_distribution_separation(
                pre_stats, post_stats, kpi_positivity
            )
            
            if distribution_analysis["is_degrade"]:
                reasoning = self._generate_degrade_reasoning(
                    distribution_analysis, kpi_positivity
                )
                
                return KPIAnalysisResult(
                    judgement_type=JudgementType.NOK,  # Degrade는 NOK로 분류
                    compare_detail=CompareDetail.DEGRADE,
                    reasoning=reasoning,
                    confidence=0.9,  # 분포 기반 분석의 높은 신뢰도
                    metrics=distribution_analysis,
                    thresholds_used={}
                )
            
            return None  # Degrade가 아님
            
        except Exception as e:
            self.logger.error(f"Degrade analysis error: {e}")
            raise
    
    def _analyze_distribution_separation(self, 
                                       pre_stats: PegPeriodStats, 
                                       post_stats: PegPeriodStats, 
                                       kpi_positivity: str) -> Dict[str, Any]:
        """
        분포 분리 분석 (Degrade 관점)
        
        Args:
            pre_stats: Pre 기간 통계
            post_stats: Post 기간 통계
            kpi_positivity: KPI 극성
            
        Returns:
            Dict[str, Any]: 분포 분석 결과
        """
        try:
            pre_min, pre_max = pre_stats.min, pre_stats.max
            post_min, post_max = post_stats.min, post_stats.max
            
            if kpi_positivity == "positive":
                # Positive KPI: min(pre) > max(post) → Degrade
                is_degrade = pre_min > post_max
                comparison_type = "min_pre_vs_max_post"
                comparison_values = {"pre_min": pre_min, "post_max": post_max}
            else:
                # Negative KPI: max(pre) < min(post) → Degrade
                is_degrade = pre_max < post_min
                comparison_type = "max_pre_vs_min_post"
                comparison_values = {"pre_max": pre_max, "post_min": post_min}
            
            # 분포 겹침 정도 계산 (ImproveAnalyzer와 동일한 로직 재사용)
            improve_analyzer = ImproveAnalyzer()
            overlap_analysis = improve_analyzer._calculate_distribution_overlap(pre_stats, post_stats)
            
            return {
                "is_degrade": is_degrade,
                "kpi_positivity": kpi_positivity,
                "comparison_type": comparison_type,
                "comparison_values": comparison_values,
                "distribution_overlap": overlap_analysis,
                "separation_clear": not overlap_analysis["has_overlap"]
            }
            
        except Exception as e:
            self.logger.error(f"Distribution separation analysis error: {e}")
            return {"is_degrade": False, "error": str(e)}
    
    def _generate_degrade_reasoning(self, distribution_analysis: Dict[str, Any], kpi_positivity: str) -> str:
        """Degrade 판정 근거 생성"""
        comparison_values = distribution_analysis["comparison_values"]
        overlap_info = distribution_analysis["distribution_overlap"]
        
        if kpi_positivity == "positive":
            pre_min = comparison_values["pre_min"]
            post_max = comparison_values["post_max"]
            return (f"성능 저하: Positive KPI에서 Pre 최솟값({pre_min:.1f}) > Post 최댓값({post_max:.1f}), "
                   f"분포 분리됨 (겹침 비율: {overlap_info['overlap_ratio']:.1%})")
        else:
            pre_max = comparison_values["pre_max"]
            post_min = comparison_values["post_min"]
            return (f"성능 저하: Negative KPI에서 Pre 최댓값({pre_max:.1f}) < Post 최솟값({post_min:.1f}), "
                   f"분포 분리됨 (겹침 비율: {overlap_info['overlap_ratio']:.1%})")
    
    def _validate_specific_config(self, config: Dict[str, Any]) -> bool:
        """Degrade 분석기 특화 설정 검증"""
        # KPI 극성 정보가 있으면 검증
        kpi_positivity = config.get("kpi_positivity")
        if kpi_positivity and kpi_positivity not in ["positive", "negative"]:
            self.logger.error(f"Invalid kpi_positivity: {kpi_positivity}")
            return False
        
        return True


# =============================================================================
# Similar 분석기 (복잡한 의사결정 트리)
# =============================================================================

class SimilarAnalyzer(BaseKPIAnalyzer):
    """
    Similar 분석기
    
    복잡한 의사결정 트리를 가진 분석기입니다.
    1. 트래픽 볼륨 분류 (β0 기준)
    2. 볼륨에 따른 임계값 선택 (β1 또는 β2)
    3. 상대적 델타 검사 (선택된 임계값)
    4. 절대적 델타 검사 (β5)
    
    Single Responsibility: Similar 조건 검사만 담당
    """
    
    def __init__(self):
        """Similar 분석기 초기화"""
        super().__init__("SimilarAnalyzer", AnalysisRulePriority.SIMILAR)
    
    def _is_rule_applicable(self, 
                           pre_stats: PegPeriodStats,
                           post_stats: PegPeriodStats,
                           compare_metrics: PegCompareMetrics,
                           config: Dict[str, Any]) -> bool:
        """
        Similar 규칙 적용 가능성 확인
        
        평균값이 있고 ND가 없으며, Improve/Degrade가 아닌 경우에만 적용
        """
        return (not compare_metrics.has_nd and 
                pre_stats.mean is not None and post_stats.mean is not None and
                compare_metrics.delta_pct is not None)
    
    def _execute_analysis(self, 
                         pre_stats: PegPeriodStats,
                         post_stats: PegPeriodStats,
                         compare_metrics: PegCompareMetrics,
                         config: Dict[str, Any]) -> Optional[KPIAnalysisResult]:
        """
        Similar 분석 실행
        
        Args:
            pre_stats: Pre 기간 통계
            post_stats: Post 기간 통계
            compare_metrics: 비교 지표
            config: 분석 설정
            
        Returns:
            Optional[KPIAnalysisResult]: Similar 분석 결과
        """
        try:
            # β 임계값들 추출
            beta_0 = config.get("beta_0", 1000.0)  # 트래픽 볼륨 임계값
            beta_1 = config.get("beta_1", 5.0)     # 고트래픽 임계값
            beta_2 = config.get("beta_2", 10.0)    # 저트래픽 임계값
            beta_5 = config.get("beta_5", 3.0)     # 절대값 임계값
            
            # 1단계: 트래픽 볼륨 분류
            volume_analysis = self._classify_traffic_volume(pre_stats, post_stats, beta_0)
            
            # 2단계: 델타 계산
            delta_analysis = self._calculate_delta_percentage(pre_stats, post_stats)
            
            # 3단계: Similar 판정 로직 적용
            similar_analysis = self._apply_similar_logic(
                volume_analysis, delta_analysis, beta_1, beta_2, beta_5
            )
            
            if similar_analysis["is_similar"]:
                reasoning = self._generate_similar_reasoning(
                    similar_analysis, volume_analysis, delta_analysis
                )
                
                return KPIAnalysisResult(
                    judgement_type=JudgementType.OK,  # Similar는 OK로 분류
                    compare_detail=CompareDetail.SIMILAR,
                    reasoning=reasoning,
                    confidence=0.95,  # 수학적 계산의 높은 신뢰도
                    metrics={**volume_analysis, **delta_analysis, **similar_analysis},
                    thresholds_used={
                        "beta_0": beta_0, "beta_1": beta_1, "beta_2": beta_2, "beta_5": beta_5
                    }
                )
            
            return None  # Similar가 아님
            
        except Exception as e:
            self.logger.error(f"Similar analysis error: {e}")
            raise
    
    def _classify_traffic_volume(self, 
                               pre_stats: PegPeriodStats, 
                               post_stats: PegPeriodStats, 
                               beta_0: float) -> Dict[str, Any]:
        """
        트래픽 볼륨 분류 (β0 기준)
        
        Args:
            pre_stats: Pre 기간 통계
            post_stats: Post 기간 통계
            beta_0: 트래픽 볼륨 임계값
            
        Returns:
            Dict[str, Any]: 볼륨 분류 결과
        """
        try:
            pre_mean = pre_stats.mean or 0.0
            post_mean = post_stats.mean or 0.0
            
            # 5.2 Similar 조건: pre < β0 OR post < β0 → 저트래픽 (β2 적용)
            is_low_traffic = pre_mean < beta_0 or post_mean < beta_0
            
            # 고트래픽: pre ≥ β0 AND post ≥ β0 → 고트래픽 (β1 적용)
            is_high_traffic = pre_mean >= beta_0 and post_mean >= beta_0
            
            # 선택된 임계값 결정
            if is_low_traffic:
                selected_threshold = 10.0  # β2 (저트래픽용)
                traffic_classification = "low"
            else:
                selected_threshold = 5.0   # β1 (고트래픽용)
                traffic_classification = "high"
            
            return {
                "pre_mean": pre_mean,
                "post_mean": post_mean,
                "beta_0_threshold": beta_0,
                "is_low_traffic": is_low_traffic,
                "is_high_traffic": is_high_traffic,
                "traffic_classification": traffic_classification,
                "selected_threshold": selected_threshold,
                "threshold_type": "beta_2" if is_low_traffic else "beta_1"
            }
            
        except Exception as e:
            self.logger.error(f"Traffic volume classification error: {e}")
            return {"traffic_classification": "unknown", "selected_threshold": 10.0}
    
    def _calculate_delta_percentage(self, pre_stats: PegPeriodStats, post_stats: PegPeriodStats) -> Dict[str, Any]:
        """
        델타 백분율 계산
        
        Args:
            pre_stats: Pre 기간 통계
            post_stats: Post 기간 통계
            
        Returns:
            Dict[str, Any]: 델타 계산 결과
        """
        try:
            pre_mean = pre_stats.mean or 0.0
            post_mean = post_stats.mean or 0.0
            
            # δ = (post-pre)/pre * 100 계산
            if pre_mean == 0:
                if post_mean == 0:
                    delta_pct = 0.0
                    calculation_note = "both_zero"
                else:
                    delta_pct = 100.0 if post_mean > 0 else -100.0
                    calculation_note = "pre_zero_special_case"
            else:
                delta_pct = ((post_mean - pre_mean) / pre_mean) * 100
                calculation_note = "normal_calculation"
            
            # 절댓값 계산
            abs_delta = abs(delta_pct)
            
            return {
                "pre_mean": pre_mean,
                "post_mean": post_mean,
                "delta_percentage": delta_pct,
                "abs_delta": abs_delta,
                "calculation_note": calculation_note
            }
            
        except Exception as e:
            self.logger.error(f"Delta calculation error: {e}")
            return {"delta_percentage": 0.0, "abs_delta": 0.0, "calculation_note": "error"}
    
    def _apply_similar_logic(self, 
                           volume_analysis: Dict[str, Any], 
                           delta_analysis: Dict[str, Any], 
                           beta_1: float, 
                           beta_2: float, 
                           beta_5: float) -> Dict[str, Any]:
        """
        Similar 판정 로직 적용
        
        복잡한 의사결정 트리:
        1. 트래픽 볼륨에 따른 임계값 선택
        2. 상대적 델타 검사 (β1 또는 β2)
        3. 절대적 델타 검사 (β5)
        4. 두 조건 모두 만족해야 Similar
        
        Args:
            volume_analysis: 볼륨 분류 결과
            delta_analysis: 델타 계산 결과
            beta_1: 고트래픽 임계값
            beta_2: 저트래픽 임계값
            beta_5: 절대값 임계값
            
        Returns:
            Dict[str, Any]: Similar 판정 결과
        """
        try:
            abs_delta = delta_analysis["abs_delta"]
            selected_threshold = volume_analysis["selected_threshold"]
            
            # 조건 1: 상대적 델타 검사 (선택된 임계값 기준)
            relative_check_passed = abs_delta <= selected_threshold
            
            # 조건 2: 절대적 델타 검사 (β5 기준)
            absolute_check_passed = abs_delta < beta_5
            
            # Similar 판정: 두 조건 모두 만족해야 함
            is_similar = relative_check_passed and absolute_check_passed
            
            return {
                "is_similar": is_similar,
                "relative_check_passed": relative_check_passed,
                "absolute_check_passed": absolute_check_passed,
                "selected_threshold": selected_threshold,
                "beta_5_threshold": beta_5,
                "abs_delta": abs_delta,
                "relative_margin": selected_threshold - abs_delta,
                "absolute_margin": beta_5 - abs_delta
            }
            
        except Exception as e:
            self.logger.error(f"Similar logic application error: {e}")
            return {"is_similar": False, "error": str(e)}
    
    def _generate_similar_reasoning(self, 
                                  similar_analysis: Dict[str, Any], 
                                  volume_analysis: Dict[str, Any], 
                                  delta_analysis: Dict[str, Any]) -> str:
        """Similar 판정 근거 생성"""
        traffic_type = volume_analysis["traffic_classification"]
        threshold_type = volume_analysis["threshold_type"]
        selected_threshold = similar_analysis["selected_threshold"]
        abs_delta = similar_analysis["abs_delta"]
        beta_5 = similar_analysis["beta_5_threshold"]
        
        return (f"유사 수준: {traffic_type.upper()} 트래픽 ({threshold_type}={selected_threshold}) 기준 "
               f"|δ|={abs_delta:.1f} ≤ {selected_threshold} AND |δ| < {beta_5}, "
               f"두 조건 모두 만족")
    
    def _validate_specific_config(self, config: Dict[str, Any]) -> bool:
        """Similar 분석기 특화 설정 검증"""
        required_betas = ["beta_0", "beta_1", "beta_2", "beta_5"]
        for beta in required_betas:
            value = config.get(beta)
            if value is None:
                self.logger.error(f"Required threshold {beta} is missing")
                return False
            
            if not isinstance(value, (int, float)) or value < 0:
                self.logger.error(f"{beta} must be a non-negative number, got {value}")
                return False
        
        # β1 < β2 관계 검증
        beta_1 = config.get("beta_1", 0)
        beta_2 = config.get("beta_2", 0)
        if beta_1 > beta_2:
            self.logger.warning(f"beta_1 ({beta_1}) > beta_2 ({beta_2}), unusual configuration")
        
        return True


# =============================================================================
# Delta 계층 분석기들 (Low/Medium/High Delta)
# =============================================================================

class LowDeltaAnalyzer(BaseKPIAnalyzer):
    """
    Low Delta 분석기
    
    β2 < δ ≤ 2*β2 (저트래픽) 또는 β1 < δ ≤ 2*β1 (고트래픽) 조건 검사
    
    Single Responsibility: Low Delta 조건 검사만 담당
    """
    
    def __init__(self):
        """Low Delta 분석기 초기화"""
        super().__init__("LowDeltaAnalyzer", AnalysisRulePriority.LOW_DELTA)
        
        # Similar 분석기 로직 재사용 (DRY 원칙)
        self.similar_analyzer = SimilarAnalyzer()
    
    def _is_rule_applicable(self, 
                           pre_stats: PegPeriodStats,
                           post_stats: PegPeriodStats,
                           compare_metrics: PegCompareMetrics,
                           config: Dict[str, Any]) -> bool:
        """Low Delta 규칙 적용 가능성 확인"""
        return (not compare_metrics.has_nd and 
                pre_stats.mean is not None and post_stats.mean is not None and
                compare_metrics.delta_pct is not None)
    
    def _execute_analysis(self, 
                         pre_stats: PegPeriodStats,
                         post_stats: PegPeriodStats,
                         compare_metrics: PegCompareMetrics,
                         config: Dict[str, Any]) -> Optional[KPIAnalysisResult]:
        """Low Delta 분석 실행"""
        try:
            # β 임계값들 추출
            beta_0 = config.get("beta_0", 1000.0)
            beta_1 = config.get("beta_1", 5.0)
            beta_2 = config.get("beta_2", 10.0)
            
            # Similar 분석기의 로직 재사용
            volume_analysis = self.similar_analyzer._classify_traffic_volume(pre_stats, post_stats, beta_0)
            delta_analysis = self.similar_analyzer._calculate_delta_percentage(pre_stats, post_stats)
            
            # Low Delta 조건 검사
            delta_classification = self._classify_delta_level(
                volume_analysis, delta_analysis, beta_1, beta_2
            )
            
            if delta_classification["is_low_delta"]:
                reasoning = self._generate_low_delta_reasoning(
                    delta_classification, volume_analysis, delta_analysis
                )
                
                return KPIAnalysisResult(
                    judgement_type=JudgementType.NOK,  # Low Delta는 NOK
                    compare_detail=CompareDetail.LOW_DELTA,
                    reasoning=reasoning,
                    confidence=0.95,
                    metrics={**volume_analysis, **delta_analysis, **delta_classification},
                    thresholds_used={"beta_0": beta_0, "beta_1": beta_1, "beta_2": beta_2}
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Low Delta analysis error: {e}")
            raise
    
    def _classify_delta_level(self, 
                            volume_analysis: Dict[str, Any], 
                            delta_analysis: Dict[str, Any], 
                            beta_1: float, 
                            beta_2: float) -> Dict[str, Any]:
        """델타 수준 분류"""
        try:
            abs_delta = delta_analysis["abs_delta"]
            is_low_traffic = volume_analysis["is_low_traffic"]
            
            if is_low_traffic:
                # 저트래픽: β2 < δ ≤ 2*β2
                lower_bound = beta_2
                upper_bound = 2 * beta_2
                threshold_type = "beta_2"
            else:
                # 고트래픽: β1 < δ ≤ 2*β1
                lower_bound = beta_1
                upper_bound = 2 * beta_1
                threshold_type = "beta_1"
            
            is_low_delta = lower_bound < abs_delta <= upper_bound
            
            return {
                "is_low_delta": is_low_delta,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "threshold_type": threshold_type,
                "abs_delta": abs_delta,
                "within_range": is_low_delta
            }
            
        except Exception as e:
            self.logger.error(f"Delta level classification error: {e}")
            return {"is_low_delta": False, "error": str(e)}
    
    def _generate_low_delta_reasoning(self, 
                                    delta_classification: Dict[str, Any], 
                                    volume_analysis: Dict[str, Any], 
                                    delta_analysis: Dict[str, Any]) -> str:
        """Low Delta 판정 근거 생성"""
        traffic_type = volume_analysis["traffic_classification"]
        threshold_type = delta_classification["threshold_type"]
        lower_bound = delta_classification["lower_bound"]
        upper_bound = delta_classification["upper_bound"]
        abs_delta = delta_classification["abs_delta"]
        
        return (f"낮은 변화량: {traffic_type.upper()} 트래픽 ({threshold_type}) 기준 "
               f"{lower_bound} < |δ|={abs_delta:.1f} ≤ {upper_bound}")
    
    def _validate_specific_config(self, config: Dict[str, Any]) -> bool:
        """Low Delta 분석기 특화 설정 검증"""
        required_betas = ["beta_0", "beta_1", "beta_2"]
        for beta in required_betas:
            if config.get(beta) is None:
                self.logger.error(f"Required threshold {beta} is missing")
                return False
        return True


class MediumDeltaAnalyzer(BaseKPIAnalyzer):
    """
    Medium Delta 분석기
    
    2*β2 < δ ≤ β3 (저트래픽) 또는 2*β1 < δ ≤ β3 (고트래픽) 조건 검사
    
    Single Responsibility: Medium Delta 조건 검사만 담당
    """
    
    def __init__(self):
        """Medium Delta 분석기 초기화"""
        super().__init__("MediumDeltaAnalyzer", AnalysisRulePriority.MEDIUM_DELTA)
        
        # 기존 분석기 로직 재사용
        self.similar_analyzer = SimilarAnalyzer()
        self.low_delta_analyzer = LowDeltaAnalyzer()
    
    def _is_rule_applicable(self, 
                           pre_stats: PegPeriodStats,
                           post_stats: PegPeriodStats,
                           compare_metrics: PegCompareMetrics,
                           config: Dict[str, Any]) -> bool:
        """Medium Delta 규칙 적용 가능성 확인"""
        return (not compare_metrics.has_nd and 
                pre_stats.mean is not None and post_stats.mean is not None and
                compare_metrics.delta_pct is not None)
    
    def _execute_analysis(self, 
                         pre_stats: PegPeriodStats,
                         post_stats: PegPeriodStats,
                         compare_metrics: PegCompareMetrics,
                         config: Dict[str, Any]) -> Optional[KPIAnalysisResult]:
        """Medium Delta 분석 실행"""
        try:
            # β 임계값들 추출
            beta_0 = config.get("beta_0", 1000.0)
            beta_1 = config.get("beta_1", 5.0)
            beta_2 = config.get("beta_2", 10.0)
            beta_3 = config.get("beta_3", 500.0)
            
            # 기존 로직 재사용
            volume_analysis = self.similar_analyzer._classify_traffic_volume(pre_stats, post_stats, beta_0)
            delta_analysis = self.similar_analyzer._calculate_delta_percentage(pre_stats, post_stats)
            
            # Medium Delta 조건 검사
            delta_classification = self._classify_medium_delta_level(
                volume_analysis, delta_analysis, beta_1, beta_2, beta_3
            )
            
            if delta_classification["is_medium_delta"]:
                reasoning = self._generate_medium_delta_reasoning(
                    delta_classification, volume_analysis, delta_analysis
                )
                
                return KPIAnalysisResult(
                    judgement_type=JudgementType.NOK,  # Medium Delta는 NOK
                    compare_detail=CompareDetail.MEDIUM_DELTA,
                    reasoning=reasoning,
                    confidence=0.95,
                    metrics={**volume_analysis, **delta_analysis, **delta_classification},
                    thresholds_used={"beta_0": beta_0, "beta_1": beta_1, "beta_2": beta_2, "beta_3": beta_3}
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Medium Delta analysis error: {e}")
            raise
    
    def _classify_medium_delta_level(self, 
                                   volume_analysis: Dict[str, Any], 
                                   delta_analysis: Dict[str, Any], 
                                   beta_1: float, 
                                   beta_2: float, 
                                   beta_3: float) -> Dict[str, Any]:
        """Medium Delta 수준 분류"""
        try:
            abs_delta = delta_analysis["abs_delta"]
            is_low_traffic = volume_analysis["is_low_traffic"]
            
            if is_low_traffic:
                # 저트래픽: 2*β2 < δ ≤ β3
                lower_bound = 2 * beta_2
                upper_bound = beta_3
                threshold_type = "2*beta_2"
            else:
                # 고트래픽: 2*β1 < δ ≤ β3
                lower_bound = 2 * beta_1
                upper_bound = beta_3
                threshold_type = "2*beta_1"
            
            is_medium_delta = lower_bound < abs_delta <= upper_bound
            
            return {
                "is_medium_delta": is_medium_delta,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "threshold_type": threshold_type,
                "abs_delta": abs_delta,
                "within_range": is_medium_delta
            }
            
        except Exception as e:
            self.logger.error(f"Medium delta classification error: {e}")
            return {"is_medium_delta": False, "error": str(e)}
    
    def _generate_medium_delta_reasoning(self, 
                                       delta_classification: Dict[str, Any], 
                                       volume_analysis: Dict[str, Any], 
                                       delta_analysis: Dict[str, Any]) -> str:
        """Medium Delta 판정 근거 생성"""
        traffic_type = volume_analysis["traffic_classification"]
        threshold_type = delta_classification["threshold_type"]
        lower_bound = delta_classification["lower_bound"]
        upper_bound = delta_classification["upper_bound"]
        abs_delta = delta_classification["abs_delta"]
        
        return (f"중간 변화량: {traffic_type.upper()} 트래픽 ({threshold_type}) 기준 "
               f"{lower_bound} < |δ|={abs_delta:.1f} ≤ {upper_bound}")
    
    def _validate_specific_config(self, config: Dict[str, Any]) -> bool:
        """Medium Delta 분석기 특화 설정 검증"""
        required_betas = ["beta_0", "beta_1", "beta_2", "beta_3"]
        for beta in required_betas:
            if config.get(beta) is None:
                self.logger.error(f"Required threshold {beta} is missing")
                return False
        return True


class HighDeltaAnalyzer(BaseKPIAnalyzer):
    """
    High Delta 분석기 (KPI 분석용)
    
    δ > β3 조건 검사 (이상 탐지의 High Delta와 동일하지만 다른 컨텍스트)
    
    Single Responsibility: High Delta 조건 검사만 담당
    """
    
    def __init__(self):
        """High Delta 분석기 초기화"""
        super().__init__("HighDeltaAnalyzer", AnalysisRulePriority.HIGH_DELTA)
        
        # 기존 로직 재사용
        self.similar_analyzer = SimilarAnalyzer()
    
    def _is_rule_applicable(self, 
                           pre_stats: PegPeriodStats,
                           post_stats: PegPeriodStats,
                           compare_metrics: PegCompareMetrics,
                           config: Dict[str, Any]) -> bool:
        """High Delta 규칙 적용 가능성 확인"""
        return (not compare_metrics.has_nd and 
                pre_stats.mean is not None and post_stats.mean is not None and
                compare_metrics.delta_pct is not None)
    
    def _execute_analysis(self, 
                         pre_stats: PegPeriodStats,
                         post_stats: PegPeriodStats,
                         compare_metrics: PegCompareMetrics,
                         config: Dict[str, Any]) -> Optional[KPIAnalysisResult]:
        """High Delta 분석 실행"""
        try:
            beta_3 = config.get("beta_3", 500.0)
            
            # 델타 계산 (재사용)
            delta_analysis = self.similar_analyzer._calculate_delta_percentage(pre_stats, post_stats)
            
            # High Delta 조건 검사
            abs_delta = delta_analysis["abs_delta"]
            is_high_delta = abs_delta > beta_3
            
            if is_high_delta:
                reasoning = f"높은 변화량: |δ|={abs_delta:.1f} > β3({beta_3})"
                
                return KPIAnalysisResult(
                    judgement_type=JudgementType.NOK,  # High Delta는 NOK
                    compare_detail=CompareDetail.HIGH_DELTA,
                    reasoning=reasoning,
                    confidence=1.0,  # 수학적 계산이므로 확실함
                    metrics=delta_analysis,
                    thresholds_used={"beta_3": beta_3}
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"High Delta analysis error: {e}")
            raise
    
    def _validate_specific_config(self, config: Dict[str, Any]) -> bool:
        """High Delta 분석기 특화 설정 검증"""
        beta_3 = config.get("beta_3")
        if beta_3 is None:
            self.logger.error("beta_3 threshold is required")
            return False
        
        if not isinstance(beta_3, (int, float)) or beta_3 <= 0:
            self.logger.error(f"beta_3 must be a positive number, got {beta_3}")
            return False
        
        return True


# =============================================================================
# KPI 분석기 팩토리 (Factory Pattern + Dependency Injection)
# =============================================================================

class KPIAnalyzerFactory:
    """
    KPI 분석기 팩토리
    
    Factory Pattern을 사용하여 분석기 인스턴스를 생성하고 관리합니다.
    우선순위에 따라 정렬된 분석기 체인을 제공합니다.
    """
    
    def __init__(self):
        """팩토리 초기화"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("KPI analyzer factory initialized")
    
    def create_cant_judge_analyzer(self) -> CantJudgeAnalyzer:
        """Can't Judge 분석기 생성"""
        return CantJudgeAnalyzer()
    
    def create_high_variation_analyzer(self) -> HighVariationAnalyzer:
        """High Variation 분석기 생성"""
        return HighVariationAnalyzer()
    
    def create_improve_analyzer(self) -> ImproveAnalyzer:
        """Improve 분석기 생성"""
        return ImproveAnalyzer()
    
    def create_degrade_analyzer(self) -> DegradeAnalyzer:
        """Degrade 분석기 생성"""
        return DegradeAnalyzer()
    
    def create_similar_analyzer(self) -> 'SimilarAnalyzer':
        """Similar 분석기 생성"""
        return SimilarAnalyzer()
    
    def create_low_delta_analyzer(self) -> 'LowDeltaAnalyzer':
        """Low Delta 분석기 생성"""
        return LowDeltaAnalyzer()
    
    def create_medium_delta_analyzer(self) -> 'MediumDeltaAnalyzer':
        """Medium Delta 분석기 생성"""
        return MediumDeltaAnalyzer()
    
    def create_high_delta_analyzer(self) -> 'HighDeltaAnalyzer':
        """High Delta 분석기 생성"""
        return HighDeltaAnalyzer()
    
    def create_priority_ordered_analyzers(self) -> List[BaseKPIAnalyzer]:
        """
        우선순위 순서로 정렬된 분석기 목록 생성
        
        Returns:
            List[BaseKPIAnalyzer]: 우선순위 내림차순 분석기 목록
        """
        analyzers = [
            self.create_cant_judge_analyzer(),      # 우선순위 100
            self.create_high_variation_analyzer(),  # 우선순위 90
            self.create_improve_analyzer(),         # 우선순위 80
            self.create_degrade_analyzer(),         # 우선순위 80
            self.create_high_delta_analyzer(),      # 우선순위 70
            self.create_medium_delta_analyzer(),    # 우선순위 60
            self.create_low_delta_analyzer(),       # 우선순위 50
            self.create_similar_analyzer()          # 우선순위 40
        ]
        
        # 우선순위 내림차순 정렬
        analyzers.sort(key=lambda x: x.get_priority(), reverse=True)
        
        self.logger.info(f"Created {len(analyzers)} analyzers in priority order")
        return analyzers
    
    def get_available_analyzers(self) -> List[str]:
        """사용 가능한 분석기 목록"""
        return ["cant_judge", "high_variation", "improve", "degrade"]


# =============================================================================
# High Variation 분석기 (실제 구현)
# =============================================================================

class HighVariationAnalyzer(BaseKPIAnalyzer):
    """
    High Variation 분석기
    
    CV > β4 조건과 특수 케이스들을 검사합니다.
    """
    
    def __init__(self):
        """High Variation 분석기 초기화"""
        super().__init__("HighVariationAnalyzer", AnalysisRulePriority.HIGH_VARIATION)
    
    def _is_rule_applicable(self, 
                           pre_stats: PegPeriodStats,
                           post_stats: PegPeriodStats,
                           compare_metrics: PegCompareMetrics,
                           config: Dict[str, Any]) -> bool:
        """High Variation 규칙 적용 가능성 확인"""
        # Can't Judge가 이미 적용된 경우 제외
        if compare_metrics.has_nd:
            return False
        
        # CV 계산이 가능하거나 특수 케이스가 있는 경우
        return (pre_stats.cv is not None or post_stats.cv is not None or 
                pre_stats.zero_ratio > 0 or post_stats.zero_ratio > 0)
    
    def _execute_analysis(self, 
                         pre_stats: PegPeriodStats,
                         post_stats: PegPeriodStats,
                         compare_metrics: PegCompareMetrics,
                         config: Dict[str, Any]) -> Optional[KPIAnalysisResult]:
        """High Variation 분석 실행"""
        try:
            beta_4 = config.get("beta_4", 10.0)
            
            # CV 분석 (CantJudgeAnalyzer와 동일한 로직 재사용)
            cant_judge_analyzer = CantJudgeAnalyzer()
            cv_analysis = cant_judge_analyzer._analyze_coefficient_of_variation(pre_stats, post_stats, beta_4)
            
            if cv_analysis["is_high_variation"]:
                reasoning = cant_judge_analyzer._generate_high_variation_reasoning(cv_analysis, beta_4)
                
                return KPIAnalysisResult(
                    judgement_type=JudgementType.CANT_JUDGE,
                    compare_detail=CompareDetail.HIGH_VARIATION,
                    reasoning=reasoning,
                    confidence=0.95,  # CV 계산의 높은 신뢰도
                    metrics=cv_analysis,
                    thresholds_used={"beta_4": beta_4}
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"High Variation analysis error: {e}")
            raise
    
    def _validate_specific_config(self, config: Dict[str, Any]) -> bool:
        """High Variation 분석기 특화 설정 검증"""
        beta_4 = config.get("beta_4")
        if beta_4 is None:
            self.logger.error("beta_4 threshold is required")
            return False
        
        if not isinstance(beta_4, (int, float)) or beta_4 <= 0:
            self.logger.error(f"beta_4 must be a positive number, got {beta_4}")
            return False
        
        return True


# =============================================================================
# 초기화 및 로깅
# =============================================================================

logger.info("KPI analyzers module loaded successfully")
