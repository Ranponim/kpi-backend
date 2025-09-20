"""
Choi 알고리즘 Strategy 패턴 인터페이스

이 모듈은 3GPP KPI PEGs 판정 알고리즘의 핵심 구성 요소들을
Strategy 패턴으로 추상화한 인터페이스를 정의합니다.

주요 인터페이스:
- FilteringStrategy: 6장 필터링 알고리즘 인터페이스
- JudgementStrategy: 4장, 5장 판정 알고리즘 인터페이스

이 추상화를 통해 알고리즘 구현체를 쉽게 교체하거나 확장할 수 있으며,
테스트 시에는 Mock 구현체를 사용할 수 있습니다.

PRD 참조: 섹션 3.1.2 (Strategy 패턴 적용)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import logging

from ..models.judgement import (
    FilteringResult,
    AbnormalDetectionResult,
    MainKPIJudgement,
    PegSampleSeries
)

logger = logging.getLogger(__name__)


# =============================================================================
# 필터링 전략 인터페이스
# =============================================================================

class FilteringStrategy(ABC):
    """
    필터링 알고리즘 Strategy 인터페이스 (6장 대응)
    
    이 인터페이스를 구현하는 클래스는 다음 기능을 제공해야 합니다:
    1. 데이터 전처리 (이웃 시각 0 처리, DL/UL 합 0 제외)
    2. PEG별 중앙값 계산
    3. 시계열 정규화 (sample/median)
    4. 임계값 적용 (Min_Th ≤ normalized ≤ Max_Th)
    5. 시간 슬롯 교집합 계산
    6. 50% 규칙 적용
    
    구현 예시:
    - ChoiFiltering: 원본 Choi 알고리즘 구현
    - ImprovedFiltering: 개선된 필터링 알고리즘
    - MockFiltering: 테스트용 Mock 구현
    """
    
    @abstractmethod
    def apply(self, 
              peg_data: Dict[str, List[PegSampleSeries]], 
              config: Dict[str, Any]) -> FilteringResult:
        """
        필터링 알고리즘 실행
        
        Args:
            peg_data: 셀별 PEG 시계열 데이터
                     {"cell_id": [PegSampleSeries, ...], ...}
            config: 필터링 설정 (임계값, 참조 PEG 목록 등)
        
        Returns:
            FilteringResult: 필터링 결과
            - valid_time_slots: 셀별 유효 시간 슬롯 인덱스
            - filter_ratio: 필터링 비율
            - warning_message: 50% 규칙 위반 시 경고
            - preprocessing_stats: 전처리 통계
            - median_values: PEG별 중앙값
        
        Raises:
            ValueError: 입력 데이터가 유효하지 않은 경우
            RuntimeError: 필터링 처리 중 오류 발생
        """
        pass
    
    @abstractmethod
    def validate_input(self, 
                      peg_data: Dict[str, List[PegSampleSeries]], 
                      config: Dict[str, Any]) -> bool:
        """
        입력 데이터 유효성 검증
        
        Args:
            peg_data: 검증할 PEG 데이터
            config: 필터링 설정
            
        Returns:
            bool: 유효성 여부
        """
        pass
    
    @abstractmethod
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        전략 정보 반환 (디버깅/로깅용)
        
        Returns:
            Dict[str, Any]: 전략 이름, 버전, 설명 등
        """
        pass


# =============================================================================
# 판정 전략 인터페이스  
# =============================================================================

class JudgementStrategy(ABC):
    """
    판정 알고리즘 Strategy 인터페이스 (4장, 5장 대응)
    
    이 인터페이스를 구현하는 클래스는 다음 기능을 제공해야 합니다:
    
    4장 이상 탐지:
    - Range: DIMS [min,max] 범위 검사
    - New: 신규 통계 검사  
    - ND: No Data 검사
    - Zero: 0값 검사
    - High Delta: 높은 변화율 검사
    - α0 규칙 적용 (셀 수 기준 표시 여부)
    
    5장 통계 분석:
    - Can't Judge: ND 포함 시 판정 불가
    - High Variation: CV > β4
    - Improve/Degrade: 분포 기반 성능 변화 판정
    - Similar/Delta 계층: 변화율 기반 세부 판정
    - Main/Sub KPI 결과 종합
    
    구현 예시:
    - ChoiJudgement: 원본 Choi 알고리즘 구현
    - EnhancedJudgement: 개선된 판정 알고리즘
    - MockJudgement: 테스트용 Mock 구현
    """
    
    @abstractmethod
    def apply(self,
              filtered_data: Dict[str, List[PegSampleSeries]],
              filtering_result: FilteringResult,
              config: Dict[str, Any]) -> Dict[str, Any]:
        """
        판정 알고리즘 실행
        
        Args:
            filtered_data: 필터링된 PEG 데이터
            filtering_result: 필터링 결과 (유효 시간 슬롯 정보)
            config: 판정 설정 (임계값, KPI 정의 등)
        
        Returns:
            Dict[str, Any]: 판정 결과
            - abnormal_detection: 이상 탐지 결과
            - kpi_judgement: KPI 판정 결과
            - processing_metadata: 처리 메타데이터
        
        Raises:
            ValueError: 입력 데이터가 유효하지 않은 경우
            RuntimeError: 판정 처리 중 오류 발생
        """
        pass
    
    @abstractmethod
    def detect_abnormal_stats(self,
                             peg_data: Dict[str, List[PegSampleSeries]],
                             config: Dict[str, Any]) -> AbnormalDetectionResult:
        """
        이상 통계 탐지 (4장)
        
        Args:
            peg_data: PEG 데이터
            config: 이상 탐지 설정
            
        Returns:
            AbnormalDetectionResult: 이상 탐지 결과
        """
        pass
    
    @abstractmethod
    def analyze_kpi_stats(self,
                         kpi_data: Dict[str, Dict[str, List[PegSampleSeries]]],
                         filtering_result: FilteringResult,
                         config: Dict[str, Any]) -> Dict[str, MainKPIJudgement]:
        """
        KPI 통계 분석 (5장)
        
        Args:
            kpi_data: KPI 토픽별 데이터
                     {"topic_name": {"main": [PegSampleSeries], "subs": [...]}}
            filtering_result: 필터링 결과
            config: KPI 분석 설정
            
        Returns:
            Dict[str, MainKPIJudgement]: KPI 토픽별 판정 결과
        """
        pass
    
    @abstractmethod
    def validate_input(self,
                      filtered_data: Dict[str, List[PegSampleSeries]],
                      filtering_result: FilteringResult,
                      config: Dict[str, Any]) -> bool:
        """
        입력 데이터 유효성 검증
        
        Args:
            filtered_data: 검증할 필터링된 데이터
            filtering_result: 필터링 결과
            config: 판정 설정
            
        Returns:
            bool: 유효성 여부
        """
        pass
    
    @abstractmethod
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        전략 정보 반환 (디버깅/로깅용)
        
        Returns:
            Dict[str, Any]: 전략 이름, 버전, 설명 등
        """
        pass


# =============================================================================
# 전략 팩토리 인터페이스
# =============================================================================

class StrategyFactory(ABC):
    """
    전략 생성 팩토리 인터페이스
    
    구체적인 전략 구현체들을 생성하고 관리하는 팩토리 클래스의
    인터페이스를 정의합니다. 이를 통해 런타임에 전략을 동적으로
    선택하거나 교체할 수 있습니다.
    """
    
    @abstractmethod
    def create_filtering_strategy(self, strategy_type: str, config: Dict[str, Any]) -> FilteringStrategy:
        """
        필터링 전략 생성
        
        Args:
            strategy_type: 전략 타입 ("choi", "improved", "mock" 등)
            config: 전략 설정
            
        Returns:
            FilteringStrategy: 필터링 전략 인스턴스
        """
        pass
    
    @abstractmethod
    def create_judgement_strategy(self, strategy_type: str, config: Dict[str, Any]) -> JudgementStrategy:
        """
        판정 전략 생성
        
        Args:
            strategy_type: 전략 타입 ("choi", "enhanced", "mock" 등)
            config: 전략 설정
            
        Returns:
            JudgementStrategy: 판정 전략 인스턴스
        """
        pass
    
    @abstractmethod
    def get_available_strategies(self) -> Dict[str, List[str]]:
        """
        사용 가능한 전략 목록 반환
        
        Returns:
            Dict[str, List[str]]: 전략 타입별 사용 가능한 구현체 목록
            {"filtering": ["choi", "improved"], "judgement": ["choi", "enhanced"]}
        """
        pass


# =============================================================================
# 기본 전략 베이스 클래스
# =============================================================================

class BaseFilteringStrategy(FilteringStrategy):
    """
    필터링 전략 기본 구현
    
    공통 기능을 제공하는 기본 클래스입니다.
    구체적인 필터링 로직은 하위 클래스에서 구현해야 합니다.
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        """
        기본 전략 초기화
        
        Args:
            name: 전략 이름
            version: 전략 버전
        """
        self.name = name
        self.version = version
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.logger.info(f"Filtering strategy '{name}' v{version} initialized")
    
    def validate_input(self, 
                      peg_data: Dict[str, List[PegSampleSeries]], 
                      config: Dict[str, Any]) -> bool:
        """기본 입력 검증 구현"""
        try:
            # 기본 구조 검증
            if not peg_data:
                self.logger.error("Empty PEG data provided")
                return False
            
            if not config:
                self.logger.error("Empty config provided")
                return False
            
            # 필수 설정 키 검증 (필터링 설정은 직접 전달됨)
            required_keys = ['min_threshold', 'max_threshold', 'filter_ratio']
            for key in required_keys:
                if key not in config:
                    self.logger.error(f"Required config key '{key}' missing")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating input: {e}")
            return False
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """전략 정보 반환"""
        return {
            "name": self.name,
            "version": self.version,
            "type": "filtering",
            "description": f"Filtering strategy implementation: {self.name}"
        }


class BaseJudgementStrategy(JudgementStrategy):
    """
    판정 전략 기본 구현
    
    공통 기능을 제공하는 기본 클래스입니다.
    구체적인 판정 로직은 하위 클래스에서 구현해야 합니다.
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        """
        기본 전략 초기화
        
        Args:
            name: 전략 이름
            version: 전략 버전
        """
        self.name = name
        self.version = version
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.logger.info(f"Judgement strategy '{name}' v{version} initialized")
    
    def validate_input(self,
                      filtered_data: Dict[str, List[PegSampleSeries]],
                      filtering_result: FilteringResult,
                      config: Dict[str, Any]) -> bool:
        """기본 입력 검증 구현"""
        try:
            # 기본 구조 검증
            if not filtered_data:
                self.logger.error("Empty filtered data provided")
                return False
            
            if not filtering_result:
                self.logger.error("Empty filtering result provided")
                return False
            
            if not config:
                self.logger.error("Empty config provided")
                return False
            
            # 필수 설정 키 검증 (더 유연하게)
            if 'abnormal_detection' not in config and 'stats_analyzing' not in config:
                # 최소한 하나의 설정은 있어야 함
                if not any(key in config for key in ['alpha_0', 'beta_0', 'beta_1', 'beta_3', 'beta_4']):
                    self.logger.error("No valid judgement config found")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating input: {e}")
            return False
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """전략 정보 반환"""
        return {
            "name": self.name,
            "version": self.version,
            "type": "judgement",
            "description": f"Judgement strategy implementation: {self.name}"
        }


# =============================================================================
# 유틸리티 함수
# =============================================================================

def validate_strategy_interface(strategy: Union[FilteringStrategy, JudgementStrategy]) -> bool:
    """
    전략 인터페이스 준수 여부 검증
    
    Args:
        strategy: 검증할 전략 인스턴스
        
    Returns:
        bool: 인터페이스 준수 여부
    """
    try:
        if isinstance(strategy, FilteringStrategy):
            # 필수 메서드 존재 여부 확인
            required_methods = ['apply', 'validate_input', 'get_strategy_info']
            for method in required_methods:
                if not hasattr(strategy, method) or not callable(getattr(strategy, method)):
                    logger.error(f"FilteringStrategy missing method: {method}")
                    return False
                    
        elif isinstance(strategy, JudgementStrategy):
            # 필수 메서드 존재 여부 확인
            required_methods = ['apply', 'detect_abnormal_stats', 'analyze_kpi_stats', 
                              'validate_input', 'get_strategy_info']
            for method in required_methods:
                if not hasattr(strategy, method) or not callable(getattr(strategy, method)):
                    logger.error(f"JudgementStrategy missing method: {method}")
                    return False
        else:
            logger.error("Unknown strategy type")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error validating strategy interface: {e}")
        return False


# =============================================================================
# 로깅 설정
# =============================================================================

logger.info("Choi Algorithm strategy interfaces loaded successfully")
