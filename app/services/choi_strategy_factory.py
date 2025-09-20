"""
Choi 알고리즘 Strategy 팩토리 모듈

이 모듈은 Choi 알고리즘의 모든 Strategy 인스턴스를 생성하고 관리하는
팩토리 클래스를 제공합니다. 완전한 의존성 주입과 설정 관리를 지원합니다.

Author: Choi Algorithm Implementation Team
Created: 2025-09-20
"""

import logging
from typing import Dict, Any, Optional
from functools import lru_cache

# Strategy 인터페이스
from app.services.strategies import FilteringStrategy, JudgementStrategy

# 구체적인 Strategy 구현체들
from app.services.choi_filtering_service import ChoiFiltering
from app.services.choi_judgement_service import ChoiJudgement

# 의존성들
from app.services.anomaly_detectors import AnomalyDetectorFactory, DimsDataProvider, MockDimsDataProvider
from app.services.kpi_analyzers import KPIAnalyzerFactory

# 설정 관리
from app.utils.choi_config import load_choi_config, ChoiAlgorithmConfig

# 로깅
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# DIMS Data Provider 팩토리 (실제 구현 vs Mock)
# =============================================================================

class DimsDataProviderFactory:
    """
    DIMS Data Provider 팩토리
    
    실제 환경에서는 실제 DIMS 연동 구현체를,
    테스트/개발 환경에서는 Mock 구현체를 제공합니다.
    
    Single Responsibility: DIMS 제공자 생성만 담당
    """
    
    @staticmethod
    def create_provider(config: Optional[Dict[str, Any]] = None) -> DimsDataProvider:
        """
        DIMS Data Provider 생성
        
        Args:
            config: 설정 정보 (실제 DIMS 연결 정보 등)
            
        Returns:
            DimsDataProvider: DIMS 데이터 제공자 인스턴스
        """
        try:
            # 현재는 Mock 구현체만 제공
            # TODO: 실제 DIMS 연동 구현체 추가 예정
            logger.info("Creating Mock DIMS Data Provider")
            return MockDimsDataProvider()
            
        except Exception as e:
            logger.error(f"DIMS Data Provider 생성 실패: {e}")
            # 실패 시 Mock으로 fallback
            logger.warning("Falling back to Mock DIMS Data Provider")
            return MockDimsDataProvider()


# =============================================================================
# Choi Strategy 통합 팩토리
# =============================================================================

class ChoiStrategyFactory:
    """
    Choi 알고리즘 Strategy 통합 팩토리
    
    모든 Strategy 인스턴스를 생성하고 의존성을 주입하는 중앙화된 팩토리입니다.
    설정 로딩, 인스턴스 캐싱, 오류 처리를 통합 관리합니다.
    
    Factory Pattern + Dependency Injection + Singleton Pattern 적용
    """
    
    _instance: Optional['ChoiStrategyFactory'] = None
    _config: Optional[ChoiAlgorithmConfig] = None
    _filtering_strategy: Optional[FilteringStrategy] = None
    _judgement_strategy: Optional[JudgementStrategy] = None
    
    def __new__(cls) -> 'ChoiStrategyFactory':
        """싱글톤 패턴 구현"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """팩토리 초기화 (싱글톤이므로 한 번만 실행)"""
        if self._initialized:
            return
        
        self.logger = get_logger(__name__)
        self._initialized = True
        
        # 설정 로드
        self._load_configuration()
        
        self.logger.info("ChoiStrategyFactory initialized successfully")
    
    def _load_configuration(self) -> None:
        """
        Choi 알고리즘 설정 로드
        
        설정 로딩 실패 시 기본값으로 fallback하여 시스템 안정성 보장
        """
        try:
            self._config = load_choi_config()
            self.logger.info("Choi 알고리즘 설정 로드 성공")
            
            # 설정 검증
            if not self._validate_configuration():
                raise ValueError("Configuration validation failed")
                
        except Exception as e:
            self.logger.error(f"설정 로드 실패: {e}")
            self.logger.warning("기본 설정으로 fallback")
            
            # 기본 설정으로 fallback (방어적 프로그래밍)
            self._config = self._create_default_config()
    
    def _validate_configuration(self) -> bool:
        """설정 유효성 검증"""
        try:
            if not self._config:
                return False
            
            # 필수 설정 항목 검증
            required_sections = ['filtering', 'abnormal_detection', 'stats_analyzing']
            for section in required_sections:
                if not hasattr(self._config, section):
                    self.logger.error(f"Missing required config section: {section}")
                    return False
            
            # 임계값 유효성 검증
            filtering_config = self._config.filtering
            if not (0 < filtering_config.min_threshold < filtering_config.max_threshold):
                self.logger.error("Invalid filtering thresholds")
                return False
            
            self.logger.debug("Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            return False
    
    def _create_default_config(self) -> ChoiAlgorithmConfig:
        """기본 설정 생성 (fallback용)"""
        from app.utils.choi_config import (
            FilteringConfig, AbnormalDetectionConfig, StatsAnalyzingConfig,
            KPIDefinitionsConfig, GeneralConfig, ChoiAlgorithmConfig
        )
        
        return ChoiAlgorithmConfig(
            filtering=FilteringConfig(
                min_threshold=0.87,
                max_threshold=1.13,
                filter_ratio=0.50
            ),
            abnormal_detection=AbnormalDetectionConfig(
                alpha_0=2,
                beta_3=500.0
            ),
            stats_analyzing=StatsAnalyzingConfig(
                beta_0=1000.0,
                beta_1=5.0,
                beta_2=10.0,
                beta_4=10.0,
                beta_5=3.0
            ),
            kpi_definitions=KPIDefinitionsConfig(
                lte_topics=[],
                nr_topics=[],
                kpi_positivity_map={}
            ),
            general=GeneralConfig()
        )
    
    @lru_cache(maxsize=1)
    def create_filtering_strategy(self) -> FilteringStrategy:
        """
        필터링 Strategy 생성 (캐시됨)
        
        Returns:
            FilteringStrategy: Choi 필터링 구현체
        """
        try:
            if self._filtering_strategy is None:
                self.logger.info("Creating Choi Filtering Strategy")
                self._filtering_strategy = ChoiFiltering()
                
                # 설정 검증 (빈 데이터는 검증하지 않고 설정만 검증)
                test_config = self._get_filtering_config_dict()
                self.logger.debug(f"Filtering strategy created with config: {test_config}")
                
                self.logger.info("Choi Filtering Strategy created successfully")
            
            return self._filtering_strategy
            
        except Exception as e:
            self.logger.error(f"Filtering Strategy 생성 실패: {e}")
            raise
    
    @lru_cache(maxsize=1)
    def create_judgement_strategy(self) -> JudgementStrategy:
        """
        판정 Strategy 생성 (캐시됨)
        
        Returns:
            JudgementStrategy: Choi 판정 구현체 (완전한 의존성 주입)
        """
        try:
            if self._judgement_strategy is None:
                self.logger.info("Creating Choi Judgement Strategy with full dependency injection")
                
                # 의존성들 생성
                dims_provider = DimsDataProviderFactory.create_provider()
                detector_factory = AnomalyDetectorFactory(dims_provider)
                analyzer_factory = KPIAnalyzerFactory()
                
                # 완전한 의존성 주입으로 생성
                self._judgement_strategy = ChoiJudgement(
                    detector_factory=detector_factory,
                    analyzer_factory=analyzer_factory,
                    dims_provider=dims_provider
                )
                
                # 설정 검증 (빈 데이터는 검증하지 않고 설정만 검증)
                test_config = self._get_judgement_config_dict()
                self.logger.debug(f"Judgement strategy created with config keys: {list(test_config.keys())}")
                
                self.logger.info("Choi Judgement Strategy created successfully with all dependencies")
            
            return self._judgement_strategy
            
        except Exception as e:
            self.logger.error(f"Judgement Strategy 생성 실패: {e}")
            raise
    
    def create_strategy_pair(self) -> tuple[FilteringStrategy, JudgementStrategy]:
        """
        필터링과 판정 Strategy 쌍 생성
        
        Returns:
            tuple: (FilteringStrategy, JudgementStrategy) 쌍
        """
        try:
            self.logger.info("Creating complete Choi Strategy pair")
            
            filtering = self.create_filtering_strategy()
            judgement = self.create_judgement_strategy()
            
            self.logger.info("Complete Choi Strategy pair created successfully")
            return filtering, judgement
            
        except Exception as e:
            self.logger.error(f"Strategy pair 생성 실패: {e}")
            raise
    
    def get_configuration(self) -> ChoiAlgorithmConfig:
        """현재 설정 반환"""
        return self._config
    
    def _get_filtering_config_dict(self) -> Dict[str, Any]:
        """필터링용 설정 딕셔너리 생성"""
        if not self._config:
            return {}
        
        return {
            'min_threshold': self._config.filtering.min_threshold,
            'max_threshold': self._config.filtering.max_threshold,
            'filter_ratio': self._config.filtering.filter_ratio,
            'warning_message': 'Generated by ChoiStrategyFactory'
        }
    
    def _get_judgement_config_dict(self) -> Dict[str, Any]:
        """판정용 설정 딕셔너리 생성"""
        if not self._config:
            return {}
        
        return {
            # 이상 탐지 설정
            'alpha_0': self._config.abnormal_detection.alpha_0,
            'beta_3': self._config.abnormal_detection.beta_3,
            
            # 통계 분석 설정
            'beta_0': self._config.stats_analyzing.beta_0,
            'beta_1': self._config.stats_analyzing.beta_1,
            'beta_2': self._config.stats_analyzing.beta_2,
            'beta_4': self._config.stats_analyzing.beta_4,
            'beta_5': self._config.stats_analyzing.beta_5,
            
            # 기능 활성화 설정 (기본값으로 모두 활성화)
            'enable_range_check': True,
            'enable_new_check': True,
            'enable_nd_check': True,
            'enable_zero_check': True,
            'enable_high_delta_check': True,
        }
    
    def reload_configuration(self) -> bool:
        """
        설정 다시 로드 (Hot Reload)
        
        Returns:
            bool: 성공 여부
        """
        try:
            self.logger.info("Reloading Choi algorithm configuration")
            
            # 캐시 클리어
            self.create_filtering_strategy.cache_clear()
            self.create_judgement_strategy.cache_clear()
            
            # 인스턴스 초기화
            self._filtering_strategy = None
            self._judgement_strategy = None
            
            # 설정 다시 로드
            self._load_configuration()
            
            self.logger.info("Configuration reloaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration reload failed: {e}")
            return False
    
    def get_factory_info(self) -> Dict[str, Any]:
        """팩토리 정보 반환 (디버깅/모니터링용)"""
        return {
            "factory_type": "ChoiStrategyFactory",
            "config_loaded": self._config is not None,
            "filtering_cached": self._filtering_strategy is not None,
            "judgement_cached": self._judgement_strategy is not None,
            "algorithm_version": "1.0.0",  # 기본 버전
            "cache_info": {
                "filtering": self.create_filtering_strategy.cache_info(),
                "judgement": self.create_judgement_strategy.cache_info()
            }
        }


# =============================================================================
# 편의 함수들
# =============================================================================

@lru_cache(maxsize=1)
def get_choi_strategy_factory() -> ChoiStrategyFactory:
    """
    전역 ChoiStrategyFactory 인스턴스 반환 (싱글톤)
    
    Returns:
        ChoiStrategyFactory: 싱글톤 팩토리 인스턴스
    """
    return ChoiStrategyFactory()


def create_choi_strategies() -> tuple[FilteringStrategy, JudgementStrategy]:
    """
    Choi Strategy 쌍 생성 편의 함수
    
    Returns:
        tuple: (FilteringStrategy, JudgementStrategy) 쌍
    """
    factory = get_choi_strategy_factory()
    return factory.create_strategy_pair()


def get_choi_configuration() -> ChoiAlgorithmConfig:
    """
    현재 Choi 설정 반환 편의 함수
    
    Returns:
        ChoiAlgorithmConfig: 현재 설정
    """
    factory = get_choi_strategy_factory()
    return factory.get_configuration()


# =============================================================================
# 초기화 및 로깅
# =============================================================================

logger.info("Choi Strategy Factory module loaded successfully")
