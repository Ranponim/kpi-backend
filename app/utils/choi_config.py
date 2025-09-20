"""
Choi 알고리즘 설정 로더 유틸리티

이 모듈은 config/choi_algorithm.yml 파일을 로드하고 파싱하여
타입 안전한 Pydantic BaseModel 모델로 변환하는 기능을 제공합니다.

주요 기능:
- YAML 설정 파일 로딩 및 파싱
- Pydantic 기반 타입 검증 및 변환
- 설정 유효성 검증
- 싱글톤 패턴으로 설정 인스턴스 관리
- 핫 리로드 지원 (개발 환경)

PRD 참조: 섹션 3.1.3 (설정 외부화), 작업 4 (설정 로더 구현)
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from functools import lru_cache
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# 설정 모델 정의 (Pydantic BaseModel)
# =============================================================================

class FilteringConfig(BaseModel):
    """필터링 알고리즘 설정 (6장)"""
    
    min_threshold: float = Field(0.87, description="정규화 최소 임계값")
    max_threshold: float = Field(1.13, description="정규화 최대 임계값")
    filter_ratio: float = Field(0.50, description="필터링 비율 임계값", ge=0.0, le=1.0)
    warning_message: str = Field(
        "TES can't filter the valid samples because test results are unstable",
        description="필터링 실패 경고 메시지"
    )
    reference_pegs: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict, 
        description="참조 PEG 목록 (nr/lte별)"
    )
    
    @field_validator('min_threshold', 'max_threshold')
    @classmethod
    def validate_thresholds(cls, v):
        """임계값 유효성 검증"""
        if not 0.1 <= v <= 2.0:
            raise ValueError(f"Threshold must be between 0.1 and 2.0, got {v}")
        return v
    
    @model_validator(mode='after')
    def validate_threshold_order(self):
        """임계값 순서 검증"""
        if self.min_threshold >= self.max_threshold:
            raise ValueError(f"min_threshold ({self.min_threshold}) must be less than max_threshold ({self.max_threshold})")
        return self


class AbnormalDetectionConfig(BaseModel):
    """이상 탐지 알고리즘 설정 (4장)"""
    
    alpha_0: int = Field(2, description="최소 셀 수 임계값", ge=1)
    alpha_1: int = Field(10, description="표에 정의된 값 (현재 미사용)", ge=1)
    beta_3: float = Field(500.0, description="High Delta 임계값", ge=0.0)
    
    enable_range_check: bool = Field(True, description="Range 검사 활성화")
    detection_types: Dict[str, bool] = Field(
        default_factory=lambda: {
            "range": True,
            "new": True, 
            "nd": True,
            "zero": True,
            "high_delta": True
        },
        description="이상 탐지 유형별 활성화"
    )


class StatsAnalyzingConfig(BaseModel):
    """통계 분석 알고리즘 설정 (5장)"""
    
    beta_0: float = Field(1000.0, description="트래픽량 임계값", ge=0.0)
    beta_1: float = Field(5.0, description="고트래픽 Similar 임계값", ge=0.0)
    beta_2: float = Field(10.0, description="저트래픽 Similar 임계값", ge=0.0)
    beta_3: float = Field(500.0, description="High Delta 임계값", ge=0.0)
    beta_4: float = Field(10.0, description="High Variation CV 임계값", ge=0.0)
    beta_5: float = Field(3.0, description="Similar 절대값 임계값", ge=0.0)
    
    rule_priorities: Dict[str, int] = Field(
        default_factory=lambda: {
            "cant_judge": 100,
            "high_variation": 90,
            "improve": 80,
            "degrade": 80,
            "high_delta": 70,
            "medium_delta": 60,
            "low_delta": 50,
            "similar": 40
        },
        description="판정 규칙 우선순위"
    )
    
    @model_validator(mode='after')
    def validate_beta_order(self):
        """β1 < β2 조건 검증"""
        if self.beta_2 <= self.beta_1:
            logger.warning(f"beta_2 ({self.beta_2}) should be greater than beta_1 ({self.beta_1})")
        return self


class KPITopicConfig(BaseModel):
    """KPI 토픽 정의"""
    
    main_kpi: Dict[str, Any] = Field(..., description="Main KPI 정보")
    sub_kpis: List[Dict[str, Any]] = Field(default_factory=list, description="Sub KPI 목록")


class KPIDefinitionsConfig(BaseModel):
    """KPI 정의 설정 (2.4장)"""
    
    lte_topics: Dict[str, KPITopicConfig] = Field(default_factory=dict, description="LTE KPI 토픽")
    nr_topics: Dict[str, KPITopicConfig] = Field(default_factory=dict, description="NR KPI 토픽")


class GeneralConfig(BaseModel):
    """일반 설정"""
    
    logging: Dict[str, Any] = Field(
        default_factory=lambda: {
            "level": "INFO",
            "detailed_reasoning": True
        },
        description="로깅 설정"
    )
    performance: Dict[str, Any] = Field(
        default_factory=lambda: {
            "max_processing_time": 5.0,
            "enable_caching": True
        },
        description="성능 설정"
    )
    data_quality: Dict[str, Any] = Field(
        default_factory=lambda: {
            "min_sample_size": 10,
            "max_nd_ratio": 0.5,
            "max_zero_ratio": 0.3
        },
        description="데이터 품질 설정"
    )


class ChoiAlgorithmConfig(BaseModel):
    """
    Choi 알고리즘 전체 설정
    
    config/choi_algorithm.yml 파일의 모든 설정을 타입 안전하게 관리
    """
    
    filtering: FilteringConfig = Field(default_factory=FilteringConfig)
    abnormal_detection: AbnormalDetectionConfig = Field(default_factory=AbnormalDetectionConfig)
    stats_analyzing: StatsAnalyzingConfig = Field(default_factory=StatsAnalyzingConfig)
    kpi_definitions: KPIDefinitionsConfig = Field(default_factory=KPIDefinitionsConfig)
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    
    # 메타데이터
    config_version: str = Field("1.0.0", description="설정 파일 버전")
    loaded_at: datetime = Field(default_factory=datetime.now, description="로딩 시각")
    config_file_path: Optional[str] = Field(None, description="설정 파일 경로")
    
    class Config:
        """Pydantic 설정"""
        env_prefix = "CHOI_"  # 환경변수 접두사
        case_sensitive = False
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# 설정 로더 클래스
# =============================================================================

class ChoiConfigLoader:
    """
    Choi 알고리즘 설정 로더
    
    싱글톤 패턴을 사용하여 설정을 로드하고 관리합니다.
    """
    
    _instance = None
    _config = None
    _config_file_path = None
    _last_modified = None
    
    def __new__(cls):
        """싱글톤 인스턴스 생성"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.logger = logging.getLogger(f"{__name__}.{cls.__name__}")
        return cls._instance
    
    def load_config(self, config_file_path: Optional[str] = None) -> ChoiAlgorithmConfig:
        """
        설정 파일 로드
        
        Args:
            config_file_path: 설정 파일 경로 (None이면 기본 경로 사용)
            
        Returns:
            ChoiAlgorithmConfig: 로드된 설정 객체
            
        Raises:
            FileNotFoundError: 설정 파일이 없는 경우
            yaml.YAMLError: YAML 파싱 오류
            ValueError: 설정 검증 오류
        """
        try:
            # 기본 설정 파일 경로 결정
            if config_file_path is None:
                config_file_path = self._get_default_config_path()
            
            # 파일 존재 여부 확인
            if not os.path.exists(config_file_path):
                raise FileNotFoundError(f"Configuration file not found: {config_file_path}")
            
            # 파일 수정 시간 확인 (핫 리로드 지원)
            current_mtime = os.path.getmtime(config_file_path)
            if (self._config is not None and 
                self._config_file_path == config_file_path and 
                self._last_modified == current_mtime):
                self.logger.debug("Configuration already loaded and up-to-date")
                return self._config
            
            # YAML 파일 로드
            self.logger.info(f"Loading configuration from: {config_file_path}")
            with open(config_file_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
            
            if not yaml_data:
                raise ValueError("Empty configuration file")
            
            # Pydantic 모델로 변환
            config = self._parse_yaml_to_config(yaml_data, config_file_path)
            
            # 설정 검증
            self._validate_config(config)
            
            # 캐시 업데이트
            self._config = config
            self._config_file_path = config_file_path
            self._last_modified = current_mtime
            
            self.logger.info(f"Configuration loaded successfully from: {config_file_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _get_default_config_path(self) -> str:
        """기본 설정 파일 경로 반환"""
        # 프로젝트 루트 기준으로 경로 계산
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent  # app/utils -> app -> backend
        config_path = project_root / "config" / "choi_algorithm.yml"
        return str(config_path)
    
    def _parse_yaml_to_config(self, yaml_data: Dict[str, Any], file_path: str) -> ChoiAlgorithmConfig:
        """
        YAML 데이터를 Pydantic 설정 객체로 변환
        
        Args:
            yaml_data: YAML에서 로드된 딕셔너리
            file_path: 설정 파일 경로
            
        Returns:
            ChoiAlgorithmConfig: 변환된 설정 객체
        """
        try:
            # 메타데이터 추가
            yaml_data['config_file_path'] = file_path
            yaml_data['loaded_at'] = datetime.now()
            
            # 중첩된 설정 객체들 개별 파싱
            if 'filtering' in yaml_data:
                yaml_data['filtering'] = FilteringConfig(**yaml_data['filtering'])
            
            if 'abnormal_detection' in yaml_data:
                yaml_data['abnormal_detection'] = AbnormalDetectionConfig(**yaml_data['abnormal_detection'])
            
            if 'stats_analyzing' in yaml_data:
                yaml_data['stats_analyzing'] = StatsAnalyzingConfig(**yaml_data['stats_analyzing'])
            
            if 'kpi_definitions' in yaml_data:
                kpi_data = yaml_data['kpi_definitions']
                
                # LTE/NR 토픽 개별 파싱
                if 'lte_topics' in kpi_data:
                    lte_topics = {}
                    for topic_name, topic_data in kpi_data['lte_topics'].items():
                        lte_topics[topic_name] = KPITopicConfig(**topic_data)
                    kpi_data['lte_topics'] = lte_topics
                
                if 'nr_topics' in kpi_data:
                    nr_topics = {}
                    for topic_name, topic_data in kpi_data['nr_topics'].items():
                        nr_topics[topic_name] = KPITopicConfig(**topic_data)
                    kpi_data['nr_topics'] = nr_topics
                
                yaml_data['kpi_definitions'] = KPIDefinitionsConfig(**kpi_data)
            
            if 'general' in yaml_data:
                yaml_data['general'] = GeneralConfig(**yaml_data['general'])
            
            # 전체 설정 객체 생성
            config = ChoiAlgorithmConfig(**yaml_data)
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to parse YAML to config: {e}")
            raise ValueError(f"Configuration parsing error: {e}")
    
    def _validate_config(self, config: ChoiAlgorithmConfig) -> None:
        """
        설정 유효성 검증
        
        Args:
            config: 검증할 설정 객체
            
        Raises:
            ValueError: 설정이 유효하지 않은 경우
        """
        try:
            # 기본 구조 검증
            required_sections = ['filtering', 'abnormal_detection', 'stats_analyzing']
            for section in required_sections:
                if not hasattr(config, section):
                    raise ValueError(f"Required configuration section missing: {section}")
            
            # 임계값 관계 검증
            filtering = config.filtering
            if filtering.min_threshold >= filtering.max_threshold:
                raise ValueError("filtering.min_threshold must be less than max_threshold")
            
            stats = config.stats_analyzing
            if stats.beta_1 > stats.beta_2:
                self.logger.warning("beta_1 should typically be less than beta_2")
            
            # KPI 정의 검증
            kpi_defs = config.kpi_definitions
            if not kpi_defs.lte_topics and not kpi_defs.nr_topics:
                self.logger.warning("No KPI topics defined")
            
            self.logger.debug("Configuration validation passed")
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise
    
    def get_config(self) -> Optional[ChoiAlgorithmConfig]:
        """현재 로드된 설정 반환"""
        return self._config
    
    def reload_config(self) -> ChoiAlgorithmConfig:
        """설정 강제 리로드"""
        self.logger.info("Force reloading configuration")
        self._config = None
        self._last_modified = None
        return self.load_config(self._config_file_path)


# =============================================================================
# 편의 함수들
# =============================================================================

@lru_cache(maxsize=1)
def get_config_loader() -> ChoiConfigLoader:
    """
    설정 로더 싱글톤 인스턴스 반환
    
    Returns:
        ChoiConfigLoader: 설정 로더 인스턴스
    """
    return ChoiConfigLoader()


def load_choi_config(config_file_path: Optional[str] = None) -> ChoiAlgorithmConfig:
    """
    Choi 알고리즘 설정 로드 (편의 함수)
    
    Args:
        config_file_path: 설정 파일 경로 (None이면 기본 경로)
        
    Returns:
        ChoiAlgorithmConfig: 로드된 설정 객체
    """
    loader = get_config_loader()
    return loader.load_config(config_file_path)


def get_choi_config() -> Optional[ChoiAlgorithmConfig]:
    """
    현재 로드된 Choi 알고리즘 설정 반환
    
    Returns:
        ChoiAlgorithmConfig: 현재 설정 객체 (없으면 None)
    """
    loader = get_config_loader()
    return loader.get_config()


def validate_config_file(config_file_path: str) -> bool:
    """
    설정 파일 유효성 검증 (로드하지 않고)
    
    Args:
        config_file_path: 검증할 설정 파일 경로
        
    Returns:
        bool: 유효성 여부
    """
    try:
        loader = ChoiConfigLoader()
        loader.load_config(config_file_path)
        return True
    except Exception as e:
        logger.error(f"Config validation failed: {e}")
        return False


# =============================================================================
# 초기화 및 로깅
# =============================================================================

logger.info("Choi Algorithm configuration loader initialized")
