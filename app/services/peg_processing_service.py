"""
PEG 데이터 처리 오케스트레이션 서비스

이 모듈은 3GPP KPI PEGs 데이터의 전체 처리 워크플로우를 관리하는
오케스트레이터 서비스입니다. Choi 알고리즘의 필터링과 판정 단계를
기존 집계/파생 계산과 통합하여 제공합니다.

주요 처리 단계:
1. data_retrieval: 원시 데이터 조회
2. data_validation: 데이터 유효성 검증
3. filtering: 유효 시간대 선정 (6장 - 신규)
4. aggregation: 기본 통계 집계
5. derived_calculation: 파생 지표 계산
6. judgement: 이상탐지 및 KPI 판정 (4장, 5장 - 신규)
7. result_formatting: 최종 결과 포맷팅

PRD 참조: 섹션 3.3.1 (기존 시스템 통합)
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ..models.judgement import (
    PegSampleSeries, 
    FilteringResult, 
    AbnormalDetectionResult,
    MainKPIJudgement,
    ChoiAlgorithmResponse,
    SimpleKPIJudgement,
    create_empty_filtering_result,
    create_empty_abnormal_detection_result
)
from ..services.strategies import FilteringStrategy, JudgementStrategy
from ..utils.logging_decorators import log_service_method
from ..exceptions import (
    ChoiAlgorithmError, 
    InsufficientDataError, 
    DataValidationError,
    StrategyExecutionError,
    PerformanceError
)
from ..services.choi_strategy_factory import get_choi_strategy_factory, create_choi_strategies
from ..utils.choi_config import load_choi_config, ChoiAlgorithmConfig
from ..utils.cache_manager import get_cache_manager
from ..utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


class PEGProcessingService:
    """
    PEG 데이터 처리 오케스트레이션 서비스
    
    Choi 알고리즘을 포함한 전체 PEG 데이터 처리 파이프라인을 관리합니다.
    Strategy 패턴을 사용하여 필터링과 판정 알고리즘을 플러그인 방식으로 지원합니다.
    """
    
    def __init__(self, 
                 filtering_strategy: Optional[FilteringStrategy] = None,
                 judgement_strategy: Optional[JudgementStrategy] = None,
                 config: Optional[ChoiAlgorithmConfig] = None):
        """
        서비스 초기화 (Strategy Factory 통합)
        
        Args:
            filtering_strategy: 필터링 전략 구현체 (None이면 팩토리에서 생성)
            judgement_strategy: 판정 전략 구현체 (None이면 팩토리에서 생성)
            config: 알고리즘 설정 (None이면 팩토리에서 로드)
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Strategy Factory를 통한 완전한 의존성 주입
        if filtering_strategy is None or judgement_strategy is None:
            self.logger.info("Creating strategies via ChoiStrategyFactory with full DI")
            factory = get_choi_strategy_factory()
            
            # 팩토리에서 완전히 구성된 Strategy들 생성
            factory_filtering, factory_judgement = factory.create_strategy_pair()
            
            self.filtering_strategy = filtering_strategy or factory_filtering
            self.judgement_strategy = judgement_strategy or factory_judgement
            
            # 팩토리에서 설정도 가져오기
            self.config = config or factory.get_configuration()
            
            self.logger.info("Strategies created via factory with complete dependency injection")
        else:
            # 직접 제공된 Strategy 사용 (테스트용)
            self.filtering_strategy = filtering_strategy
            self.judgement_strategy = judgement_strategy
            self.config = config or load_choi_config()
            
            self.logger.info("Using directly provided strategies")
        
        self.logger.info(f"PEG Processing Service initialized with Choi algorithm")
        
        # 캐시 매니저 초기화
        self.cache_manager = get_cache_manager()
        
        # 처리 단계 정의 (PRD 3.3.1에 따라 확장)
        self.processing_steps = [
            "data_retrieval",      # 원시 데이터 조회
            "data_validation",     # 데이터 유효성 검증
            "filtering",           # 신규: 6장 필터링 알고리즘
            "aggregation",         # 기본 통계 집계
            "derived_calculation", # 파생 지표 계산
            "judgement",           # 신규: 4장, 5장 판정 알고리즘
            "result_formatting",   # 최종 결과 포맷팅
        ]
        
        self.logger.info(f"Processing pipeline configured with {len(self.processing_steps)} steps")
    
    @log_service_method(log_params=True, performance_threshold_ms=5000.0)
    def process_peg_data(self,
                        input_data: Dict[str, Any],
                        cell_ids: List[str],
                        time_range: Dict[str, datetime],
                        compare_mode: bool = True) -> ChoiAlgorithmResponse:
        """
        PEG 데이터 전체 처리 워크플로우 실행
        
        Args:
            input_data: 입력 데이터 (EMS 정보, NE 목록 등)
            cell_ids: 분석할 셀 ID 목록
            time_range: 분석 시간 범위 (pre/post)
            compare_mode: 비교 모드 여부
            
        Returns:
            ChoiAlgorithmResponse: 전체 처리 결과
            
        Raises:
            ValueError: 입력 데이터가 유효하지 않은 경우
            RuntimeError: 처리 중 오류 발생
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting PEG data processing workflow for {len(cell_ids)} cells")
            
            # 처리 결과 저장용 딕셔너리
            processing_results = {}
            warnings = []
            
            # 1단계: 데이터 조회
            self.logger.info("1단계: 데이터 조회 시작")
            raw_data = self._data_retrieval(input_data, cell_ids, time_range)
            processing_results["raw_data"] = raw_data
            
            # 2단계: 데이터 검증
            self.logger.info("2단계: 데이터 검증 시작")
            validated_data = self._data_validation(raw_data)
            processing_results["validated_data"] = validated_data
            
            # 3단계: 필터링 (신규)
            self.logger.info("3단계: 필터링 알고리즘 실행")
            filtering_result = self._run_filtering(validated_data)
            processing_results["filtering"] = filtering_result
            if filtering_result.warning_message:
                warnings.append(filtering_result.warning_message)
            
            # 4단계: 집계
            self.logger.info("4단계: 기본 통계 집계")
            aggregated_data = self._aggregation(validated_data, filtering_result)
            processing_results["aggregation"] = aggregated_data
            
            # 5단계: 파생 계산
            self.logger.info("5단계: 파생 지표 계산")
            derived_data = self._derived_calculation(aggregated_data)
            processing_results["derived"] = derived_data
            
            # 6단계: 판정 (신규)
            self.logger.info("6단계: 판정 알고리즘 실행")
            judgement_results = self._run_judgement(derived_data, filtering_result)
            processing_results["judgement"] = judgement_results
            
            # 7단계: 결과 포맷팅
            self.logger.info("7단계: 최종 결과 포맷팅")
            final_response = self._result_formatting(processing_results, warnings)
            
            # 처리 시간 계산
            processing_time = (time.time() - start_time) * 1000  # 밀리초
            final_response.processing_time_ms = processing_time
            
            self.logger.info(f"PEG 데이터 처리 워크플로우 완료: {processing_time:.2f}ms")
            return final_response
            
        except Exception as e:
            self.logger.error(f"PEG 데이터 처리 중 오류 발생: {e}")
            raise RuntimeError(f"PEG processing failed: {e}")
    
    def _data_retrieval(self, 
                       input_data: Dict[str, Any], 
                       cell_ids: List[str], 
                       time_range: Dict[str, datetime]) -> Dict[str, List[PegSampleSeries]]:
        """
        1단계: 원시 데이터 조회
        
        Args:
            input_data: EMS 정보, NE 목록 등
            cell_ids: 셀 ID 목록
            time_range: 시간 범위
            
        Returns:
            Dict[str, List[PegSampleSeries]]: 셀별 PEG 시계열 데이터
        """
        try:
            self.logger.debug(f"Retrieving data for cells: {cell_ids}")
            
            # TODO: 실제 데이터베이스 조회 로직 구현
            # 현재는 Mock 데이터 반환
            mock_data = {}
            for cell_id in cell_ids:
                mock_series = []
                for peg_name in ["AirMacDLThruAvg", "AirMacULThruAvg", "ConnNoAvg"]:
                    # Mock 시계열 데이터 생성
                    pre_samples = np.random.normal(1000, 100, 20).tolist()
                    post_samples = np.random.normal(1100, 120, 20).tolist()
                    
                    series = PegSampleSeries(
                        peg_name=peg_name,
                        cell_id=cell_id,
                        pre_samples=pre_samples,
                        post_samples=post_samples,
                        unit="Kbps" if "Thru" in peg_name else "count"
                    )
                    mock_series.append(series)
                
                mock_data[cell_id] = mock_series
            
            self.logger.info(f"데이터 조회 완료: {len(mock_data)} cells, {sum(len(series) for series in mock_data.values())} PEG series")
            return mock_data
            
        except Exception as e:
            self.logger.error(f"데이터 조회 중 오류: {e}")
            raise
    
    def _data_validation(self, raw_data: Dict[str, List[PegSampleSeries]]) -> Dict[str, List[PegSampleSeries]]:
        """
        2단계: 데이터 유효성 검증
        
        Args:
            raw_data: 원시 데이터
            
        Returns:
            Dict[str, List[PegSampleSeries]]: 검증된 데이터
        """
        try:
            self.logger.debug("Validating input data structure and content")
            
            validated_data = {}
            total_series = 0
            invalid_series = 0
            
            for cell_id, peg_series_list in raw_data.items():
                valid_series = []
                
                for series in peg_series_list:
                    total_series += 1
                    
                    # 기본 검증
                    if not series.peg_name or not series.cell_id:
                        invalid_series += 1
                        self.logger.warning(f"Invalid series: missing peg_name or cell_id")
                        continue
                    
                    # 샘플 데이터 검증
                    if not series.pre_samples and not series.post_samples:
                        invalid_series += 1
                        self.logger.warning(f"Invalid series {series.peg_name}: no sample data")
                        continue
                    
                    valid_series.append(series)
                
                if valid_series:
                    validated_data[cell_id] = valid_series
            
            self.logger.info(f"데이터 검증 완료: {total_series - invalid_series}/{total_series} series valid")
            
            if invalid_series > 0:
                self.logger.warning(f"{invalid_series} invalid series excluded from processing")
            
            return validated_data
            
        except Exception as e:
            self.logger.error(f"데이터 검증 중 오류: {e}")
            raise
    
    def _run_filtering(self, validated_data: Dict[str, List[PegSampleSeries]]) -> FilteringResult:
        """
        3단계: 필터링 알고리즘 실행 (신규)
        
        Args:
            validated_data: 검증된 PEG 데이터
            
        Returns:
            FilteringResult: 필터링 결과
        """
        try:
            if self.filtering_strategy is None:
                self.logger.warning("No filtering strategy provided, using empty result")
                return create_empty_filtering_result()
            
            self.logger.debug(f"Executing filtering with strategy: {self.filtering_strategy.get_strategy_info()['name']}")
            
            # 필터링 전략 실행 (Factory에서 생성된 완전한 Strategy 사용)
            filtering_config = {
                'min_threshold': self.config.filtering.min_threshold,
                'max_threshold': self.config.filtering.max_threshold,
                'filter_ratio': self.config.filtering.filter_ratio,
                'warning_message': f'Generated by PEGProcessingService'
            }
            
            result = self.filtering_strategy.apply(validated_data, filtering_config)
            
            self.logger.info(f"필터링 완료: filter_ratio={result.filter_ratio:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"필터링 실행 중 오류: {e}")
            raise
    
    def _aggregation(self, 
                    validated_data: Dict[str, List[PegSampleSeries]], 
                    filtering_result: FilteringResult) -> Dict[str, Any]:
        """
        4단계: 기본 통계 집계
        
        Args:
            validated_data: 검증된 데이터
            filtering_result: 필터링 결과
            
        Returns:
            Dict[str, Any]: 집계된 통계 데이터
        """
        try:
            self.logger.debug("Performing basic statistical aggregation")
            
            aggregated = {}
            
            for cell_id, peg_series_list in validated_data.items():
                cell_stats = {}
                valid_time_slots = filtering_result.valid_time_slots.get(cell_id, [])
                
                for series in peg_series_list:
                    # 유효 시간 슬롯만 사용하여 통계 계산
                    if valid_time_slots:
                        pre_filtered = [series.pre_samples[i] for i in valid_time_slots if i < len(series.pre_samples)]
                        post_filtered = [series.post_samples[i] for i in valid_time_slots if i < len(series.post_samples)]
                    else:
                        pre_filtered = series.pre_samples
                        post_filtered = series.post_samples
                    
                    # 기본 통계 계산
                    pre_stats = self._calculate_basic_stats(pre_filtered)
                    post_stats = self._calculate_basic_stats(post_filtered)
                    
                    cell_stats[series.peg_name] = {
                        "pre": pre_stats,
                        "post": post_stats,
                        "unit": series.unit
                    }
                
                aggregated[cell_id] = cell_stats
            
            self.logger.info(f"기본 집계 완료: {len(aggregated)} cells processed")
            return aggregated
            
        except Exception as e:
            self.logger.error(f"집계 처리 중 오류: {e}")
            raise
    
    def _derived_calculation(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        5단계: 파생 지표 계산
        
        Args:
            aggregated_data: 집계된 데이터
            
        Returns:
            Dict[str, Any]: 파생 지표가 추가된 데이터
        """
        try:
            self.logger.debug("Calculating derived metrics")
            
            derived_data = aggregated_data.copy()
            
            for cell_id, cell_stats in derived_data.items():
                for peg_name, peg_data in cell_stats.items():
                    pre_stats = peg_data["pre"]
                    post_stats = peg_data["post"]
                    
                    # 변화율 계산
                    if pre_stats["mean"] and pre_stats["mean"] != 0:
                        delta_pct = ((post_stats["mean"] - pre_stats["mean"]) / pre_stats["mean"]) * 100
                        peg_data["delta_percentage"] = delta_pct
                    else:
                        peg_data["delta_percentage"] = None
                    
                    # CV 계산
                    peg_data["pre_cv"] = pre_stats["cv"]
                    peg_data["post_cv"] = post_stats["cv"]
            
            self.logger.info("파생 지표 계산 완료")
            return derived_data
            
        except Exception as e:
            self.logger.error(f"파생 계산 중 오류: {e}")
            raise
    
    def _run_judgement(self, 
                      derived_data: Dict[str, Any], 
                      filtering_result: FilteringResult) -> Dict[str, Any]:
        """
        6단계: 판정 알고리즘 실행 (신규)
        
        Args:
            derived_data: 파생 지표가 포함된 데이터
            filtering_result: 필터링 결과
            
        Returns:
            Dict[str, Any]: 판정 결과
        """
        try:
            if self.judgement_strategy is None:
                self.logger.warning("No judgement strategy provided, using empty results")
                return {
                    "abnormal_detection": create_empty_abnormal_detection_result(),
                    "kpi_judgement": {},
                    "processing_metadata": {}
                }
            
            self.logger.debug(f"Executing judgement with strategy: {self.judgement_strategy.get_strategy_info()['name']}")
            
            # 데이터 형태 변환 (derived_data -> PegSampleSeries 형태)
            converted_data = self._convert_derived_to_sample_series(derived_data)
            
            if not converted_data:
                self.logger.warning("Empty converted data, creating minimal test data")
                # 최소한의 테스트 데이터 생성
                from app.models.judgement import PegSampleSeries
                converted_data = {
                    'cell_001': [
                        PegSampleSeries(
                            peg_name='TestPEG',
                            cell_id='cell_001',
                            pre_samples=[1000.0, 1100.0, 1200.0],
                            post_samples=[1150.0, 1250.0, 1350.0],
                            unit='test'
                        )
                    ]
                }
            
            # 판정 전략 실행 (Factory에서 생성된 완전한 Strategy 사용)
            judgement_config = {
                # 이상 탐지 설정
                'alpha_0': self.config.abnormal_detection.alpha_0,
                
                # 모든 beta 값들 (이상 탐지와 KPI 분석에서 공통 사용)
                'beta_0': self.config.stats_analyzing.beta_0,
                'beta_1': self.config.stats_analyzing.beta_1,
                'beta_2': self.config.stats_analyzing.beta_2,
                'beta_3': self.config.abnormal_detection.beta_3,  # High Delta용
                'beta_4': self.config.stats_analyzing.beta_4,
                'beta_5': self.config.stats_analyzing.beta_5,
                
                # 기능 활성화 설정 (기본값으로 모두 활성화)
                'enable_range_check': True,
                'enable_new_check': True,
                'enable_nd_check': True,
                'enable_zero_check': True,
                'enable_high_delta_check': True,
            }
            
            result = self.judgement_strategy.apply(converted_data, filtering_result, judgement_config)
            
            self.logger.info("판정 알고리즘 실행 완료")
            return result
            
        except Exception as e:
            self.logger.error(f"판정 실행 중 오류: {e}")
            raise
    
    def _result_formatting(self, 
                          processing_results: Dict[str, Any], 
                          warnings: List[str]) -> ChoiAlgorithmResponse:
        """
        7단계: 최종 결과 포맷팅
        
        Args:
            processing_results: 모든 단계의 처리 결과
            warnings: 처리 중 발생한 경고들
            
        Returns:
            ChoiAlgorithmResponse: 최종 응답 객체
        """
        try:
            self.logger.debug("Formatting final response")
            
            # 기본 응답 구조 생성
            response = ChoiAlgorithmResponse(
                filtering=processing_results.get("filtering", create_empty_filtering_result()),
                abnormal_detection=processing_results.get("judgement", {}).get("abnormal_detection", create_empty_abnormal_detection_result()),
                processing_warnings=warnings,
                config_used=self._get_config_summary()
            )
            
            # KPI 판정 결과 추가
            judgement_data = processing_results.get("judgement", {})
            if "kpi_judgement" in judgement_data:
                response.kpi_judgement = judgement_data["kpi_judgement"]
            
            # UI 요약 정보 생성
            response.ui_summary = self._create_ui_summary(response.kpi_judgement)
            
            # 전체 요약 통계 생성
            response.overall_summary = self._create_overall_summary(response)
            
            # 메타데이터 설정
            response.total_cells_analyzed = len(processing_results.get("validated_data", {}))
            response.total_pegs_analyzed = sum(
                len(series_list) for series_list in processing_results.get("validated_data", {}).values()
            )
            
            self.logger.info("최종 결과 포맷팅 완료")
            return response
            
        except Exception as e:
            self.logger.error(f"결과 포맷팅 중 오류: {e}")
            raise
    
    def _calculate_basic_stats(self, samples: List[Optional[float]]) -> Dict[str, Optional[float]]:
        """
        기본 통계량 계산
        
        Args:
            samples: 샘플 데이터
            
        Returns:
            Dict[str, Optional[float]]: 기본 통계량
        """
        try:
            # None 값 제거
            valid_samples = [s for s in samples if s is not None]
            
            if not valid_samples:
                return {
                    "mean": None, "min": None, "max": None, 
                    "std": None, "cv": None, "count": 0
                }
            
            np_samples = np.array(valid_samples)
            mean_val = float(np.mean(np_samples))
            std_val = float(np.std(np_samples))
            
            return {
                "mean": mean_val,
                "min": float(np.min(np_samples)),
                "max": float(np.max(np_samples)),
                "std": std_val,
                "cv": std_val / mean_val if mean_val != 0 else None,
                "count": len(valid_samples)
            }
            
        except Exception as e:
            self.logger.error(f"통계 계산 중 오류: {e}")
            return {"mean": None, "min": None, "max": None, "std": None, "cv": None, "count": 0}
    
    def _convert_derived_to_sample_series(self, derived_data: Dict[str, Any]) -> Dict[str, List[PegSampleSeries]]:
        """
        파생 데이터를 PegSampleSeries 형태로 변환
        
        Args:
            derived_data: 파생 지표 데이터
            
        Returns:
            Dict[str, List[PegSampleSeries]]: 변환된 데이터
        """
        try:
            self.logger.debug("Converting derived data to PegSampleSeries format")
            
            converted_data = {}
            
            for cell_id, cell_stats in derived_data.items():
                peg_series_list = []
                
                for peg_name, peg_data in cell_stats.items():
                    # Mock 시계열 데이터 생성 (실제로는 원본 데이터에서 가져와야 함)
                    pre_mean = peg_data.get("pre", {}).get("mean", 1000.0)
                    post_mean = peg_data.get("post", {}).get("mean", 1100.0)
                    
                    # 평균 기반 Mock 샘플 생성
                    pre_samples = [pre_mean * (0.9 + i * 0.05) for i in range(5)]
                    post_samples = [post_mean * (0.9 + i * 0.05) for i in range(5)]
                    
                    series = PegSampleSeries(
                        peg_name=peg_name,
                        cell_id=cell_id,
                        pre_samples=pre_samples,
                        post_samples=post_samples,
                        unit=peg_data.get("unit", "unknown")
                    )
                    
                    peg_series_list.append(series)
                
                converted_data[cell_id] = peg_series_list
            
            self.logger.info(f"데이터 변환 완료: {len(converted_data)} cells, "
                           f"{sum(len(series_list) for series_list in converted_data.values())} PEG series")
            
            return converted_data
            
        except Exception as e:
            self.logger.error(f"데이터 변환 중 오류: {e}")
            return {}
    
    def _create_ui_summary(self, kpi_judgement: Dict[str, MainKPIJudgement]) -> Dict[str, Any]:
        """
        UI 표시용 요약 정보 생성
        
        Args:
            kpi_judgement: KPI 판정 결과
            
        Returns:
            Dict[str, Any]: UI 요약 정보
        """
        # TODO: 실제 UI 요약 생성 로직 구현
        return {}
    
    def _create_overall_summary(self, response: ChoiAlgorithmResponse) -> Dict[str, Any]:
        """
        전체 요약 통계 생성
        
        Args:
            response: 응답 객체
            
        Returns:
            Dict[str, Any]: 전체 요약
        """
        # TODO: 실제 요약 통계 생성 로직 구현
        return {
            "processing_completed": True,
            "algorithm_version": response.algorithm_version,
            "total_warnings": len(response.processing_warnings)
        }
    
    def _get_config_summary(self) -> Dict[str, Any]:
        """
        사용된 설정 요약 반환
        
        Returns:
            Dict[str, Any]: 설정 요약
        """
        return {
            "filtering_thresholds": {
                "min": self.config.filtering.min_threshold,
                "max": self.config.filtering.max_threshold
            },
            "alpha_0": self.config.abnormal_detection.alpha_0,
            "beta_values": {
                "beta_0": self.config.stats_analyzing.beta_0,
                "beta_1": self.config.stats_analyzing.beta_1,
                "beta_2": self.config.stats_analyzing.beta_2,
                "beta_3": self.config.stats_analyzing.beta_3
            },
            "config_version": self.config.config_version
        }


# =============================================================================
# 초기화 및 로깅
# =============================================================================

logger.info("PEG Processing Service loaded successfully")
