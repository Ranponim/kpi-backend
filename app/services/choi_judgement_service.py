"""
Choi 판정 서비스 구현

이 모듈은 TES.web_Choi.md 문서의 4장(이상 탐지)과 5장(통계 분석) 
판정 알고리즘을 Strategy 패턴으로 구현합니다.

주요 기능:
- 4장: Abnormal Stats Detecting Algorithm
  - Range, New, ND, Zero, High Delta 탐지
  - α0 규칙에 따른 결과 표시 로직
- 5장: Stats Analyzing Algorithm  
  - Can't Judge, High Variation, Improve/Degrade 판정
  - Similar/Delta 계층 판정 (β0-β5 임계값 적용)
  - Main/Sub KPI 결과 종합

PRD 참조: 섹션 2.2 (이상 탐지), 2.3 (통계 분석)
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict

from ..models.judgement import (
    PegSampleSeries,
    FilteringResult,
    AbnormalDetectionResult,
    MainKPIJudgement,
    PegPeriodStats,
    PegCompareMetrics,
    PegCompareDecision,
    JudgementType,
    CompareDetail,
    KPIPositivity
)
from ..services.strategies import BaseJudgementStrategy
from ..services.anomaly_detectors import (
    AnomalyDetectorFactory,
    AnomalyDetectionResult as DetectorResult,
    DimsDataProvider,
    MockDimsDataProvider
)
from ..services.kpi_analyzers import (
    KPIAnalyzerFactory,
    KPIAnalysisResult,
    BaseKPIAnalyzer
)

logger = logging.getLogger(__name__)


class ChoiJudgement(BaseJudgementStrategy):
    """
    Choi 판정 알고리즘 구현 (4장, 5장)
    
    TES.web_Choi.md 문서의 4장, 5장 판정 알고리즘을 정확히 구현합니다.
    """
    
    def __init__(self, 
                 detector_factory: Optional[AnomalyDetectorFactory] = None,
                 analyzer_factory: Optional[KPIAnalyzerFactory] = None,
                 dims_provider: Optional[DimsDataProvider] = None):
        """
        Choi 판정 전략 초기화
        
        Args:
            detector_factory: 이상 탐지기 팩토리 (의존성 주입)
            analyzer_factory: KPI 분석기 팩토리 (의존성 주입)
            dims_provider: DIMS 데이터 제공자 (의존성 주입)
        """
        super().__init__("ChoiJudgement", "1.0.0")
        
        # 의존성 주입 (Dependency Injection)
        self.dims_provider = dims_provider or MockDimsDataProvider()
        self.detector_factory = detector_factory or AnomalyDetectorFactory(self.dims_provider)
        self.analyzer_factory = analyzer_factory or KPIAnalyzerFactory()
        
        # 이상 탐지기들 초기화 (Lazy Loading)
        self._detectors = None
        
        # KPI 분석기들 초기화 (Lazy Loading)
        self._analyzers = None
        
        self.logger.info(f"Choi Judgement 알고리즘 초기화 완료 "
                        f"(DIMS provider: {type(self.dims_provider).__name__}, "
                        f"Factories: detector, analyzer)")
    
    def apply(self,
              filtered_data: Dict[str, List[PegSampleSeries]],
              filtering_result: FilteringResult,
              config: Dict[str, Any]) -> Dict[str, Any]:
        """
        판정 알고리즘 전체 실행
        
        Args:
            filtered_data: 필터링된 PEG 데이터
            filtering_result: 필터링 결과
            config: 판정 설정
            
        Returns:
            Dict[str, Any]: 판정 결과
        """
        try:
            self.logger.info(f"Choi 판정 알고리즘 시작: {len(filtered_data)} cells")
            
            # 입력 검증
            if not self.validate_input(filtered_data, filtering_result, config):
                raise ValueError("Invalid input data for judgement")
            
            # 4장: 이상 탐지 실행
            self.logger.debug("4장: 이상 통계 탐지 실행")
            abnormal_detection_config = config.get("abnormal_detection", {})
            abnormal_result = self.detect_abnormal_stats(filtered_data, abnormal_detection_config)
            
            # 5장: KPI 통계 분석 실행
            self.logger.debug("5장: KPI 통계 분석 실행")
            kpi_data = self._organize_data_by_kpi_topics(filtered_data, config.get("kpi_definitions", {}))
            stats_config = config.get("stats_analyzing", {})
            
            # KPI 분석기들 초기화 (Lazy Loading)
            if self._analyzers is None:
                self._analyzers = self.analyzer_factory.create_priority_ordered_analyzers()
                self.logger.debug(f"Initialized {len(self._analyzers)} KPI analyzers")
            
            kpi_judgement_result = self.analyze_kpi_stats(kpi_data, filtering_result, stats_config)
            
            # 결과 종합
            result = {
                "abnormal_detection": abnormal_result,
                "kpi_judgement": kpi_judgement_result,
                "processing_metadata": {
                    "algorithm_version": self.version,
                    "processed_cells": len(filtered_data),
                    "processed_pegs": sum(len(series_list) for series_list in filtered_data.values())
                }
            }
            
            self.logger.info(f"Choi 판정 알고리즘 완료: abnormal_types={len(abnormal_result.model_dump())}, "
                           f"kpi_topics={len(kpi_judgement_result)}")
            return result
            
        except Exception as e:
            self.logger.error(f"판정 실행 중 오류: {e}")
            raise RuntimeError(f"Judgement failed: {e}")
    
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
        try:
            self.logger.debug("이상 통계 탐지 시작")
            
            # 설정값 추출
            alpha_0 = config.get("alpha_0", 2)
            beta_3 = config.get("beta_3", 500.0)
            detection_types = config.get("detection_types", {})
            enable_range_check = config.get("enable_range_check", True)
            
            # 이상 탐지기들 초기화 (Lazy Loading)
            if self._detectors is None:
                self._detectors = self.detector_factory.create_all_detectors()
                self.logger.debug(f"Initialized {len(self._detectors)} anomaly detectors")
            
            # 각 이상 탐지 규칙 실행 (SOLID 원칙 준수)
            detection_results = {}
            
            for detector_type, detector in self._detectors.items():
                if detection_types.get(detector_type, True):
                    try:
                        # 각 탐지기는 독립적으로 실행 (Single Responsibility)
                        result = detector.detect(peg_data, config)
                        detection_results[result.anomaly_type] = result
                        
                        self.logger.debug(f"{detector_type} detection completed: "
                                        f"{len(result.affected_cells)} cells affected")
                    except Exception as e:
                        self.logger.error(f"Error in {detector_type} detection: {e}")
                        # 하나의 탐지기 실패가 전체를 중단시키지 않음 (견고한 오류 처리)
                        continue
            
            # 탐지 결과를 기존 형태로 변환
            converted_results = self._convert_detection_results(detection_results)
            
            # α0 규칙 적용하여 표시 여부 결정
            display_results = self._apply_alpha_zero_rule(converted_results, alpha_0)
            
            # 결과 객체 생성
            result = AbnormalDetectionResult(
                range_violations=converted_results.get("Range", {}),
                new_statistics=converted_results.get("New", {}),
                nd_anomalies=converted_results.get("ND", {}),
                zero_anomalies=converted_results.get("Zero", {}),
                high_delta_anomalies=converted_results.get("High Delta", {}),
                display_results=display_results
            )
            
            displayed_count = sum(1 for display in display_results.values() if display)
            self.logger.info(f"이상 탐지 완료: {displayed_count} anomaly types will be displayed")
            return result
            
        except Exception as e:
            self.logger.error(f"이상 탐지 중 오류: {e}")
            raise
    
    def analyze_kpi_stats(self,
                         kpi_data: Dict[str, Dict[str, List[PegSampleSeries]]],
                         filtering_result: FilteringResult,
                         config: Dict[str, Any]) -> Dict[str, MainKPIJudgement]:
        """
        KPI 통계 분석 (5장)
        
        Args:
            kpi_data: KPI 토픽별 데이터
            filtering_result: 필터링 결과
            config: KPI 분석 설정
            
        Returns:
            Dict[str, MainKPIJudgement]: KPI 토픽별 판정 결과
        """
        try:
            self.logger.debug("KPI 통계 분석 시작")
            
            # 설정값 추출
            beta_values = {
                "beta_0": config.get("beta_0", 1000.0),
                "beta_1": config.get("beta_1", 5.0),
                "beta_2": config.get("beta_2", 10.0),
                "beta_3": config.get("beta_3", 500.0),
                "beta_4": config.get("beta_4", 10.0),
                "beta_5": config.get("beta_5", 3.0)
            }
            
            rule_priorities = config.get("rule_priorities", {})
            
            kpi_judgement_results = {}
            
            for topic_name, topic_data in kpi_data.items():
                self.logger.debug(f"KPI 분석 시작: {topic_name}")
                
                try:
                    # Main KPI 분석 (SOLID 원칙 적용)
                    main_judgement = self._analyze_single_kpi(
                        topic_data.get("main", []), 
                        f"{topic_name}_main", 
                        beta_values, 
                        rule_priorities
                    )
                    
                    # Sub KPI 분석 (SOLID 원칙 적용)
                    sub_judgements = []
                    for i, sub_kpi_data in enumerate(topic_data.get("subs", [])):
                        sub_result = self._analyze_single_kpi(
                            [sub_kpi_data] if isinstance(sub_kpi_data, PegSampleSeries) else sub_kpi_data,
                            f"{topic_name}_sub_{i}",
                            beta_values,
                            rule_priorities
                        )
                        if sub_result:
                            sub_judgements.append(sub_result)
                    
                    # 최종 결과 종합 (5.4 규칙 적용)
                    final_judgement = self._combine_main_sub_results(main_judgement, sub_judgements, topic_name)
                    
                    if final_judgement:
                        kpi_judgement_results[topic_name] = final_judgement
                        
                except Exception as e:
                    self.logger.error(f"Error analyzing KPI topic {topic_name}: {e}")
                    # 하나의 토픽 실패가 전체를 중단시키지 않음 (견고한 오류 처리)
                    continue
            
            self.logger.info(f"KPI 분석 완료: {len(kpi_judgement_results)} topics analyzed")
            return kpi_judgement_results
            
        except Exception as e:
            self.logger.error(f"KPI 분석 중 오류: {e}")
            raise
    
    # =============================================================================
    # 이상 탐지 결과 처리 메서드들 (4장)
    # =============================================================================
    
    def _convert_detection_results(self, detection_results: Dict[str, DetectorResult]) -> Dict[str, Dict[str, List[str]]]:
        """
        탐지 결과를 기존 형태로 변환
        
        Args:
            detection_results: 탐지기별 결과
            
        Returns:
            Dict[str, Dict[str, List[str]]]: 변환된 결과
        """
        try:
            converted = {}
            
            for anomaly_type, result in detection_results.items():
                # 영향받은 셀들을 리스트로 변환
                affected_cells_list = list(result.affected_cells)
                converted[anomaly_type] = {anomaly_type: affected_cells_list}
                
                self.logger.debug(f"Converted {anomaly_type}: {len(affected_cells_list)} affected cells")
            
            return converted
            
        except Exception as e:
            self.logger.error(f"Detection result conversion error: {e}")
            return {}
    
    def _apply_alpha_zero_rule(self, anomaly_results: Dict[str, Dict[str, List[str]]], alpha_0: int) -> Dict[str, bool]:
        """
        α0 규칙 적용 (4장)
        
        셀 수 ≥ α0 조건에 따른 표시 여부 결정
        
        Args:
            anomaly_results: 이상 탐지 결과
            alpha_0: 최소 셀 수 임계값
            
        Returns:
            Dict[str, bool]: 이상 유형별 표시 여부
        """
        try:
            display_results = {}
            total_cells = self._get_total_cell_count(anomaly_results)
            
            self.logger.debug(f"α0 규칙 적용 시작: α0={alpha_0}, total_cells={total_cells}")
            
            for anomaly_type, anomaly_data in anomaly_results.items():
                affected_cells = anomaly_data.get(anomaly_type, [])
                affected_count = len(affected_cells)
                
                # α0 규칙 적용
                if total_cells >= alpha_0:
                    # 일반적인 경우: 영향받은 셀 수 ≥ α0
                    should_display = affected_count >= alpha_0
                else:
                    # 전체 셀 수 < α0인 경우: 모든 셀이 동일 판정이어야 표시
                    should_display = affected_count == total_cells
                
                display_results[anomaly_type] = should_display
                
                self.logger.debug(f"α0 규칙 - {anomaly_type}: {affected_count}/{total_cells} cells, "
                                f"display={should_display}")
            
            displayed_count = sum(1 for display in display_results.values() if display)
            self.logger.info(f"α0 규칙 적용 완료: {displayed_count}/{len(display_results)} anomaly types will be displayed")
            
            return display_results
            
        except Exception as e:
            self.logger.error(f"α0 규칙 적용 중 오류: {e}")
            return {anomaly_type: False for anomaly_type in anomaly_results.keys()}
    
    def _get_total_cell_count(self, anomaly_results: Dict[str, Dict[str, List[str]]]) -> int:
        """
        전체 셀 수 계산
        
        Args:
            anomaly_results: 이상 탐지 결과
            
        Returns:
            int: 전체 셀 수
        """
        try:
            all_cells = set()
            for anomaly_data in anomaly_results.values():
                for cell_list in anomaly_data.values():
                    all_cells.update(cell_list)
            
            return len(all_cells)
            
        except Exception as e:
            self.logger.error(f"Total cell count calculation error: {e}")
            return 0
    
    # =============================================================================
    # KPI 분석 구현 메서드들 (5장) - 구현 예정
    # =============================================================================
    
    def _organize_data_by_kpi_topics(self, 
                                   filtered_data: Dict[str, List[PegSampleSeries]], 
                                   kpi_definitions: Dict[str, Any]) -> Dict[str, Dict[str, List[PegSampleSeries]]]:
        """KPI 토픽별로 데이터 재구성"""
        # TODO: KPI 정의에 따라 Main/Sub KPI 데이터 분류 구현
        self.logger.debug("KPI 토픽별 데이터 재구성 - 구현 예정")
        return {}
    
    def _analyze_main_kpi(self, 
                         main_kpi_data: List[PegSampleSeries], 
                         beta_values: Dict[str, float], 
                         rule_priorities: Dict[str, int]) -> Dict[str, Any]:
        """Main KPI 분석 (구현 예정)"""
        # TODO: Main KPI 판정 로직 구현
        self.logger.debug("Main KPI 분석 - 구현 예정")
        return {}
    
    def _analyze_sub_kpis(self, 
                         sub_kpi_data: List[PegSampleSeries], 
                         beta_values: Dict[str, float], 
                         rule_priorities: Dict[str, int]) -> List[Dict[str, Any]]:
        """Sub KPI 분석 (구현 예정)"""
        # TODO: Sub KPI 판정 로직 구현
        self.logger.debug("Sub KPI 분석 - 구현 예정")
        return []
    
    def _analyze_single_kpi(self,
                           kpi_series_list: List[PegSampleSeries],
                           kpi_name: str,
                           beta_values: Dict[str, float],
                           rule_priorities: Dict[str, int]) -> Optional[KPIAnalysisResult]:
        """
        단일 KPI 분석 (SOLID 원칙 적용)
        
        Args:
            kpi_series_list: KPI 시계열 데이터
            kpi_name: KPI 이름
            beta_values: β 임계값들
            rule_priorities: 규칙 우선순위
            
        Returns:
            Optional[KPIAnalysisResult]: 분석 결과
        """
        try:
            if not kpi_series_list:
                self.logger.warning(f"No data for KPI: {kpi_name}")
                return None
            
            # 첫 번째 시리즈 사용 (Main KPI 또는 대표 Sub KPI)
            series = kpi_series_list[0]
            
            # 기본 통계 계산
            pre_stats = self._calculate_period_stats(series.pre_samples)
            post_stats = self._calculate_period_stats(series.post_samples)
            compare_metrics = self._calculate_compare_metrics(pre_stats, post_stats)
            
            # 우선순위 순서로 분석기 적용 (Chain of Responsibility Pattern)
            analysis_config = {**beta_values, **rule_priorities}
            
            for analyzer in self._analyzers:
                try:
                    result = analyzer.analyze(pre_stats, post_stats, compare_metrics, analysis_config)
                    if result:
                        self.logger.debug(f"{kpi_name}: {analyzer.analyzer_name} rule applied")
                        return result
                except Exception as e:
                    self.logger.error(f"Error in {analyzer.analyzer_name} for {kpi_name}: {e}")
                    # 하나의 분석기 실패가 전체를 중단시키지 않음
                    continue
            
            # 모든 분석기가 적용되지 않은 경우 기본값 반환
            self.logger.warning(f"No analysis rule applied for {kpi_name}, using default")
            return None
            
        except Exception as e:
            self.logger.error(f"Single KPI analysis error for {kpi_name}: {e}")
            return None
    
    def _combine_main_sub_results(self, 
                                 main_result: Optional[KPIAnalysisResult], 
                                 sub_results: List[KPIAnalysisResult],
                                 topic_name: str) -> Optional[MainKPIJudgement]:
        """Main/Sub KPI 결과 종합 (구현 예정)"""
        # TODO: 5.4 최종 결과 요약 로직 구현
        self.logger.debug("Main/Sub KPI 결과 종합 - 구현 예정")
        
        # 임시 결과 반환
        from ..models.judgement import PegPeriodStats, PegCompareMetrics
        
        return MainKPIJudgement(
            main_kpi_name="TempMainKPI",
            main_result=JudgementType.OK,
            main_decision=PegCompareDecision(
                detail=CompareDetail.SIMILAR,
                reason="임시 판정 결과",
                thresholds_used={}
            ),
            sub_results=[],
            final_result=JudgementType.OK,
            summary_text="임시 요약",
            pre_stats=PegPeriodStats(),
            post_stats=PegPeriodStats(),
            compare_metrics=PegCompareMetrics()
        )
    
    # =============================================================================
    # 통계 계산 유틸리티 메서드들
    # =============================================================================
    
    def _calculate_period_stats(self, samples: List[Optional[float]]) -> PegPeriodStats:
        """
        기간별 통계 계산
        
        Args:
            samples: 샘플 데이터
            
        Returns:
            PegPeriodStats: 계산된 통계
        """
        try:
            # None 값 제거
            valid_samples = [s for s in samples if s is not None]
            
            if not valid_samples:
                return PegPeriodStats(sample_count=0)
            
            np_samples = np.array(valid_samples)
            mean_val = float(np.mean(np_samples))
            std_val = float(np.std(np_samples))
            
            # ND 및 Zero 비율 계산
            nd_count = sum(1 for s in samples if s is None)
            zero_count = sum(1 for s in valid_samples if s == 0.0)
            total_count = len(samples)
            
            return PegPeriodStats(
                mean=mean_val,
                min=float(np.min(np_samples)),
                max=float(np.max(np_samples)),
                std=std_val,
                cv=std_val / mean_val if mean_val != 0 else None,
                nd_ratio=nd_count / total_count if total_count > 0 else 0,
                zero_ratio=zero_count / len(valid_samples) if valid_samples else 0,
                sample_count=len(valid_samples)
            )
            
        except Exception as e:
            self.logger.error(f"통계 계산 중 오류: {e}")
            return PegPeriodStats(sample_count=0)
    
    def _calculate_compare_metrics(self, 
                                  pre_stats: PegPeriodStats, 
                                  post_stats: PegPeriodStats) -> PegCompareMetrics:
        """
        비교 지표 계산
        
        Args:
            pre_stats: Pre 기간 통계
            post_stats: Post 기간 통계
            
        Returns:
            PegCompareMetrics: 비교 지표
        """
        try:
            # 변화율 계산
            delta_pct = None
            if pre_stats.mean is not None and pre_stats.mean != 0:
                delta_pct = ((post_stats.mean - pre_stats.mean) / pre_stats.mean) * 100
            
            # 플래그 설정
            has_nd = pre_stats.nd_ratio > 0 or post_stats.nd_ratio > 0
            has_zero = pre_stats.zero_ratio > 0 or post_stats.zero_ratio > 0
            
            # 트래픽 볼륨 분류 (β0 기준)
            beta_0 = 1000.0  # 설정에서 가져와야 함
            traffic_class = "low"
            if (pre_stats.mean and pre_stats.mean >= beta_0) and (post_stats.mean and post_stats.mean >= beta_0):
                traffic_class = "high"
            
            return PegCompareMetrics(
                delta_pct=delta_pct,
                has_nd=has_nd,
                has_zero=has_zero,
                has_new=False,  # 구현 예정
                out_of_range=False,  # 구현 예정
                traffic_volume_class=traffic_class
            )
            
        except Exception as e:
            self.logger.error(f"비교 지표 계산 중 오류: {e}")
            return PegCompareMetrics()
    
    # =============================================================================
    # 최종 KPI 결과 요약 로직 (PRD 2.3.5)
    # =============================================================================
    
    def summarize_final_kpi_results(self, 
                                  main_kpi_judgements: Dict[str, 'SimpleKPIJudgement'],
                                  sub_kpi_judgements: Dict[str, 'SimpleKPIJudgement']) -> Dict[str, 'SimpleKPIJudgement']:
        """
        최종 KPI 결과 요약 로직 (PRD 2.3.5)
        
        요약 규칙:
        1. Main NOK -> NOK
        2. Main OK + any Sub NOK -> POK (Partially OK)
        3. Main OK + all Sub OK -> OK
        4. Main Can't judge -> Can't judge
        
        Args:
            main_kpi_judgements: Main KPI 판정 결과
            sub_kpi_judgements: Sub KPI 판정 결과
            
        Returns:
            Dict[str, SimpleKPIJudgement]: 최종 요약된 KPI 판정 결과
        """
        try:
            self.logger.info("최종 KPI 결과 요약 시작")
            
            final_results = {}
            
            for main_kpi_name, main_judgement in main_kpi_judgements.items():
                
                # 해당 Main KPI의 Sub KPI들 찾기
                related_sub_kpis = {
                    sub_name: sub_judgement 
                    for sub_name, sub_judgement in sub_kpi_judgements.items()
                    if self._is_related_sub_kpi(main_kpi_name, sub_name)
                }
                
                # 최종 판정 적용
                final_judgement = self._apply_final_summary_rules(
                    main_kpi_name, main_judgement, related_sub_kpis
                )
                
                final_results[main_kpi_name] = final_judgement
                
                self.logger.debug(f"KPI {main_kpi_name} 최종 판정: {final_judgement.judgement_type}")
            
            self.logger.info(f"최종 KPI 결과 요약 완료: {len(final_results)}개 KPI")
            return final_results
            
        except Exception as e:
            self.logger.error(f"최종 KPI 결과 요약 중 오류: {e}")
            # 오류 시 원본 Main KPI 결과 반환
            return main_kpi_judgements
    
    def _is_related_sub_kpi(self, main_kpi_name: str, sub_kpi_name: str) -> bool:
        """
        Sub KPI가 Main KPI와 관련있는지 확인
        
        Args:
            main_kpi_name: Main KPI 이름
            sub_kpi_name: Sub KPI 이름
            
        Returns:
            bool: 관련 여부
        """
        try:
            # 일반적인 관련성 판단 로직
            # 예: AirMacDLThruAvg와 AirMacDLThruMax가 관련
            main_base = main_kpi_name.replace("Avg", "").replace("Max", "").replace("Min", "")
            sub_base = sub_kpi_name.replace("Avg", "").replace("Max", "").replace("Min", "")
            
            # 같은 기본 이름을 가지면 관련 KPI로 판단
            is_related = main_base == sub_base and main_kpi_name != sub_kpi_name
            
            if is_related:
                self.logger.debug(f"Sub KPI {sub_kpi_name}는 Main KPI {main_kpi_name}와 관련")
            
            return is_related
            
        except Exception as e:
            self.logger.error(f"Sub KPI 관련성 판단 오류: {e}")
            return False
    
    def _apply_final_summary_rules(self, 
                                 main_kpi_name: str,
                                 main_judgement: 'SimpleKPIJudgement',
                                 related_sub_kpis: Dict[str, 'SimpleKPIJudgement']) -> 'SimpleKPIJudgement':
        """
        최종 요약 규칙 적용
        
        Args:
            main_kpi_name: Main KPI 이름
            main_judgement: Main KPI 판정 결과
            related_sub_kpis: 관련 Sub KPI 판정 결과들
            
        Returns:
            SimpleKPIJudgement: 최종 요약 판정 결과
        """
        try:
            # 규칙 1: Main Can't judge -> Can't judge
            if main_judgement.judgement_type == JudgementType.CANT_JUDGE:
                return self._create_summary_judgement(
                    main_judgement,
                    "Main KPI 판정 불가로 인한 전체 판정 불가",
                    related_sub_kpis,
                    "rule_1_main_cant_judge"
                )
            
            # 규칙 2: Main NOK -> NOK
            if main_judgement.judgement_type == JudgementType.NOK:
                return self._create_summary_judgement(
                    main_judgement,
                    f"Main KPI NOK ({main_judgement.compare_detail}) → 전체 NOK",
                    related_sub_kpis,
                    "rule_2_main_nok"
                )
            
            # 규칙 3 & 4: Main OK인 경우 Sub KPI 검토
            if main_judgement.judgement_type == JudgementType.OK:
                return self._evaluate_main_ok_with_subs(
                    main_kpi_name, main_judgement, related_sub_kpis
                )
            
            # 예상하지 못한 경우 (방어적 프로그래밍)
            self.logger.warning(f"예상하지 못한 Main KPI 판정 타입: {main_judgement.judgement_type}")
            return main_judgement
            
        except Exception as e:
            self.logger.error(f"최종 요약 규칙 적용 오류: {e}")
            return main_judgement
    
    def _evaluate_main_ok_with_subs(self, 
                                  main_kpi_name: str,
                                  main_judgement: 'SimpleKPIJudgement',
                                  related_sub_kpis: Dict[str, 'SimpleKPIJudgement']) -> 'SimpleKPIJudgement':
        """
        Main OK인 경우 Sub KPI들을 검토하여 최종 판정
        
        Args:
            main_kpi_name: Main KPI 이름
            main_judgement: Main KPI 판정 (OK)
            related_sub_kpis: 관련 Sub KPI 판정 결과들
            
        Returns:
            SimpleKPIJudgement: 최종 판정 결과
        """
        try:
            if not related_sub_kpis:
                # Sub KPI가 없으면 Main OK 그대로
                return self._create_summary_judgement(
                    main_judgement,
                    "Main KPI OK, Sub KPI 없음 → OK",
                    related_sub_kpis,
                    "rule_3_main_ok_no_subs"
                )
            
            # Sub KPI들의 판정 상태 분석
            sub_analysis = self._analyze_sub_kpi_results(related_sub_kpis)
            
            # 규칙 4: Main OK + any Sub NOK -> POK
            if sub_analysis["has_nok"]:
                from app.models.judgement import SimpleKPIJudgement
                pok_judgement = SimpleKPIJudgement(
                    judgement_type=JudgementType.POK,  # Partially OK
                    compare_detail=CompareDetail.PARTIALLY_OK,
                    reasoning=f"Main KPI OK이나 Sub KPI 중 NOK 존재 → POK",
                    confidence=min(main_judgement.confidence, sub_analysis["min_confidence"]),
                    metrics=main_judgement.metrics,
                    thresholds_used=main_judgement.thresholds_used
                )
                
                return self._create_summary_judgement(
                    pok_judgement,
                    f"Main OK + Sub NOK({sub_analysis['nok_count']}개) → POK",
                    related_sub_kpis,
                    "rule_4_main_ok_sub_nok"
                )
            
            # 규칙 3: Main OK + all Sub OK -> OK
            return self._create_summary_judgement(
                main_judgement,
                f"Main KPI OK + 모든 Sub KPI OK({sub_analysis['ok_count']}개) → OK",
                related_sub_kpis,
                "rule_3_main_ok_all_sub_ok"
            )
            
        except Exception as e:
            self.logger.error(f"Main OK Sub 평가 오류: {e}")
            return main_judgement
    
    def _analyze_sub_kpi_results(self, related_sub_kpis: Dict[str, 'SimpleKPIJudgement']) -> Dict[str, Any]:
        """Sub KPI 결과들 분석"""
        try:
            analysis = {
                "total_count": len(related_sub_kpis),
                "ok_count": 0,
                "nok_count": 0,
                "pok_count": 0,
                "cant_judge_count": 0,
                "has_nok": False,
                "has_cant_judge": False,
                "min_confidence": 1.0,
                "nok_details": []
            }
            
            for sub_name, sub_judgement in related_sub_kpis.items():
                analysis["min_confidence"] = min(analysis["min_confidence"], sub_judgement.confidence)
                
                if sub_judgement.judgement_type == JudgementType.OK:
                    analysis["ok_count"] += 1
                elif sub_judgement.judgement_type == JudgementType.NOK:
                    analysis["nok_count"] += 1
                    analysis["has_nok"] = True
                    analysis["nok_details"].append(f"{sub_name}({sub_judgement.compare_detail})")
                elif sub_judgement.judgement_type == JudgementType.POK:
                    analysis["pok_count"] += 1
                    analysis["has_nok"] = True  # POK도 NOK로 취급
                    analysis["nok_details"].append(f"{sub_name}(POK)")
                elif sub_judgement.judgement_type == JudgementType.CANT_JUDGE:
                    analysis["cant_judge_count"] += 1
                    analysis["has_cant_judge"] = True
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Sub KPI 결과 분석 오류: {e}")
            return {"total_count": 0, "has_nok": False, "has_cant_judge": False, "min_confidence": 0.5}
    
    def _create_summary_judgement(self, 
                                base_judgement: 'SimpleKPIJudgement',
                                summary_reasoning: str,
                                related_sub_kpis: Dict[str, 'SimpleKPIJudgement'],
                                rule_applied: str) -> 'SimpleKPIJudgement':
        """요약 판정 결과 생성"""
        try:
            # 기존 메트릭스에 요약 정보 추가
            enhanced_metrics = {
                **base_judgement.metrics,
                "summary_rule_applied": rule_applied,
                "sub_kpi_count": len(related_sub_kpis),
                "sub_kpi_names": list(related_sub_kpis.keys()),
                "original_reasoning": base_judgement.reasoning
            }
            
            from app.models.judgement import SimpleKPIJudgement
            return SimpleKPIJudgement(
                judgement_type=base_judgement.judgement_type,
                compare_detail=base_judgement.compare_detail,
                reasoning=summary_reasoning,
                confidence=base_judgement.confidence,
                metrics=enhanced_metrics,
                thresholds_used=base_judgement.thresholds_used
            )
            
        except Exception as e:
            self.logger.error(f"요약 판정 생성 오류: {e}")
            return base_judgement


# =============================================================================
# 초기화 및 로깅
# =============================================================================

logger.info("Choi Judgement Service loaded successfully")
