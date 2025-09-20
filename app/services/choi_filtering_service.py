"""
Choi 필터링 서비스 구현

이 모듈은 TES.web_Choi.md 문서의 6장 필터링 알고리즘을
Strategy 패턴으로 구현합니다.

주요 알고리즘 단계 (PRD 2.1.2):
1. 전처리: 이웃 시각 0 처리, DL/UL 합 0 제외
2. 중앙값 계산: 각 통계 PEG별 샘플 중앙값 산출
3. 정규화: 각 샘플값을 중앙값으로 나누어 정규화
4. 임계값 적용: Min_Threshold ≤ 정규화값 ≤ Max_Threshold
5. 교집합 선택: 모든 PEG에서 공통으로 남는 시각 선택
6. 50% 규칙: 필터링 결과가 50% 이하이면 경고 후 전체 사용

PRD 참조: 섹션 2.1 (필터링 알고리즘), 6장 원본 문서
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict

from ..models.judgement import (
    PegSampleSeries,
    FilteringResult
)
from ..services.strategies import BaseFilteringStrategy
from ..utils.logging_decorators import log_strategy_execution, log_service_method
from ..exceptions import FilteringError, InsufficientDataError, ConfigurationError

logger = logging.getLogger(__name__)


class ChoiFiltering(BaseFilteringStrategy):
    """
    Choi 필터링 알고리즘 구현 (6장)
    
    TES.web_Choi.md 문서의 6장 필터링 알고리즘을 정확히 구현합니다.
    """
    
    def __init__(self):
        """Choi 필터링 전략 초기화"""
        super().__init__("ChoiFiltering", "1.0.0")
        self.logger.info("Choi Filtering 알고리즘 초기화 완료")
    
    @log_strategy_execution("filtering")
    def apply(self, 
              peg_data: Dict[str, List[PegSampleSeries]], 
              config: Dict[str, Any]) -> FilteringResult:
        """
        필터링 알고리즘 전체 실행
        
        Args:
            peg_data: 셀별 PEG 시계열 데이터
            config: 필터링 설정
            
        Returns:
            FilteringResult: 필터링 결과
        """
        try:
            self.logger.info(f"Choi 필터링 알고리즘 시작: {len(peg_data)} cells")
            
            # 입력 검증
            if not peg_data:
                raise InsufficientDataError(
                    "필터링을 위한 PEG 데이터가 없습니다",
                    required_data=["peg_data"],
                    provided_data=peg_data
                )
            
            if not self.validate_input(peg_data, config):
                raise ConfigurationError(
                    "필터링 설정이 올바르지 않습니다",
                    config_section="filtering",
                    invalid_keys=[k for k in ['min_threshold', 'max_threshold', 'filter_ratio'] if k not in config]
                )
            
            # 설정값 추출
            min_threshold = config.get('min_threshold', 0.87)
            max_threshold = config.get('max_threshold', 1.13)
            filter_ratio_threshold = config.get('filter_ratio', 0.50)
            warning_message_template = config.get('warning_message', 
                "TES can't filter the valid samples because test results are unstable")
            
            # 1단계: 데이터 전처리
            self.logger.debug("1단계: 데이터 전처리 실행")
            preprocessed_data, preprocessing_stats = self._preprocess_data(peg_data)
            
            # 2단계: 중앙값 계산
            self.logger.debug("2단계: 중앙값 계산 실행")
            median_values = self._calculate_medians(preprocessed_data)
            
            # 3단계: 정규화
            self.logger.debug("3단계: 시계열 정규화 실행")
            normalized_data = self._normalize_by_median(preprocessed_data, median_values)
            
            # 4단계: 임계값 적용
            self.logger.debug("4단계: 임계값 필터링 실행")
            threshold_filtered_data = self._apply_threshold_filtering(normalized_data, min_threshold, max_threshold)
            
            # 5단계: 교집합 계산
            self.logger.debug("5단계: 시간 슬롯 교집합 계산")
            intersection_result = self._calculate_time_slot_intersections(threshold_filtered_data)
            
            # 6단계: 50% 규칙 적용
            self.logger.debug("6단계: 50% 규칙 적용")
            final_result = self._apply_fifty_percent_rule(
                intersection_result, 
                preprocessed_data, 
                filter_ratio_threshold, 
                warning_message_template
            )
            
            # 최종 결과 생성
            result = FilteringResult(
                valid_time_slots=final_result["valid_time_slots"],
                filter_ratio=final_result["filter_ratio"],
                warning_message=final_result["warning_message"],
                preprocessing_stats=preprocessing_stats,
                median_values=median_values
            )
            
            self.logger.info(f"Choi 필터링 전체 완료: filter_ratio={result.filter_ratio:.3f}, "
                           f"warning={result.warning_message is not None}")
            return result
            
        except Exception as e:
            self.logger.error(f"필터링 실행 중 오류: {e}")
            raise RuntimeError(f"Filtering failed: {e}")
    
    def _apply_threshold_filtering(self, 
                                  normalized_data: Dict[str, List[PegSampleSeries]], 
                                  min_threshold: float, 
                                  max_threshold: float) -> Dict[str, Dict[str, List[int]]]:
        """
        4단계: 임계값 필터링 (6장 4단계)
        
        Min_Threshold ≤ 정규화값 ≤ Max_Threshold 조건 적용
        
        Args:
            normalized_data: 정규화된 데이터
            min_threshold: 최소 임계값 (0.87)
            max_threshold: 최대 임계값 (1.13)
            
        Returns:
            Dict[str, Dict[str, List[int]]]: 셀별 PEG별 유효 시간 슬롯 인덱스
        """
        try:
            self.logger.debug(f"임계값 필터링 시작: {min_threshold} ≤ normalized ≤ {max_threshold}")
            
            threshold_filtered = {}
            total_slots_before = 0
            total_slots_after = 0
            
            for cell_id, peg_series_list in normalized_data.items():
                cell_filtered = {}
                
                for series in peg_series_list:
                    # Pre 기간 필터링
                    pre_valid_indices = []
                    for i, sample in enumerate(series.pre_samples):
                        if sample is not None and min_threshold <= sample <= max_threshold:
                            pre_valid_indices.append(i)
                    
                    # Post 기간 필터링
                    post_valid_indices = []
                    for i, sample in enumerate(series.post_samples):
                        if sample is not None and min_threshold <= sample <= max_threshold:
                            post_valid_indices.append(i)
                    
                    # 전체 유효 인덱스 (pre + post, post는 pre 길이만큼 오프셋)
                    pre_len = len(series.pre_samples)
                    all_valid_indices = pre_valid_indices + [i + pre_len for i in post_valid_indices]
                    
                    cell_filtered[series.peg_name] = all_valid_indices
                    
                    total_slots_before += len(series.pre_samples) + len(series.post_samples)
                    total_slots_after += len(all_valid_indices)
                    
                    self.logger.debug(f"임계값 필터링 - {cell_id}.{series.peg_name}: "
                                    f"{len(all_valid_indices)}/{len(series.pre_samples) + len(series.post_samples)} slots valid")
                
                threshold_filtered[cell_id] = cell_filtered
            
            filter_efficiency = total_slots_after / total_slots_before if total_slots_before > 0 else 0
            self.logger.info(f"임계값 필터링 완료: {total_slots_after}/{total_slots_before} slots retained "
                           f"(효율성: {filter_efficiency:.2%})")
            
            return threshold_filtered
            
        except Exception as e:
            self.logger.error(f"임계값 필터링 중 오류: {e}")
            raise
    
    def _calculate_time_slot_intersections(self, 
                                         threshold_filtered_data: Dict[str, Dict[str, List[int]]]) -> Dict[str, List[int]]:
        """
        5단계: 시간 슬롯 교집합 계산 (6장 5단계)
        
        각 셀에서 모든 PEG에 공통으로 유효한 시간 슬롯만 선택
        
        Args:
            threshold_filtered_data: 임계값 필터링된 데이터
            
        Returns:
            Dict[str, List[int]]: 셀별 교집합 시간 슬롯 인덱스
        """
        try:
            self.logger.debug("시간 슬롯 교집합 계산 시작")
            
            intersection_result = {}
            
            for cell_id, peg_filtered_data in threshold_filtered_data.items():
                if not peg_filtered_data:
                    self.logger.warning(f"No PEG data for cell {cell_id}")
                    intersection_result[cell_id] = []
                    continue
                
                # 첫 번째 PEG의 유효 슬롯으로 시작
                peg_names = list(peg_filtered_data.keys())
                if not peg_names:
                    intersection_result[cell_id] = []
                    continue
                
                # 모든 PEG의 유효 슬롯 교집합 계산
                common_slots = set(peg_filtered_data[peg_names[0]])
                
                for peg_name in peg_names[1:]:
                    peg_slots = set(peg_filtered_data[peg_name])
                    common_slots = common_slots.intersection(peg_slots)
                
                # 교집합 결과를 정렬된 리스트로 변환
                final_slots = sorted(list(common_slots))
                intersection_result[cell_id] = final_slots
                
                # 교집합 통계 로깅
                original_counts = [len(slots) for slots in peg_filtered_data.values()]
                max_count = max(original_counts) if original_counts else 0
                intersection_efficiency = len(final_slots) / max_count if max_count > 0 else 0
                
                self.logger.debug(f"교집합 계산 - {cell_id}: {len(final_slots)} common slots "
                                f"(효율성: {intersection_efficiency:.2%}, PEG별 슬롯: {original_counts})")
            
            total_intersection_slots = sum(len(slots) for slots in intersection_result.values())
            self.logger.info(f"교집합 계산 완료: {len(intersection_result)} cells, "
                           f"{total_intersection_slots} total intersection slots")
            
            return intersection_result
            
        except Exception as e:
            self.logger.error(f"교집합 계산 중 오류: {e}")
            raise
    
    def _apply_fifty_percent_rule(self, 
                                 intersection_result: Dict[str, List[int]], 
                                 original_data: Dict[str, List[PegSampleSeries]], 
                                 filter_ratio_threshold: float, 
                                 warning_message_template: str) -> Dict[str, Any]:
        """
        6단계: 50% 규칙 적용 (6장 5-6단계)
        
        필터링 결과가 50% 이하이면 경고 메시지와 함께 전체 시간 구간 사용
        
        Args:
            intersection_result: 교집합 결과
            original_data: 원본 데이터 (전체 시간 구간 확인용)
            filter_ratio_threshold: 필터링 비율 임계값 (0.50)
            warning_message_template: 경고 메시지 템플릿
            
        Returns:
            Dict[str, Any]: 최종 필터링 결과
        """
        try:
            self.logger.debug(f"50% 규칙 적용 시작: 임계값={filter_ratio_threshold}")
            
            # 전체 입력 샘플 수 계산
            total_input_samples = 0
            total_filtered_samples = 0
            original_time_slots = {}
            
            for cell_id, peg_series_list in original_data.items():
                if peg_series_list:
                    # 첫 번째 시리즈의 전체 길이를 기준으로 원본 시간 슬롯 생성
                    first_series = peg_series_list[0]
                    total_length = len(first_series.pre_samples) + len(first_series.post_samples)
                    original_time_slots[cell_id] = list(range(total_length))
                    total_input_samples += total_length
                else:
                    original_time_slots[cell_id] = []
                
                # 필터링된 샘플 수 합계
                filtered_slots = intersection_result.get(cell_id, [])
                total_filtered_samples += len(filtered_slots)
            
            # 전체 필터링 비율 계산
            overall_filter_ratio = total_filtered_samples / total_input_samples if total_input_samples > 0 else 0
            
            self.logger.info(f"필터링 비율 계산: {total_filtered_samples}/{total_input_samples} = {overall_filter_ratio:.2%}")
            
            # 50% 규칙 적용
            if overall_filter_ratio <= filter_ratio_threshold:
                # 필터링 실패: 경고 메시지와 함께 전체 시간 구간 사용
                self.logger.warning(f"필터링 비율 {overall_filter_ratio:.2%} ≤ {filter_ratio_threshold:.0%}, "
                                  f"전체 시간 구간 사용")
                
                return {
                    "valid_time_slots": original_time_slots,
                    "filter_ratio": overall_filter_ratio,
                    "warning_message": warning_message_template
                }
            else:
                # 필터링 성공: 교집합 결과 사용
                self.logger.info(f"필터링 성공: {overall_filter_ratio:.2%} > {filter_ratio_threshold:.0%}")
                
                return {
                    "valid_time_slots": intersection_result,
                    "filter_ratio": overall_filter_ratio,
                    "warning_message": None
                }
                
        except Exception as e:
            self.logger.error(f"50% 규칙 적용 중 오류: {e}")
            raise
    
    def _preprocess_data(self, 
                        peg_data: Dict[str, List[PegSampleSeries]]) -> Tuple[Dict[str, List[PegSampleSeries]], Dict[str, Any]]:
        """
        1단계: 데이터 전처리 (6장 1단계)
        
        이웃 시각(n-1, n+1) 값이 0인 경우 n 시각 제외
        DL + UL 처리량 합계가 0인 경우 제외
        
        Args:
            peg_data: 원본 PEG 데이터
            
        Returns:
            Tuple[Dict, Dict]: (전처리된 데이터, 전처리 통계)
        """
        try:
            self.logger.debug("데이터 전처리 시작")
            
            preprocessed_data = {}
            total_samples_before = 0
            total_samples_after = 0
            removed_by_neighbor_zero = 0
            removed_by_dl_ul_zero = 0
            
            for cell_id, peg_series_list in peg_data.items():
                preprocessed_series = []
                
                for series in peg_series_list:
                    total_samples_before += len(series.pre_samples) + len(series.post_samples)
                    
                    # 이웃 시각 0 처리 (pre/post 각각)
                    cleaned_pre, removed_pre = self._remove_neighbor_zero_samples(series.pre_samples)
                    cleaned_post, removed_post = self._remove_neighbor_zero_samples(series.post_samples)
                    
                    removed_by_neighbor_zero += removed_pre + removed_post
                    
                    # DL/UL 합계 0 처리 (해당되는 경우만)
                    if self._is_throughput_peg(series.peg_name):
                        cleaned_pre, removed_dl_ul_pre = self._remove_zero_sum_samples(cleaned_pre, series.peg_name)
                        cleaned_post, removed_dl_ul_post = self._remove_zero_sum_samples(cleaned_post, series.peg_name)
                        removed_by_dl_ul_zero += removed_dl_ul_pre + removed_dl_ul_post
                    
                    # 전처리된 시리즈 생성
                    cleaned_series = PegSampleSeries(
                        peg_name=series.peg_name,
                        cell_id=series.cell_id,
                        pre_samples=cleaned_pre,
                        post_samples=cleaned_post,
                        unit=series.unit
                    )
                    
                    preprocessed_series.append(cleaned_series)
                    total_samples_after += len(cleaned_pre) + len(cleaned_post)
                
                preprocessed_data[cell_id] = preprocessed_series
            
            # 전처리 통계 생성
            preprocessing_stats = {
                "total_samples_before": total_samples_before,
                "total_samples_after": total_samples_after,
                "removed_by_neighbor_zero": removed_by_neighbor_zero,
                "removed_by_dl_ul_zero": removed_by_dl_ul_zero,
                "removal_ratio": (total_samples_before - total_samples_after) / total_samples_before if total_samples_before > 0 else 0
            }
            
            self.logger.info(f"전처리 완료: {total_samples_before} → {total_samples_after} samples "
                           f"(제거: neighbor_zero={removed_by_neighbor_zero}, dl_ul_zero={removed_by_dl_ul_zero})")
            
            return preprocessed_data, preprocessing_stats
            
        except Exception as e:
            self.logger.error(f"데이터 전처리 중 오류: {e}")
            raise
    
    def _remove_neighbor_zero_samples(self, samples: List[Optional[float]]) -> Tuple[List[Optional[float]], int]:
        """
        이웃 시각 0 처리
        
        n-1 또는 n+1 시각의 값이 0이면 n 시각을 제외
        
        Args:
            samples: 원본 샘플 리스트
            
        Returns:
            Tuple[List, int]: (정리된 샘플, 제거된 샘플 수)
        """
        try:
            if len(samples) <= 2:
                return samples, 0  # 너무 짧은 시리즈는 그대로 유지
            
            cleaned_samples = []
            removed_count = 0
            
            for i, sample in enumerate(samples):
                should_remove = False
                
                # 이웃 시각 체크
                if i > 0 and samples[i-1] == 0:  # n-1이 0
                    should_remove = True
                if i < len(samples) - 1 and samples[i+1] == 0:  # n+1이 0
                    should_remove = True
                
                if should_remove:
                    removed_count += 1
                    self.logger.debug(f"Removing sample at index {i} due to neighbor zero")
                else:
                    cleaned_samples.append(sample)
            
            return cleaned_samples, removed_count
            
        except Exception as e:
            self.logger.error(f"이웃 시각 0 처리 중 오류: {e}")
            return samples, 0
    
    def _remove_zero_sum_samples(self, samples: List[Optional[float]], peg_name: str) -> Tuple[List[Optional[float]], int]:
        """
        DL/UL 합계 0 처리
        
        DL + UL 처리량 합계가 0인 경우 제외
        (현재는 개별 PEG 기준으로 0값 제거로 단순화)
        
        Args:
            samples: 샘플 리스트
            peg_name: PEG 이름
            
        Returns:
            Tuple[List, int]: (정리된 샘플, 제거된 샘플 수)
        """
        try:
            cleaned_samples = []
            removed_count = 0
            
            for sample in samples:
                if sample == 0 or sample == 0.0:
                    removed_count += 1
                    self.logger.debug(f"Removing zero sample from {peg_name}")
                else:
                    cleaned_samples.append(sample)
            
            return cleaned_samples, removed_count
            
        except Exception as e:
            self.logger.error(f"DL/UL 합계 0 처리 중 오류: {e}")
            return samples, 0
    
    def _is_throughput_peg(self, peg_name: str) -> bool:
        """
        처리량 관련 PEG인지 확인
        
        Args:
            peg_name: PEG 이름
            
        Returns:
            bool: 처리량 PEG 여부
        """
        throughput_keywords = ["Thru", "Throughput", "AirMac", "IpThru"]
        return any(keyword in peg_name for keyword in throughput_keywords)
    
    def _calculate_medians(self, 
                          preprocessed_data: Dict[str, List[PegSampleSeries]]) -> Dict[str, Dict[str, float]]:
        """
        2단계: 중앙값 계산 (6장 2단계)
        
        각 통계 PEG별로 샘플의 중앙값 산출
        
        Args:
            preprocessed_data: 전처리된 데이터
            
        Returns:
            Dict[str, Dict[str, float]]: 셀별 PEG별 중앙값
        """
        try:
            self.logger.debug("중앙값 계산 시작")
            
            median_values = {}
            
            for cell_id, peg_series_list in preprocessed_data.items():
                cell_medians = {}
                
                for series in peg_series_list:
                    # Pre 기간 중앙값
                    pre_valid = [s for s in series.pre_samples if s is not None]
                    if pre_valid:
                        pre_median = float(np.median(pre_valid))
                    else:
                        pre_median = 0.0
                        self.logger.warning(f"No valid pre samples for {series.peg_name} in {cell_id}")
                    
                    # Post 기간 중앙값
                    post_valid = [s for s in series.post_samples if s is not None]
                    if post_valid:
                        post_median = float(np.median(post_valid))
                    else:
                        post_median = 0.0
                        self.logger.warning(f"No valid post samples for {series.peg_name} in {cell_id}")
                    
                    # 전체 중앙값 (pre + post 통합)
                    all_valid = pre_valid + post_valid
                    if all_valid:
                        overall_median = float(np.median(all_valid))
                    else:
                        overall_median = 0.0
                        self.logger.warning(f"No valid samples for {series.peg_name} in {cell_id}")
                    
                    cell_medians[series.peg_name] = {
                        "pre_median": pre_median,
                        "post_median": post_median,
                        "overall_median": overall_median,
                        "pre_sample_count": len(pre_valid),
                        "post_sample_count": len(post_valid)
                    }
                    
                    self.logger.debug(f"중앙값 계산 완료 - {cell_id}.{series.peg_name}: "
                                    f"pre={pre_median:.2f}, post={post_median:.2f}, overall={overall_median:.2f}")
                
                median_values[cell_id] = cell_medians
            
            total_medians = sum(len(cell_medians) for cell_medians in median_values.values())
            self.logger.info(f"중앙값 계산 완료: {total_medians} PEG medians calculated")
            
            return median_values
            
        except Exception as e:
            self.logger.error(f"중앙값 계산 중 오류: {e}")
            raise
    
    def _normalize_by_median(self, 
                           preprocessed_data: Dict[str, List[PegSampleSeries]], 
                           median_values: Dict[str, Dict[str, float]]) -> Dict[str, List[PegSampleSeries]]:
        """
        3단계: 시계열 정규화 (6장 3단계)
        
        각 샘플값을 중앙값으로 나누어 정규화 (sample/median)
        
        Args:
            preprocessed_data: 전처리된 데이터
            median_values: PEG별 중앙값
            
        Returns:
            Dict[str, List[PegSampleSeries]]: 정규화된 데이터
        """
        try:
            self.logger.debug("시계열 정규화 시작")
            
            normalized_data = {}
            total_normalized = 0
            division_by_zero_count = 0
            
            for cell_id, peg_series_list in preprocessed_data.items():
                normalized_series = []
                cell_medians = median_values.get(cell_id, {})
                
                for series in peg_series_list:
                    peg_median_info = cell_medians.get(series.peg_name, {})
                    overall_median = peg_median_info.get("overall_median", 0.0)
                    
                    # 중앙값이 0인 경우 처리
                    if overall_median == 0:
                        division_by_zero_count += 1
                        self.logger.warning(f"Median is zero for {series.peg_name} in {cell_id}, skipping normalization")
                        # 정규화하지 않고 원본값 사용
                        normalized_pre = series.pre_samples.copy()
                        normalized_post = series.post_samples.copy()
                    else:
                        # 정규화 수행: sample / median
                        normalized_pre = []
                        for sample in series.pre_samples:
                            if sample is not None:
                                normalized_value = sample / overall_median
                                normalized_pre.append(normalized_value)
                            else:
                                normalized_pre.append(None)
                        
                        normalized_post = []
                        for sample in series.post_samples:
                            if sample is not None:
                                normalized_value = sample / overall_median
                                normalized_post.append(normalized_value)
                            else:
                                normalized_post.append(None)
                    
                    # 정규화된 시리즈 생성
                    normalized_series_obj = PegSampleSeries(
                        peg_name=series.peg_name,
                        cell_id=series.cell_id,
                        pre_samples=normalized_pre,
                        post_samples=normalized_post,
                        unit="normalized"  # 정규화된 데이터는 단위 없음
                    )
                    
                    normalized_series.append(normalized_series_obj)
                    total_normalized += 1
                    
                    self.logger.debug(f"정규화 완료 - {cell_id}.{series.peg_name}: "
                                    f"median={overall_median:.2f}, pre_samples={len(normalized_pre)}, post_samples={len(normalized_post)}")
                
                normalized_data[cell_id] = normalized_series
            
            self.logger.info(f"정규화 완료: {total_normalized} PEG series normalized, "
                           f"division_by_zero_cases={division_by_zero_count}")
            
            return normalized_data
            
        except Exception as e:
            self.logger.error(f"정규화 중 오류: {e}")
            raise
    
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
        try:
            # 기본 검증 (부모 클래스)
            if not super().validate_input(peg_data, config):
                return False
            
            # Choi 필터링 특화 검증
            for cell_id, peg_series_list in peg_data.items():
                if not peg_series_list:
                    self.logger.error(f"Empty PEG series list for cell: {cell_id}")
                    return False
                
                for series in peg_series_list:
                    if not series.pre_samples and not series.post_samples:
                        self.logger.error(f"No sample data for {series.peg_name} in {cell_id}")
                        return False
            
            # 필수 설정 확인
            required_config_keys = ['min_threshold', 'max_threshold', 'filter_ratio']
            for key in required_config_keys:
                if key not in config:
                    self.logger.error(f"Required filtering config key missing: {key}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"입력 검증 중 오류: {e}")
            return False


# =============================================================================
# 초기화 및 로깅
# =============================================================================

logger.info("Choi Filtering Service loaded successfully")
