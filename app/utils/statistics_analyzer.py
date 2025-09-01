"""
Statistics 비교 분석을 위한 Pandas 기반 통계 분석 유틸리티

이 모듈은 조회된 KPI 데이터를 Pandas를 사용하여 처리하고,
통계 분석(평균, Delta, RSD, t-검정 등)을 수행합니다.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from scipy import stats
import logging

from ..models.statistics import (
    PegStatistics, PegComparisonResult, 
    calculate_improvement_status, calculate_improvement_magnitude
)

# 로거 설정
logger = logging.getLogger(__name__)

class StatisticsAnalyzer:
    """Pandas 기반 통계 분석 클래스"""
    
    def __init__(self, decimal_places: int = 4):
        """
        통계 분석기 초기화
        
        Args:
            decimal_places: 소수점 자릿수 (기본값: 4)
        """
        self.decimal_places = decimal_places
        
    def round_value(self, value: float) -> float:
        """값을 지정된 소수점 자릿수로 반올림"""
        if pd.isna(value) or not np.isfinite(value):
            return 0.0
        return round(float(value), self.decimal_places)
    
    def calculate_basic_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, PegStatistics]:
        """
        기본 통계 계산 (평균, 표준편차, 최대/최소값 등)
        
        Args:
            data: KPI 데이터 리스트
            
        Returns:
            PEG별 기본 통계 딕셔너리
        """
        try:
            logger.info(f"기본 통계 계산 시작 - 데이터 포인트: {len(data)}개")
            
            if not data:
                logger.warning("데이터가 없어 빈 통계 반환")
                return {}
            
            # DataFrame 생성
            df = pd.DataFrame(data)
            
            # 숫자 변환
            df['value_numeric'] = pd.to_numeric(df['value'], errors='coerce')
            
            # NaN 값 제거
            df = df.dropna(subset=['value_numeric'])
            
            if df.empty:
                logger.warning("유효한 숫자 데이터가 없음")
                return {}
            
            logger.info(f"유효한 데이터 포인트: {len(df)}개")
            
            # PEG별 통계 계산
            peg_stats = {}
            
            for peg_name in df['peg_name'].unique():
                peg_data = df[df['peg_name'] == peg_name]['value_numeric']
                
                if len(peg_data) == 0:
                    continue
                
                # 기본 통계 계산
                stats_dict = {
                    'count': len(peg_data),
                    'mean': self.round_value(peg_data.mean()),
                    'std': self.round_value(peg_data.std()),
                    'min': self.round_value(peg_data.min()),
                    'max': self.round_value(peg_data.max())
                }
                
                # 분위수 계산 (데이터가 충분한 경우)
                if len(peg_data) >= 4:
                    stats_dict.update({
                        'median': self.round_value(peg_data.median()),
                        'percentile_25': self.round_value(peg_data.quantile(0.25)),
                        'percentile_75': self.round_value(peg_data.quantile(0.75))
                    })
                else:
                    stats_dict.update({
                        'median': stats_dict['mean'],
                        'percentile_25': stats_dict['min'],
                        'percentile_75': stats_dict['max']
                    })
                
                peg_stats[peg_name] = PegStatistics(**stats_dict)
                
                logger.info(f"PEG '{peg_name}' 통계 계산 완료 - 데이터: {len(peg_data)}개")
            
            logger.info(f"기본 통계 계산 완료 - PEG 수: {len(peg_stats)}")
            return peg_stats
            
        except Exception as e:
            logger.error(f"기본 통계 계산 실패: {e}")
            raise
    
    def calculate_comparison_statistics(
        self,
        period1_data: List[Dict[str, Any]],
        period2_data: List[Dict[str, Any]],
        peg_names: List[str]
    ) -> List[PegComparisonResult]:
        """
        두 기간의 비교 통계 계산
        
        Args:
            period1_data: 첫 번째 기간 데이터
            period2_data: 두 번째 기간 데이터
            peg_names: 분석할 PEG 이름 목록
            
        Returns:
            PEG별 비교 분석 결과 리스트
        """
        try:
            logger.info("비교 통계 계산 시작")
            
            # 각 기간의 기본 통계 계산
            period1_stats = self.calculate_basic_statistics(period1_data)
            period2_stats = self.calculate_basic_statistics(period2_data)
            
            comparison_results = []
            
            for peg_name in peg_names:
                try:
                    # 두 기간 모두에 데이터가 있는지 확인
                    if peg_name not in period1_stats or peg_name not in period2_stats:
                        logger.warning(f"PEG '{peg_name}'의 데이터가 부족하여 비교 분석 생략")
                        continue
                    
                    p1_stats = period1_stats[peg_name]
                    p2_stats = period2_stats[peg_name]
                    
                    # Delta 계산
                    delta = self.round_value(p2_stats.mean - p1_stats.mean)
                    
                    # Delta 백분율 계산 (0으로 나누기 방지)
                    if p1_stats.mean != 0:
                        delta_percentage = self.round_value((delta / abs(p1_stats.mean)) * 100)
                    else:
                        delta_percentage = 0.0
                    
                    # RSD (상대표준편차) 계산
                    rsd_period1 = self.calculate_rsd(p1_stats.mean, p1_stats.std)
                    rsd_period2 = self.calculate_rsd(p2_stats.mean, p2_stats.std)
                    
                    # 통계적 유의성 검정 (t-test)
                    t_stat, p_value, is_significant = self.perform_t_test(
                        period1_data, period2_data, peg_name
                    )
                    
                    # 개선 상태 및 정도 계산
                    improvement_status = calculate_improvement_status(delta, peg_name)
                    improvement_magnitude = calculate_improvement_magnitude(abs(delta_percentage))
                    
                    # 비교 결과 생성
                    comparison_result = PegComparisonResult(
                        peg_name=peg_name,
                        period1_stats=p1_stats,
                        period2_stats=p2_stats,
                        delta=delta,
                        delta_percentage=delta_percentage,
                        rsd_period1=rsd_period1,
                        rsd_period2=rsd_period2,
                        t_statistic=t_stat,
                        p_value=p_value,
                        is_significant=is_significant,
                        improvement_status=improvement_status,
                        improvement_magnitude=improvement_magnitude
                    )
                    
                    comparison_results.append(comparison_result)
                    
                    logger.info(f"PEG '{peg_name}' 비교 분석 완료 - Delta: {delta}, 상태: {improvement_status}")
                    
                except Exception as e:
                    logger.error(f"PEG '{peg_name}' 비교 분석 실패: {e}")
                    continue
            
            logger.info(f"비교 통계 계산 완료 - 분석된 PEG: {len(comparison_results)}개")
            return comparison_results
            
        except Exception as e:
            logger.error(f"비교 통계 계산 실패: {e}")
            raise
    
    def calculate_rsd(self, mean: float, std: float) -> float:
        """
        RSD (상대표준편차) 계산
        
        Args:
            mean: 평균값
            std: 표준편차
            
        Returns:
            RSD 백분율 값
        """
        if mean == 0 or pd.isna(mean) or pd.isna(std):
            return 0.0
        
        rsd = (std / abs(mean)) * 100
        return self.round_value(rsd)
    
    def perform_t_test(
        self,
        period1_data: List[Dict[str, Any]],
        period2_data: List[Dict[str, Any]],
        peg_name: str,
        alpha: float = 0.05
    ) -> Tuple[Optional[float], Optional[float], Optional[bool]]:
        """
        두 기간 데이터에 대한 t-검정 수행
        
        Args:
            period1_data: 첫 번째 기간 데이터
            period2_data: 두 번째 기간 데이터
            peg_name: PEG 이름
            alpha: 유의수준 (기본값: 0.05)
            
        Returns:
            (t_statistic, p_value, is_significant) 튜플
        """
        try:
            # PEG별 데이터 추출
            p1_values = [
                float(item['value']) for item in period1_data 
                if item['peg_name'] == peg_name and pd.notna(item['value'])
            ]
            
            p2_values = [
                float(item['value']) for item in period2_data
                if item['peg_name'] == peg_name and pd.notna(item['value'])
            ]
            
            # 최소 데이터 요구사항 확인
            if len(p1_values) < 3 or len(p2_values) < 3:
                logger.warning(f"PEG '{peg_name}' t-검정용 데이터 부족")
                return None, None, None
            
            # 독립표본 t-검정 수행
            t_statistic, p_value = stats.ttest_ind(p1_values, p2_values)
            
            # 유의성 판정
            is_significant = p_value < alpha
            
            return (
                self.round_value(t_statistic),
                self.round_value(p_value),
                is_significant
            )
            
        except Exception as e:
            logger.error(f"PEG '{peg_name}' t-검정 실패: {e}")
            return None, None, None
    
    def calculate_summary_statistics(
        self,
        comparison_results: List[PegComparisonResult]
    ) -> Dict[str, Any]:
        """
        전체 비교 결과의 요약 통계 계산
        
        Args:
            comparison_results: PEG별 비교 분석 결과
            
        Returns:
            요약 통계 딕셔너리
        """
        try:
            logger.info("요약 통계 계산 시작")
            
            if not comparison_results:
                return {
                    "total_pegs_analyzed": 0,
                    "improved_count": 0,
                    "degraded_count": 0,
                    "stable_count": 0,
                    "avg_improvement": 0.0,
                    "significant_changes": 0,
                    "max_improvement": 0.0,
                    "max_degradation": 0.0
                }
            
            # 상태별 카운트
            status_counts = {
                'improved': 0,
                'degraded': 0,
                'stable': 0
            }
            
            # 델타 및 유의성 통계
            deltas = []
            significant_changes = 0
            
            for result in comparison_results:
                status_counts[result.improvement_status] += 1
                deltas.append(result.delta_percentage)
                
                if result.is_significant:
                    significant_changes += 1
            
            # 요약 통계 계산
            summary = {
                "total_pegs_analyzed": len(comparison_results),
                "improved_count": status_counts['improved'],
                "degraded_count": status_counts['degraded'],
                "stable_count": status_counts['stable'],
                "avg_improvement": self.round_value(np.mean(deltas)),
                "significant_changes": significant_changes,
                "max_improvement": self.round_value(max(deltas)) if deltas else 0.0,
                "max_degradation": self.round_value(min(deltas)) if deltas else 0.0,
                "std_delta": self.round_value(np.std(deltas)) if len(deltas) > 1 else 0.0
            }
            
            logger.info(f"요약 통계 계산 완료: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"요약 통계 계산 실패: {e}")
            raise
    
    def detect_outliers(
        self,
        data: List[Dict[str, Any]],
        method: str = 'iqr'
    ) -> Dict[str, List[int]]:
        """
        이상치 탐지
        
        Args:
            data: KPI 데이터 리스트
            method: 탐지 방법 ('iqr', 'zscore')
            
        Returns:
            PEG별 이상치 인덱스 딕셔너리
        """
        try:
            logger.info(f"이상치 탐지 시작 - 방법: {method}")
            
            if not data:
                return {}
            
            df = pd.DataFrame(data)
            df['value_numeric'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['value_numeric'])
            
            outliers = {}
            
            for peg_name in df['peg_name'].unique():
                peg_data = df[df['peg_name'] == peg_name]
                peg_outliers = []
                
                if method == 'iqr':
                    # IQR 방법
                    Q1 = peg_data['value_numeric'].quantile(0.25)
                    Q3 = peg_data['value_numeric'].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outlier_mask = (
                        (peg_data['value_numeric'] < lower_bound) |
                        (peg_data['value_numeric'] > upper_bound)
                    )
                    
                elif method == 'zscore':
                    # Z-Score 방법
                    z_scores = np.abs(stats.zscore(peg_data['value_numeric']))
                    outlier_mask = z_scores > 3
                
                else:
                    logger.error(f"지원되지 않는 이상치 탐지 방법: {method}")
                    continue
                
                peg_outliers = peg_data[outlier_mask].index.tolist()
                outliers[peg_name] = peg_outliers
                
                logger.info(f"PEG '{peg_name}' 이상치: {len(peg_outliers)}개")
            
            return outliers
            
        except Exception as e:
            logger.error(f"이상치 탐지 실패: {e}")
            raise
    
    def generate_analysis_metadata(
        self,
        start_time: datetime,
        period1_data_count: int,
        period2_data_count: int,
        peg_names: List[str],
        filters_applied: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        분석 메타데이터 생성
        
        Args:
            start_time: 분석 시작 시간
            period1_data_count: 기간1 데이터 수
            period2_data_count: 기간2 데이터 수  
            peg_names: 요청된 PEG 목록
            filters_applied: 적용된 필터 정보
            
        Returns:
            메타데이터 딕셔너리
        """
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        metadata = {
            "analysis_timestamp": end_time.isoformat() + "Z",
            "processing_time_ms": round(processing_time, 2),
            "data_source": "mongodb",
            "decimal_places": self.decimal_places,
            "period1_data_count": period1_data_count,
            "period2_data_count": period2_data_count,
            "requested_pegs": peg_names,
            "filters_applied": filters_applied,
            "analysis_version": "1.0.0"
        }
        
        return metadata

def validate_data_consistency(
    period1_data: List[Dict[str, Any]],
    period2_data: List[Dict[str, Any]],
    peg_names: List[str]
) -> Dict[str, Any]:
    """
    두 기간 데이터의 일관성 검증
    
    Args:
        period1_data: 첫 번째 기간 데이터
        period2_data: 두 번째 기간 데이터
        peg_names: 요청된 PEG 목록
        
    Returns:
        데이터 일관성 검증 결과
    """
    try:
        logger.info("데이터 일관성 검증 시작")
        
        # 기간별 PEG 가용성 확인
        p1_pegs = set(item['peg_name'] for item in period1_data)
        p2_pegs = set(item['peg_name'] for item in period2_data)
        
        common_pegs = p1_pegs & p2_pegs
        missing_from_p1 = p2_pegs - p1_pegs
        missing_from_p2 = p1_pegs - p2_pegs
        
        # 요청된 PEG 중 사용 가능한 것들
        available_pegs = common_pegs & set(peg_names)
        unavailable_pegs = set(peg_names) - available_pegs
        
        validation_result = {
            "is_consistent": len(unavailable_pegs) == 0,
            "common_pegs": list(common_pegs),
            "available_pegs": list(available_pegs),
            "unavailable_pegs": list(unavailable_pegs),
            "missing_from_period1": list(missing_from_p1),
            "missing_from_period2": list(missing_from_p2),
            "period1_data_count": len(period1_data),
            "period2_data_count": len(period2_data)
        }
        
        logger.info(f"데이터 일관성 검증 완료: {validation_result['is_consistent']}")
        return validation_result
        
    except Exception as e:
        logger.error(f"데이터 일관성 검증 실패: {e}")
        raise

if __name__ == "__main__":
    # 테스트 코드
    import random
    from datetime import timedelta
    
    def generate_test_data(peg_names: List[str], count: int) -> List[Dict[str, Any]]:
        """테스트용 데이터 생성"""
        base_date = datetime(2025, 8, 1)
        data = []
        
        for i in range(count):
            for peg in peg_names:
                data.append({
                    'timestamp': base_date + timedelta(hours=i),
                    'peg_name': peg,
                    'value': random.gauss(99.0, 1.0),
                    'ne': f'nvgnb#{random.randint(10000, 30000)}',
                    'cell_id': f'20{random.randint(10, 19)}'
                })
        
        return data
    
    # 분석기 테스트
    analyzer = StatisticsAnalyzer(decimal_places=4)
    
    # 테스트 데이터 생성
    test_pegs = ['availability', 'rrc', 'erab']
    period1_data = generate_test_data(test_pegs, 100)
    period2_data = generate_test_data(test_pegs, 100)
    
    # 비교 분석 수행
    results = analyzer.calculate_comparison_statistics(period1_data, period2_data, test_pegs)
    summary = analyzer.calculate_summary_statistics(results)
    
    print(f"✅ 테스트 완료 - 분석된 PEG: {len(results)}개")
    print(f"📊 요약: {summary}")

