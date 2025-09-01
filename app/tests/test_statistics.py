"""
Statistics 비교 분석 API 및 로직 단위 테스트

이 모듈은 Statistics 비교 분석 기능의 정확성을 보장하기 위한
포괄적인 단위 테스트를 제공합니다.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np
import pandas as pd

from app.models.statistics import (
    StatisticsCompareRequest, DateRange, PegStatistics, PegComparisonResult,
    calculate_improvement_status, calculate_improvement_magnitude
)
from app.utils.statistics_analyzer import StatisticsAnalyzer, validate_data_consistency
from app.utils.statistics_db import StatisticsDataBase
import logging

# 로거 설정
logger = logging.getLogger(__name__)

class TestStatisticsModels:
    """Statistics Pydantic 모델 테스트"""
    
    def test_date_range_validation(self):
        """날짜 범위 검증 테스트"""
        
        # 정상적인 날짜 범위
        valid_range = DateRange(
            start_date=datetime(2025, 8, 1),
            end_date=datetime(2025, 8, 7)
        )
        assert valid_range.start_date < valid_range.end_date
        
        # 잘못된 날짜 범위 (종료일이 시작일보다 이전)
        with pytest.raises(ValueError, match="종료 날짜는 시작 날짜보다 이후여야 합니다"):
            DateRange(
                start_date=datetime(2025, 8, 7),
                end_date=datetime(2025, 8, 1)
            )
    
    def test_statistics_compare_request_validation(self):
        """비교 분석 요청 모델 검증 테스트"""
        
        # 정상적인 요청
        valid_request = StatisticsCompareRequest(
            period1=DateRange(
                start_date=datetime(2025, 8, 1),
                end_date=datetime(2025, 8, 7)
            ),
            period2=DateRange(
                start_date=datetime(2025, 8, 8),
                end_date=datetime(2025, 8, 14)
            ),
            peg_names=["availability", "rrc", "erab"]
        )
        assert len(valid_request.peg_names) == 3
        assert valid_request.decimal_places == 4  # 기본값
        assert valid_request.include_outliers is True  # 기본값
        
        # 빈 PEG 목록 (에러)
        with pytest.raises(ValueError):
            StatisticsCompareRequest(
                period1=DateRange(
                    start_date=datetime(2025, 8, 1),
                    end_date=datetime(2025, 8, 7)
                ),
                period2=DateRange(
                    start_date=datetime(2025, 8, 8),
                    end_date=datetime(2025, 8, 14)
                ),
                peg_names=[]  # 빈 목록
            )

class TestImprovementCalculations:
    """개선 상태 및 정도 계산 테스트"""
    
    def test_improvement_status_calculation(self):
        """개선 상태 계산 테스트"""
        
        # 높을수록 좋은 PEG (availability)
        assert calculate_improvement_status(0.5, "availability") == "improved"
        assert calculate_improvement_status(-0.3, "availability") == "degraded"
        assert calculate_improvement_status(0.0, "availability") == "stable"
        
        # 낮을수록 좋은 PEG (latency)
        assert calculate_improvement_status(-0.5, "latency") == "improved"
        assert calculate_improvement_status(0.3, "latency") == "degraded"
        assert calculate_improvement_status(0.0, "latency") == "stable"
        
        # 알 수 없는 PEG (기본적으로 증가=개선)
        assert calculate_improvement_status(0.1, "unknown_peg") == "improved"
        assert calculate_improvement_status(-0.1, "unknown_peg") == "degraded"
    
    def test_improvement_magnitude_calculation(self):
        """개선 정도 계산 테스트"""
        
        assert calculate_improvement_magnitude(10.0) == "significant"  # > 5%
        assert calculate_improvement_magnitude(3.0) == "moderate"      # 1-5%
        assert calculate_improvement_magnitude(0.5) == "minor"         # 0.1-1%
        assert calculate_improvement_magnitude(0.05) == "none"         # < 0.1%
        
        # 음수 값도 절댓값으로 처리
        assert calculate_improvement_magnitude(-8.0) == "significant"

class TestStatisticsAnalyzer:
    """StatisticsAnalyzer 클래스 테스트"""
    
    @pytest.fixture
    def analyzer(self):
        """테스트용 분석기 인스턴스"""
        return StatisticsAnalyzer(decimal_places=3)
    
    @pytest.fixture
    def sample_data(self):
        """테스트용 샘플 데이터"""
        base_date = datetime(2025, 8, 1)
        data = []
        
        # availability 데이터 (99.0 ± 0.5)
        for i in range(100):
            data.append({
                'timestamp': base_date + timedelta(hours=i),
                'peg_name': 'availability',
                'value': 99.0 + np.random.normal(0, 0.5),
                'ne': 'nvgnb#10000',
                'cell_id': '2010'
            })
        
        # rrc 데이터 (98.5 ± 1.0) 
        for i in range(100):
            data.append({
                'timestamp': base_date + timedelta(hours=i),
                'peg_name': 'rrc',
                'value': 98.5 + np.random.normal(0, 1.0),
                'ne': 'nvgnb#10000',
                'cell_id': '2010'
            })
        
        return data
    
    def test_basic_statistics_calculation(self, analyzer, sample_data):
        """기본 통계 계산 테스트"""
        
        stats = analyzer.calculate_basic_statistics(sample_data)
        
        # availability 통계 확인
        availability_stats = stats['availability']
        assert isinstance(availability_stats, PegStatistics)
        assert availability_stats.count == 100
        assert 98.0 <= availability_stats.mean <= 100.0  # 대략적인 범위
        assert availability_stats.std > 0
        assert availability_stats.min <= availability_stats.mean <= availability_stats.max
        
        # rrc 통계 확인
        rrc_stats = stats['rrc']
        assert isinstance(rrc_stats, PegStatistics)
        assert rrc_stats.count == 100
        assert 97.0 <= rrc_stats.mean <= 100.0
    
    def test_rsd_calculation(self, analyzer):
        """RSD (상대표준편차) 계산 테스트"""
        
        # 정상 케이스
        rsd = analyzer.calculate_rsd(100.0, 2.0)
        assert rsd == 2.0  # (2/100) * 100 = 2%
        
        # 평균이 0인 경우
        rsd_zero = analyzer.calculate_rsd(0.0, 1.0)
        assert rsd_zero == 0.0
        
        # NaN 값 처리
        rsd_nan = analyzer.calculate_rsd(float('nan'), 1.0)
        assert rsd_nan == 0.0
    
    def test_comparison_statistics(self, analyzer):
        """비교 통계 계산 테스트"""
        
        # 기간1 데이터 (평균 99.0)
        period1_data = []
        for i in range(50):
            period1_data.append({
                'peg_name': 'availability',
                'value': 99.0 + np.random.normal(0, 0.1),
                'timestamp': datetime(2025, 8, 1) + timedelta(hours=i)
            })
        
        # 기간2 데이터 (평균 99.5 - 개선됨)
        period2_data = []
        for i in range(50):
            period2_data.append({
                'peg_name': 'availability',
                'value': 99.5 + np.random.normal(0, 0.1),
                'timestamp': datetime(2025, 8, 8) + timedelta(hours=i)
            })
        
        results = analyzer.calculate_comparison_statistics(
            period1_data, period2_data, ['availability']
        )
        
        assert len(results) == 1
        
        result = results[0]
        assert isinstance(result, PegComparisonResult)
        assert result.peg_name == 'availability'
        assert result.delta > 0  # 개선되었으므로 양수
        assert result.improvement_status == 'improved'
        assert result.period1_stats.count == 50
        assert result.period2_stats.count == 50
    
    def test_summary_statistics(self, analyzer):
        """요약 통계 계산 테스트"""
        
        # 샘플 비교 결과 생성
        comparison_results = [
            PegComparisonResult(
                peg_name="availability",
                period1_stats=PegStatistics(count=100, mean=99.0, std=0.5, min=98.0, max=99.8),
                period2_stats=PegStatistics(count=100, mean=99.3, std=0.4, min=98.5, max=99.9),
                delta=0.3,
                delta_percentage=0.303,
                rsd_period1=0.505,
                rsd_period2=0.404,
                improvement_status="improved",
                improvement_magnitude="minor"
            ),
            PegComparisonResult(
                peg_name="rrc",
                period1_stats=PegStatistics(count=100, mean=98.5, std=1.0, min=96.0, max=99.5),
                period2_stats=PegStatistics(count=100, mean=98.2, std=0.8, min=96.5, max=99.2),
                delta=-0.3,
                delta_percentage=-0.305,
                rsd_period1=1.015,
                rsd_period2=0.815,
                improvement_status="degraded",
                improvement_magnitude="minor"
            )
        ]
        
        summary = analyzer.calculate_summary_statistics(comparison_results)
        
        assert summary["total_pegs_analyzed"] == 2
        assert summary["improved_count"] == 1
        assert summary["degraded_count"] == 1
        assert summary["stable_count"] == 0
        assert isinstance(summary["avg_improvement"], (int, float))

class TestDataValidation:
    """데이터 일관성 검증 테스트"""
    
    def test_data_consistency_validation(self):
        """데이터 일관성 검증 테스트"""
        
        # 일관된 데이터
        period1_data = [
            {'peg_name': 'availability', 'value': 99.0},
            {'peg_name': 'rrc', 'value': 98.5}
        ]
        
        period2_data = [
            {'peg_name': 'availability', 'value': 99.2},
            {'peg_name': 'rrc', 'value': 98.7},
            {'peg_name': 'erab', 'value': 99.1}  # 추가 PEG
        ]
        
        result = validate_data_consistency(period1_data, period2_data, ['availability', 'rrc'])
        
        assert 'availability' in result["common_pegs"]
        assert 'rrc' in result["common_pegs"]
        assert 'erab' in result["missing_from_period1"]
        assert result["period1_data_count"] == 2
        assert result["period2_data_count"] == 3
    
    def test_inconsistent_data_validation(self):
        """일관성 없는 데이터 검증 테스트"""
        
        period1_data = [{'peg_name': 'availability', 'value': 99.0}]
        period2_data = [{'peg_name': 'rrc', 'value': 98.5}]
        
        result = validate_data_consistency(period1_data, period2_data, ['availability', 'rrc'])
        
        assert not result["is_consistent"]
        assert len(result["common_pegs"]) == 0
        assert 'availability' in result["missing_from_period2"]
        assert 'rrc' in result["missing_from_period1"]

class TestStatisticsIntegration:
    """통합 테스트"""
    
    def generate_realistic_test_data(self, peg_name: str, base_value: float, 
                                   std: float, count: int, 
                                   period_start: datetime) -> List[Dict[str, Any]]:
        """현실적인 테스트 데이터 생성"""
        data = []
        
        for i in range(count):
            # 시간대별 패턴 추가 (예: 피크 시간에는 성능 저하)
            hour = (period_start + timedelta(hours=i)).hour
            if 9 <= hour <= 17:  # 업무 시간
                adjustment = -0.5  # 성능 저하
            elif 22 <= hour or hour <= 6:  # 심야 시간
                adjustment = 0.3   # 성능 향상
            else:
                adjustment = 0.0
            
            value = base_value + adjustment + np.random.normal(0, std)
            value = max(0, min(100, value))  # 0-100% 범위 제한
            
            data.append({
                'timestamp': period_start + timedelta(hours=i),
                'peg_name': peg_name,
                'value': round(value, 4),
                'ne': f'nvgnb#{10000 + (i % 3) * 10000}',
                'cell_id': f'20{10 + (i % 10)}'
            })
        
        return data
    
    def test_full_analysis_workflow(self):
        """전체 분석 워크플로우 테스트"""
        
        analyzer = StatisticsAnalyzer(decimal_places=4)
        
        # 현실적인 테스트 데이터 생성
        period1_start = datetime(2025, 8, 1)
        period2_start = datetime(2025, 8, 8)
        
        # 기간1: availability 99.5% (안정적)
        period1_data = self.generate_realistic_test_data(
            'availability', 99.5, 0.3, 168, period1_start  # 7일 * 24시간
        )
        
        # 기간2: availability 99.8% (개선됨)
        period2_data = self.generate_realistic_test_data(
            'availability', 99.8, 0.2, 168, period2_start
        )
        
        # rrc 데이터도 추가
        period1_data.extend(
            self.generate_realistic_test_data('rrc', 98.5, 0.8, 168, period1_start)
        )
        period2_data.extend(
            self.generate_realistic_test_data('rrc', 98.2, 1.0, 168, period2_start)
        )
        
        # 비교 분석 수행
        results = analyzer.calculate_comparison_statistics(
            period1_data, period2_data, ['availability', 'rrc']
        )
        
        # 결과 검증
        assert len(results) == 2
        
        # availability 결과 확인
        avail_result = next(r for r in results if r.peg_name == 'availability')
        assert avail_result.improvement_status == 'improved'
        assert avail_result.delta > 0
        assert avail_result.period1_stats.count == 168
        assert avail_result.period2_stats.count == 168
        
        # rrc 결과 확인 (악화됨)
        rrc_result = next(r for r in results if r.peg_name == 'rrc')
        assert rrc_result.improvement_status == 'degraded'
        assert rrc_result.delta < 0
        
        # 요약 통계 생성
        summary = analyzer.calculate_summary_statistics(results)
        assert summary["total_pegs_analyzed"] == 2
        assert summary["improved_count"] == 1
        assert summary["degraded_count"] == 1
        
        logger.info(f"✅ 전체 워크플로우 테스트 완료")
        logger.info(f"📊 결과: {summary}")

# 성능 테스트
class TestPerformance:
    """성능 테스트"""
    
    @pytest.mark.performance
    def test_large_dataset_performance(self):
        """대용량 데이터셋 성능 테스트"""
        import time
        
        analyzer = StatisticsAnalyzer()
        
        # 대용량 데이터 생성 (1만개 포인트)
        large_data = []
        base_date = datetime(2025, 8, 1)
        
        for i in range(10000):
            large_data.append({
                'timestamp': base_date + timedelta(minutes=i),
                'peg_name': f'peg_{i % 5}',  # 5개 PEG
                'value': 99.0 + np.random.normal(0, 1.0),
                'ne': f'nvgnb#{10000 + (i % 10) * 1000}',
                'cell_id': f'cell_{i % 20}'
            })
        
        # 성능 측정
        start_time = time.time()
        stats = analyzer.calculate_basic_statistics(large_data)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        assert len(stats) == 5  # 5개 PEG
        assert processing_time < 5.0  # 5초 이내 처리
        
        logger.info(f"📈 대용량 데이터 처리 시간: {processing_time:.3f}초")

if __name__ == "__main__":
    # 테스트 실행
    pytest.main([__file__, "-v", "--tb=short"])

