"""
독립적인 통계 테스트 서비스

이 모듈은 마할라노비스 분석과 별개로 사용할 수 있는
독립적인 통계 테스트 함수들을 제공합니다.
"""

import logging
import numpy as np
import scipy.stats as stats
from typing import List, Dict, Any, Tuple
from datetime import datetime

from ..models.mahalanobis import StatisticalTestResult
from ..utils.cache_manager import get_cache_manager

logger = logging.getLogger(__name__)


class StatisticalTestsService:
    """
    독립적인 통계 테스트 서비스

    마할라노비스 분석에서 사용하는 통계 테스트들을
    독립적으로 사용할 수 있도록 제공합니다.
    """

    def __init__(self):
        """서비스 초기화"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.cache_manager = get_cache_manager()

    def mann_whitney_u_test(
        self,
        group_a: List[float],
        group_b: List[float],
        significance_level: float = 0.05
    ) -> StatisticalTestResult:
        """
        Mann-Whitney U Test 수행 (캐시 지원)

        동일한 입력에 대해 15분 동안 캐시된 결과를 반환합니다.
        """
        # 캐시 키 생성
        import hashlib
        import json

        cache_data = {
            "test": "mann_whitney_u",
            "group_a": group_a,
            "group_b": group_b,
            "significance_level": significance_level
        }

        cache_key = hashlib.md5(
            json.dumps(cache_data, sort_keys=True, default=str).encode()
        ).hexdigest()

        # 캐시 조회
        cached_result = self.cache_manager.cache.get(f"stat_test:{cache_key}")
        if cached_result is not None:
            self.logger.debug(f"캐시 히트: Mann-Whitney U Test 결과 반환")
            return cached_result

        # 캐시 미스: 계산 수행
        self.logger.debug(f"캐시 미스: Mann-Whitney U Test 계산 시작")
        result = self._perform_mann_whitney_calculation(group_a, group_b, significance_level)

        # 결과 캐싱 (15분 TTL)
        self.cache_manager.cache.set(f"stat_test:{cache_key}", result, ttl=900)

        return result

    def kolmogorov_smirnov_test(
        self,
        group_a: List[float],
        group_b: List[float],
        significance_level: float = 0.05
    ) -> StatisticalTestResult:
        """
        Kolmogorov-Smirnov Test 수행 (캐시 지원)

        동일한 입력에 대해 15분 동안 캐시된 결과를 반환합니다.
        """
        # 캐시 키 생성
        import hashlib
        import json

        cache_data = {
            "test": "kolmogorov_smirnov",
            "group_a": group_a,
            "group_b": group_b,
            "significance_level": significance_level
        }

        cache_key = hashlib.md5(
            json.dumps(cache_data, sort_keys=True, default=str).encode()
        ).hexdigest()

        # 캐시 조회
        cached_result = self.cache_manager.cache.get(f"stat_test:{cache_key}")
        if cached_result is not None:
            self.logger.debug(f"캐시 히트: Kolmogorov-Smirnov Test 결과 반환")
            return cached_result

        # 캐시 미스: 계산 수행
        self.logger.debug(f"캐시 미스: Kolmogorov-Smirnov Test 계산 시작")
        result = self._perform_ks_calculation(group_a, group_b, significance_level)

        # 결과 캐싱 (15분 TTL)
        self.cache_manager.cache.set(f"stat_test:{cache_key}", result, ttl=900)

        return result

    def _validate_input_data(
        self,
        group_a: List[float],
        group_b: List[float],
        test_name: str
    ) -> None:
        """
        입력 데이터 검증

        Args:
            group_a: 첫 번째 그룹 데이터
            group_b: 두 번째 그룹 데이터
            test_name: 테스트 이름

        Raises:
            ValueError: 검증 실패 시
        """
        if not group_a or not group_b:
            raise ValueError(f"{test_name}: 빈 데이터 그룹이 있습니다")

        if len(group_a) < 3 or len(group_b) < 3:
            raise ValueError(f"{test_name}: 각 그룹은 최소 3개의 데이터 포인트가 필요합니다")

        # 수치적 안정성 검증
        all_data = group_a + group_b
        for i, val in enumerate(all_data):
            if not np.isfinite(val):
                raise ValueError(f"{test_name}: 유효하지 않은 값 발견 (인덱스 {i}): {val}")

        # 데이터 다양성 검증
        if len(set(all_data)) == 1:
            self.logger.warning(f"{test_name}: 모든 값이 동일합니다")

    def _perform_mann_whitney_calculation(
        self,
        group_a: List[float],
        group_b: List[float]
    ) -> StatisticalTestResult:
        """
        Mann-Whitney U Test 계산 수행

        Args:
            group_a: 첫 번째 그룹 데이터
            group_b: 두 번째 그룹 데이터

        Returns:
            StatisticalTestResult: 계산 결과
        """
        # 벡터화된 계산을 위한 numpy 배열 변환
        arr_a = np.array(group_a, dtype=np.float64)
        arr_b = np.array(group_b, dtype=np.float64)

        # 수치적 예외 처리
        with np.errstate(all='raise'):
            try:
                statistic, p_value = stats.mannwhitneyu(arr_a, arr_b, alternative='two-sided')
            except (FloatingPointError, np.linalg.LinAlgError):
                self.logger.warning("Mann-Whitney U Test에서 수치적 문제 발생, 기본값 사용")
                statistic, p_value = 0.0, 1.0

        # p-value 범위 검증 및 조정
        if not np.isfinite(p_value):
            p_value = 1.0
        elif p_value > 1.0:
            p_value = 1.0
        elif p_value < 0.0:
            p_value = 0.0

        # 효과 크기 계산
        n1, n2 = len(arr_a), len(arr_b)
        try:
            z_score = (statistic - (n1 * n2 / 2)) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            effect_size = abs(z_score) / np.sqrt(n1 + n2)
        except (ZeroDivisionError, FloatingPointError):
            effect_size = 0.0

        significant = p_value < 0.05

        interpretation = (
            f"통계적으로 유의한 차이 (p={p_value:.4f})"
            if significant
            else f"통계적으로 유의하지 않은 차이 (p={p_value:.4f})"
        )

        return StatisticalTestResult(
            test_name="Mann-Whitney U",
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            interpretation=interpretation,
            effect_size=effect_size
        )

    def _perform_ks_calculation(
        self,
        group_a: List[float],
        group_b: List[float]
    ) -> StatisticalTestResult:
        """
        Kolmogorov-Smirnov Test 계산 수행

        Args:
            group_a: 첫 번째 그룹 데이터
            group_b: 두 번째 그룹 데이터

        Returns:
            StatisticalTestResult: 계산 결과
        """
        # 벡터화된 계산을 위한 numpy 배열 변환
        arr_a = np.array(group_a, dtype=np.float64)
        arr_b = np.array(group_b, dtype=np.float64)

        # 수치적 예외 처리
        with np.errstate(all='raise'):
            try:
                statistic, p_value = stats.ks_2samp(arr_a, arr_b)
            except (FloatingPointError, np.linalg.LinAlgError):
                self.logger.warning("Kolmogorov-Smirnov Test에서 수치적 문제 발생, 기본값 사용")
                statistic, p_value = 0.0, 1.0

        # p-value 범위 검증 및 조정
        if not np.isfinite(p_value):
            p_value = 1.0
        elif p_value > 1.0:
            p_value = 1.0
        elif p_value < 0.0:
            p_value = 0.0

        # D 통계량 범위 검증
        if not np.isfinite(statistic) or statistic < 0 or statistic > 1:
            statistic = 0.0
            p_value = 1.0

        significant = p_value < 0.05

        # 분포 차이 정도 결정
        if statistic > 0.1:
            distribution_diff = "large"
        elif statistic > 0.05:
            distribution_diff = "medium"
        else:
            distribution_diff = "small"

        interpretation = (
            f"분포에 유의한 차이 (D={statistic:.4f}, p={p_value:.4f})"
            if significant
            else f"분포 차이 미미 (D={statistic:.4f}, p={p_value:.4f})"
        )

        return StatisticalTestResult(
            test_name="Kolmogorov-Smirnov",
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            interpretation=interpretation
        )

    def get_service_info(self) -> Dict[str, Any]:
        """
        서비스 정보 반환

        Returns:
            Dict[str, Any]: 서비스 정보
        """
        return {
            "service_name": "StatisticalTestsService",
            "version": "1.0.0",
            "supported_tests": [
                "mann_whitney_u_test",
                "kolmogorov_smirnov_test"
            ],
            "dependencies": {
                "numpy": np.__version__,
                "scipy": stats.__version__ if hasattr(stats, '__version__') else "unknown"
            },
            "capabilities": {
                "vectorized_computation": True,
                "numerical_stability": True,
                "error_handling": True,
                "logging": True
            }
        }


# 서비스 인스턴스 생성
statistical_tests_service = StatisticalTestsService()
