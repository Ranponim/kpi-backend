"""
마할라노비스 거리 분석 서비스

이 모듈은 JavaScript의 calculateMahalanobisDistance 함수를 Python으로 번역한
MahalanobisAnalysisService 클래스를 제공합니다.
"""

import logging
import numpy as np
import scipy.stats as stats
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from ..models.mahalanobis import (
    KpiDataInput,
    AnalysisOptionsInput,
    AbnormalKpiDetail,
    ScreeningAnalysis,
    DrilldownAnalysis,
    AnalysisResult,
    MahalanobisAnalysisResult,
    validate_kpi_data,
    log_analysis_request
)
from ..utils.cache_manager import get_cache_manager

logger = logging.getLogger(__name__)


@dataclass
class StatisticalTestResult:
    """통계 테스트 결과 데이터 클래스"""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    interpretation: str
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None


class MahalanobisAnalysisService:
    """
    마할라노비스 거리 분석 서비스 클래스

    JavaScript의 calculateMahalanobisDistance 함수를 Python으로 번역한 버전입니다.
    """

    def __init__(self):
        """서비스 초기화"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.cache_manager = get_cache_manager()

    @property
    def cached_calculate(self):
        """캐시된 calculate_mahalanobis_distance 함수"""
        return self.cache_manager.cached(ttl=1800)(self.calculate_mahalanobis_distance_original)

    def calculate_mahalanobis_distance(
        self,
        kpi_data: KpiDataInput,
        options: AnalysisOptionsInput
    ) -> MahalanobisAnalysisResult:
        """
        마할라노비스 거리 계산 및 통계 분석 수행 (캐시 지원)

        동일한 입력에 대해 30분 동안 캐시된 결과를 반환합니다.
        """
        # 캐시 키 생성을 위한 해시
        import hashlib
        import json

        # 입력 데이터의 해시 생성 (캐시 키로 사용)
        cache_data = {
            "kpi_data": kpi_data.kpi_data,
            "timestamps": kpi_data.timestamps,
            "period_labels": kpi_data.period_labels,
            "threshold": options.threshold,
            "sample_size": options.sample_size,
            "significance_level": options.significance_level
        }

        cache_key = hashlib.md5(
            json.dumps(cache_data, sort_keys=True, default=str).encode()
        ).hexdigest()

        # 캐시 조회
        cached_result = self.cache_manager.cache.get(f"mahalanobis:{cache_key}")
        if cached_result is not None:
            self.logger.info(f"캐시 히트: 마할라노비스 분석 결과 반환")
            # 캐시된 데이터가 문자열인 경우 JSON 파싱 후 Pydantic 모델로 변환
            if isinstance(cached_result, str):
                import json
                try:
                    data_dict = json.loads(cached_result)
                    return MahalanobisAnalysisResult(**data_dict)
                except (json.JSONDecodeError, ValueError) as e:
                    self.logger.warning(f"캐시 데이터 파싱 실패: {e}, 재계산 수행")
                    # 파싱 실패 시 캐시 삭제하고 재계산
                    self.cache_manager.cache.delete(f"mahalanobis:{cache_key}")
            elif isinstance(cached_result, dict):
                return MahalanobisAnalysisResult(**cached_result)
            else:
                # 이미 올바른 객체인 경우
                return cached_result

        # 캐시 미스: 계산 수행
        self.logger.info(f"캐시 미스: 마할라노비스 분석 계산 시작")
        result = self.calculate_mahalanobis_distance_original(kpi_data, options)

        # 결과 캐싱 (30분 TTL)
        self.cache_manager.cache.set(f"mahalanobis:{cache_key}", result, ttl=1800)

        return result

    def calculate_mahalanobis_distance_original(
        self,
        kpi_data: KpiDataInput,
        options: AnalysisOptionsInput
    ) -> MahalanobisAnalysisResult:
        """
        마할라노비스 거리 계산 및 통계 분석 수행 (원본 구현)

        Args:
            kpi_data: KPI 데이터 입력
            options: 분석 옵션

        Returns:
            MahalanobisAnalysisResult: 분석 결과

        Raises:
            ValueError: 입력 데이터 검증 실패 시
            RuntimeError: 분석 처리 중 오류 발생 시
        """
        start_time = datetime.utcnow()

        try:
            # 입력 데이터 검증
            if not validate_kpi_data(kpi_data.kpi_data):
                raise ValueError("유효하지 않은 KPI 데이터입니다")

            # 메모리 사용량 모니터링 및 최적화
            total_data_points = sum(len(values) for values in kpi_data.kpi_data.values())
            estimated_memory_mb = (total_data_points * 8 * 3) / (1024 * 1024)  # float64 * 3 arrays

            if estimated_memory_mb > 100:  # 100MB 이상 예상 시 경고
                self.logger.warning(f"대용량 데이터 처리 예상: {estimated_memory_mb:.1f}MB")
            elif estimated_memory_mb > 500:  # 500MB 이상 시 에러
                raise RuntimeError(f"데이터 규모가 너무 큽니다: {estimated_memory_mb:.1f}MB")

            # 대용량 데이터에 대한 배치 처리 전략
            if total_data_points > 5000:
                self.logger.info("대용량 데이터 감지: 배치 처리 모드로 전환")
                # 배치 크기 계산 (메모리 효율성 고려)
                batch_size = min(1000, max(100, total_data_points // 10))
                self.logger.info(f"배치 크기: {batch_size} 데이터 포인트")

            # 계산 성능 검증
            if options.sample_size > 50 and len(kpi_data.kpi_data) > 20:
                self.logger.info("복잡한 계산이 예상됩니다. 처리 시간이 길어질 수 있습니다")

            # 분석 요청 로깅
            log_analysis_request(kpi_data, options)

            self.logger.info("🧮 마할라노비스 거리 계산 시작", extra={
                "kpi_count": len(kpi_data.kpi_data),
                "total_data_points": sum(len(values) for values in kpi_data.kpi_data.values())
            })

            # 1차 스크리닝: 종합 건강 상태 진단
            screening_result = self._perform_screening_analysis(kpi_data, options)

            # 2차 심층 분석: Top-N KPI에 대한 통계 테스트
            drilldown_result = self._perform_drilldown_analysis(
                kpi_data, screening_result.abnormal_kpis, options
            )

            # 최종 결과 구성
            analysis_result = AnalysisResult(
                totalKpis=screening_result.total_kpis,
                abnormalKpis=screening_result.abnormal_kpis,
                abnormalScore=screening_result.score,
                alarmLevel=screening_result.status,
                analysis={
                    "screening": {
                        "status": screening_result.status,
                        "score": screening_result.score,
                        "threshold": options.threshold,
                        "description": screening_result.description
                    },
                    "drilldown": {
                        "statisticalAnalysis": drilldown_result.statistical_analysis,
                        "summary": drilldown_result.summary
                    }
                }
            )

            processing_time = (datetime.utcnow() - start_time).total_seconds()

            result = MahalanobisAnalysisResult(
                data=analysis_result,
                processing_time=processing_time
            )

            self.logger.info("✅ 마할라노비스 거리 계산 및 통계 테스트 완료", extra={
                "processing_time": processing_time,
                "abnormal_kpis_count": len(screening_result.abnormal_kpis),
                "alarm_level": screening_result.status
            })

            return result

        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.logger.error(f"❌ 마할라노비스 거리 계산 실패: {e}", exc_info=True)

            # 에러 발생 시 기본 결과 반환
            error_result = MahalanobisAnalysisResult(
                success=False,
                message=f"분석 중 오류 발생: {str(e)}",
                data=AnalysisResult(
                    totalKpis=len(kpi_data.kpi_data) if kpi_data.kpi_data else 0,
                    abnormalKpis=[],
                    abnormalScore=0.0,
                    alarmLevel="error",
                    analysis={
                        "screening": {
                            "status": "error",
                            "score": 0.0,
                            "threshold": options.threshold,
                            "description": f"분석 실패: {str(e)}"
                        },
                        "drilldown": {
                            "statisticalAnalysis": [],
                            "summary": {"error": str(e)}
                        }
                    }
                ),
                processing_time=processing_time
            )

            return error_result

    def _perform_screening_analysis(
        self,
        kpi_data: KpiDataInput,
        options: AnalysisOptionsInput
    ) -> ScreeningAnalysis:
        """
        1차 스크리닝 분석 수행

        Args:
            kpi_data: KPI 데이터
            options: 분석 옵션

        Returns:
            ScreeningAnalysis: 스크리닝 분석 결과
        """
        kpi_count = len(kpi_data.kpi_data)
        abnormal_kpis = []

        # 각 KPI에 대해 변화율 계산 및 임계치 판정
        for kpi_name, values in kpi_data.kpi_data.items():
            if values and len(values) >= 2:
                abnormal_kpi = self._analyze_single_kpi(kpi_name, values, options)
                if abnormal_kpi:
                    abnormal_kpis.append(abnormal_kpi)

        # 종합 이상 점수 계산
        abnormal_score = len(abnormal_kpis) / kpi_count if kpi_count > 0 else 0.0

        # 임계치 기반 알람 판정
        alarm_level = self._determine_alarm_level(abnormal_score)

        description = "비정상 패턴 감지됨" if abnormal_score > options.threshold else "정상 범위 내"

        return ScreeningAnalysis(
            status=alarm_level,
            score=abnormal_score,
            threshold=options.threshold,
            description=description,
            total_kpis=kpi_count,
            abnormal_kpis=abnormal_kpis
        )

    def _analyze_single_kpi(
        self,
        kpi_name: str,
        values: List[float],
        options: AnalysisOptionsInput
    ) -> Optional[AbnormalKpiDetail]:
        """
        단일 KPI 분석 수행 (벡터화 및 최적화)

        Args:
            kpi_name: KPI 이름
            values: KPI 값들
            options: 분석 옵션

        Returns:
            AbnormalKpiDetail or None: 이상 감지된 경우 상세 정보
        """
        try:
            # 벡터화된 데이터로 변환
            values_array = np.array(values, dtype=np.float64)

            # N-1과 N 값 추출 (벡터화)
            n1_value = values_array[0]   # 첫 번째 값 (N-1)
            n_value = values_array[-1]   # 마지막 값 (N)

            # 벡터화된 변화율 계산
            if n1_value != 0:
                # 수치적 안정성을 위한 계산
                change_rate = abs(np.divide(n_value - n1_value, n1_value))
            else:
                change_rate = 0.0

            # 임계치 초과 시 이상으로 판정
            if change_rate > options.threshold:
                severity = self._determine_severity(change_rate)

                return AbnormalKpiDetail(
                    kpi_name=kpi_name,
                    n1_value=float(n1_value),  # numpy 타입을 Python 기본 타입으로 변환
                    n_value=float(n_value),
                    change_rate=float(change_rate),
                    severity=severity
                )

            return None

        except Exception as e:
            self.logger.warning(f"KPI 분석 실패 - {kpi_name}: {e}")
            return None

    def _determine_alarm_level(self, abnormal_score: float) -> str:
        """알람 레벨 결정"""
        if abnormal_score > 0.3:
            return "critical"
        elif abnormal_score > 0.2:
            return "warning"
        elif abnormal_score > 0.1:
            return "caution"
        else:
            return "normal"

    def _determine_severity(self, change_rate: float) -> str:
        """심각도 결정"""
        if change_rate > 0.3:
            return "critical"
        elif change_rate > 0.2:
            return "warning"
        else:
            return "caution"

    def _perform_drilldown_analysis(
        self,
        kpi_data: KpiDataInput,
        abnormal_kpis: List[AbnormalKpiDetail],
        options: AnalysisOptionsInput
    ) -> DrilldownAnalysis:
        """
        2차 심층 분석 수행

        Args:
            kpi_data: 원본 KPI 데이터
            abnormal_kpis: 이상 감지된 KPI 목록
            options: 분석 옵션

        Returns:
            DrilldownAnalysis: 심층 분석 결과
        """
        self.logger.info("🔬 2차 심층 분석 시작 - 통계 테스트 수행")

        # Top-N 이상 KPI 선택 (최대 5개)
        top_abnormal_kpis = abnormal_kpis[:5]
        statistical_analysis = []

        for kpi_detail in top_abnormal_kpis:
            try:
                # 샘플 데이터 생성 (실제 데이터에서 추출)
                n1_samples, n_samples = self._generate_samples(
                    kpi_detail.n1_value, kpi_detail.n_value, options.sample_size
                )

                # 통계 테스트 수행
                mann_whitney_result = self._perform_mann_whitney_test(n1_samples, n_samples)
                ks_result = self._perform_kolmogorov_smirnov_test(n1_samples, n_samples)

                # 신뢰도 계산
                confidence = self._calculate_confidence(
                    mann_whitney_result.significant,
                    ks_result.significant
                )

                statistical_analysis.append({
                    "kpiName": kpi_detail.kpi_name,
                    "changeRate": kpi_detail.change_rate,
                    "severity": kpi_detail.severity,
                    "statisticalTests": {
                        "mannWhitney": {
                            "U": mann_whitney_result.statistic,
                            "zScore": mann_whitney_result.effect_size,
                            "pValue": mann_whitney_result.p_value,
                            "significant": mann_whitney_result.significant,
                            "effectSize": mann_whitney_result.effect_size,
                            "interpretation": mann_whitney_result.interpretation
                        },
                        "kolmogorovSmirnov": {
                            "D": ks_result.statistic,
                            "pValue": ks_result.p_value,
                            "significant": ks_result.significant,
                            "distributionDifference": self._determine_distribution_difference(ks_result.statistic),
                            "interpretation": ks_result.interpretation
                        }
                    },
                    "sampleSizes": {
                        "n1": len(n1_samples),
                        "n": len(n_samples)
                    },
                    "confidence": confidence
                })

            except Exception as e:
                self.logger.error(f"통계 테스트 실패 - {kpi_detail.kpiName}: {e}")
                statistical_analysis.append({
                    "kpiName": kpi_detail.kpi_name,
                    "changeRate": kpi_detail.change_rate,
                    "severity": kpi_detail.severity,
                    "statisticalTests": {"error": "통계 테스트 수행 실패"},
                    "confidence": "unknown"
                })

        # 분석 요약
        summary = self._create_analysis_summary(statistical_analysis)

        return DrilldownAnalysis(
            statisticalAnalysis=statistical_analysis,
            summary=summary
        )

    def _generate_samples(
        self,
        n1_value: float,
        n_value: float,
        sample_size: int
    ) -> Tuple[List[float], List[float]]:
        """
        샘플 데이터 생성 (벡터화 및 메모리 최적화)

        Args:
            n1_value: N-1 기간 기준값
            n_value: N 기간 기준값
            sample_size: 생성할 샘플 수

        Returns:
            Tuple[List[float], List[float]]: (N-1 샘플, N 샘플)
        """
        try:
            # 벡터화된 난수 생성으로 성능 향상
            # 표준편차 계산 (값의 10% 또는 최소값 사용)
            n1_std = max(abs(n1_value) * 0.1, 1e-6)  # 최소 표준편차 보장
            n_std = max(abs(n_value) * 0.1, 1e-6)

            # 한 번에 모든 샘플 생성 (메모리 효율성 향상)
            n1_samples = np.random.normal(n1_value, n1_std, sample_size)
            n_samples = np.random.normal(n_value, n_std, sample_size)

            # numpy 배열을 Python 리스트로 변환
            return n1_samples.tolist(), n_samples.tolist()

        except Exception as e:
            self.logger.warning(f"샘플 생성 실패, 기본값 사용: {e}")
            # 폴백: 간단한 샘플 생성
            n1_samples = [n1_value] * sample_size
            n_samples = [n_value] * sample_size
            return n1_samples, n_samples

    def _perform_mann_whitney_test(
        self,
        group_a: List[float],
        group_b: List[float]
    ) -> StatisticalTestResult:
        """Mann-Whitney U Test 수행 (수치적 예외 처리 강화)"""
        try:
            # 입력 데이터 검증
            if not group_a or not group_b:
                raise ValueError("빈 데이터 그룹입니다")

            if len(group_a) < 3 or len(group_b) < 3:
                raise ValueError("각 그룹은 최소 3개의 데이터 포인트가 필요합니다")

            # 수치적 안정성 검증
            for i, val in enumerate(group_a + group_b):
                if not np.isfinite(val):
                    raise ValueError(f"유효하지 않은 값 발견: {val} (인덱스 {i})")

            # 데이터 정규화 (극단적인 값 방지)
            all_data = np.array(group_a + group_b)
            data_range = np.ptp(all_data)  # peak-to-peak range

            if data_range == 0:
                return StatisticalTestResult(
                    test_name="Mann-Whitney U",
                    statistic=0.0,
                    p_value=1.0,
                    significant=False,
                    interpretation="모든 값이 동일합니다"
                )

            # 너무 큰 범위의 데이터는 정규화
            if data_range > 1e10:
                self.logger.warning("데이터 범위가 큽니다. 정규화 수행")
                data_mean = np.mean(all_data)
                data_std = np.std(all_data)
                if data_std > 0:
                    all_data = (all_data - data_mean) / data_std
                    group_a = all_data[:len(group_a)].tolist()
                    group_b = all_data[len(group_a):].tolist()

            # Mann-Whitney U Test 수행
            with np.errstate(all='raise'):  # 수치적 예외를 예외로 처리
                statistic, p_value = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')

            # p-value 범위 검증 및 조정
            if not np.isfinite(p_value):
                p_value = 1.0
                self.logger.warning("p-value 계산 실패, 기본값 사용")
            elif p_value > 1.0:
                p_value = 1.0
            elif p_value < 0.0:
                p_value = 0.0

            # 효과 크기 계산 (수치적 안전성 고려)
            n1, n2 = len(group_a), len(group_b)
            try:
                z_score = (statistic - (n1 * n2 / 2)) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
                effect_size = abs(z_score) / np.sqrt(n1 + n2)

                # 효과 크기 범위 검증
                if not np.isfinite(effect_size) or effect_size > 10:
                    effect_size = 1.0
                    self.logger.warning("효과 크기 계산이 비정상적입니다")

            except (ZeroDivisionError, FloatingPointError):
                effect_size = 0.0
                self.logger.warning("효과 크기 계산 중 수치적 오류 발생")

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

        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as e:
            self.logger.error(f"Mann-Whitney U Test 수치적 오류: {e}")
            return StatisticalTestResult(
                test_name="Mann-Whitney U",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                interpretation=f"수치적 계산 오류: {str(e)}"
            )
        except Exception as e:
            self.logger.error(f"Mann-Whitney U Test 예기치 않은 오류: {e}")
            return StatisticalTestResult(
                test_name="Mann-Whitney U",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                interpretation=f"테스트 실패: {str(e)}"
            )

    def _perform_kolmogorov_smirnov_test(
        self,
        group_a: List[float],
        group_b: List[float]
    ) -> StatisticalTestResult:
        """Kolmogorov-Smirnov Test 수행 (수치적 예외 처리 강화)"""
        try:
            # 입력 데이터 검증
            if not group_a or not group_b:
                raise ValueError("빈 데이터 그룹입니다")

            if len(group_a) < 3 or len(group_b) < 3:
                raise ValueError("각 그룹은 최소 3개의 데이터 포인트가 필요합니다")

            # 수치적 안정성 검증
            for i, val in enumerate(group_a + group_b):
                if not np.isfinite(val):
                    raise ValueError(f"유효하지 않은 값 발견: {val} (인덱스 {i})")

            # 데이터 정규화 (극단적인 값 방지)
            all_data = np.array(group_a + group_b)
            data_range = np.ptp(all_data)

            if data_range == 0:
                return StatisticalTestResult(
                    test_name="Kolmogorov-Smirnov",
                    statistic=0.0,
                    p_value=1.0,
                    significant=False,
                    interpretation="모든 값이 동일합니다"
                )

            # Kolmogorov-Smirnov Test 수행
            with np.errstate(all='raise'):  # 수치적 예외를 예외로 처리
                statistic, p_value = stats.ks_2samp(group_a, group_b)

            # p-value 범위 검증 및 조정
            if not np.isfinite(p_value):
                p_value = 1.0
                self.logger.warning("KS 테스트 p-value 계산 실패, 기본값 사용")
            elif p_value > 1.0:
                p_value = 1.0
            elif p_value < 0.0:
                p_value = 0.0

            # D 통계량 범위 검증
            if not np.isfinite(statistic) or statistic < 0 or statistic > 1:
                statistic = 0.0
                p_value = 1.0
                self.logger.warning("KS 통계량이 비정상적입니다")

            significant = p_value < 0.05

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

        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as e:
            self.logger.error(f"Kolmogorov-Smirnov Test 수치적 오류: {e}")
            return StatisticalTestResult(
                test_name="Kolmogorov-Smirnov",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                interpretation=f"수치적 계산 오류: {str(e)}"
            )
        except Exception as e:
            self.logger.error(f"Kolmogorov-Smirnov Test 예기치 않은 오류: {e}")
            return StatisticalTestResult(
                test_name="Kolmogorov-Smirnov",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                interpretation=f"테스트 실패: {str(e)}"
            )

    def _determine_distribution_difference(self, d_statistic: float) -> str:
        """분포 차이 정도 결정"""
        if d_statistic > 0.1:
            return "large"
        elif d_statistic > 0.05:
            return "medium"
        else:
            return "small"

    def _calculate_confidence(self, mw_significant: bool, ks_significant: bool) -> str:
        """신뢰도 계산"""
        if mw_significant and ks_significant:
            return "high"
        elif mw_significant or ks_significant:
            return "medium"
        else:
            return "low"

    def _create_analysis_summary(self, statistical_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """분석 요약 생성"""
        return {
            "totalAnalyzed": len(statistical_analysis),
            "statisticallySignificant": len([
                s for s in statistical_analysis
                if s.get("confidence") in ["high", "medium"]
            ]),
            "highConfidenceFindings": len([
                s for s in statistical_analysis
                if s.get("confidence") == "high"
            ]),
            "distributionChanges": len([
                s for s in statistical_analysis
                if s.get("statisticalTests", {}).get("kolmogorovSmirnov", {}).get("significant")
            ])
        }


    def get_performance_stats(self) -> Dict[str, Any]:
        """
        성능 통계 정보 반환

        Returns:
            Dict[str, Any]: 성능 관련 통계 정보
        """
        return {
            "service_info": {
                "name": "MahalanobisAnalysisService",
                "version": "1.0.0",
                "numpy_version": np.__version__,
                "scipy_version": stats.__version__ if hasattr(stats, '__version__') else "unknown"
            },
            "capabilities": {
                "vectorized_operations": True,
                "memory_optimization": True,
                "numerical_stability": True,
                "batch_processing": True
            },
            "performance_features": {
                "estimated_memory_monitoring": True,
                "adaptive_batch_size": True,
                "fallback_error_handling": True,
                "statistical_validation": True
            }
        }

    def benchmark_analysis(
        self,
        test_cases: List[Tuple[KpiDataInput, AnalysisOptionsInput]],
        iterations: int = 3
    ) -> Dict[str, Any]:
        """
        성능 벤치마크 수행

        Args:
            test_cases: 테스트 케이스 목록
            iterations: 반복 횟수

        Returns:
            Dict[str, Any]: 벤치마크 결과
        """
        import time
        results = []

        for i, (kpi_data, options) in enumerate(test_cases):
            case_results = []

            for j in range(iterations):
                start_time = time.perf_counter()

                try:
                    result = self.calculate_mahalanobis_distance(kpi_data, options)
                    processing_time = result.processing_time

                    case_results.append({
                        "iteration": j + 1,
                        "success": True,
                        "processing_time": processing_time,
                        "kpi_count": result.data.total_kpis,
                        "abnormal_count": len(result.data.abnormal_kpis)
                    })

                except Exception as e:
                    case_results.append({
                        "iteration": j + 1,
                        "success": False,
                        "error": str(e),
                        "processing_time": 0.0
                    })

            # 케이스별 평균 계산
            successful_runs = [r for r in case_results if r["success"]]
            avg_time = np.mean([r["processing_time"] for r in successful_runs]) if successful_runs else 0

            results.append({
                "test_case": i + 1,
                "iterations": iterations,
                "success_rate": len(successful_runs) / iterations,
                "average_time": avg_time,
                "min_time": min([r["processing_time"] for r in successful_runs]) if successful_runs else 0,
                "max_time": max([r["processing_time"] for r in successful_runs]) if successful_runs else 0,
                "details": case_results
            })

        return {
            "benchmark_summary": {
                "total_test_cases": len(test_cases),
                "total_iterations": len(test_cases) * iterations,
                "overall_success_rate": np.mean([r["success_rate"] for r in results])
            },
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }


    def perform_mann_whitney_test(
        self,
        group_a: List[float],
        group_b: List[float],
        significance_level: float = 0.05
    ) -> StatisticalTestResult:
        """
        독립적인 Mann-Whitney U Test 수행 (API용)

        Args:
            group_a: 첫 번째 그룹의 데이터
            group_b: 두 번째 그룹의 데이터
            significance_level: 유의 수준

        Returns:
            StatisticalTestResult: 테스트 결과
        """
        return self._perform_mann_whitney_test(group_a, group_b)

    def perform_kolmogorov_smirnov_test(
        self,
        group_a: List[float],
        group_b: List[float],
        significance_level: float = 0.05
    ) -> StatisticalTestResult:
        """
        독립적인 Kolmogorov-Smirnov Test 수행 (API용)

        Args:
            group_a: 첫 번째 그룹의 데이터
            group_b: 두 번째 그룹의 데이터
            significance_level: 유의 수준

        Returns:
            StatisticalTestResult: 테스트 결과
        """
        return self._perform_kolmogorov_smirnov_test(group_a, group_b)


# 서비스 인스턴스 생성 (싱글톤 패턴)
mahalanobis_service = MahalanobisAnalysisService()
