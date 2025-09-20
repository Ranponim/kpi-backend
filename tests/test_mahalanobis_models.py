"""
마할라노비스 거리 분석 모델 테스트

이 모듈은 마할라노비스 거리 분석 관련 Pydantic 모델들의 유효성을 테스트합니다.
"""

import pytest
from app.models.mahalanobis import (
    KpiDataInput,
    AnalysisOptionsInput,
    AbnormalKpiDetail,
    ScreeningAnalysis,
    DrilldownAnalysis,
    AnalysisResult,
    MahalanobisAnalysisResult,
    MannWhitneyTestInput,
    KolmogorovSmirnovTestInput,
    StatisticalTestResult,
    StatisticalTestResponse,
    validate_kpi_data,
    log_analysis_request
)
from datetime import datetime


class TestKpiDataInput:
    """KpiDataInput 모델 테스트"""

    def test_valid_kpi_data(self):
        """유효한 KPI 데이터로 모델 생성 테스트"""
        kpi_data = {
            "RACH Success Rate": [98.5, 97.8, 99.2, 96.7],
            "RLC DL Throughput": [45.2, 46.1, 44.8, 47.3]
        }

        model = KpiDataInput(kpiData=kpi_data)

        assert len(model.kpi_data) == 2
        assert "RACH Success Rate" in model.kpi_data
        assert len(model.kpi_data["RACH Success Rate"]) == 4

    def test_empty_kpi_data(self):
        """빈 KPI 데이터로 모델 생성 (Pydantic에서는 허용됨)"""
        model = KpiDataInput(kpiData={})
        assert len(model.kpi_data) == 0

    def test_invalid_kpi_values(self):
        """잘못된 타입의 KPI 값으로 모델 생성 시도"""
        kpi_data = {
            "RACH Success Rate": ["invalid", "values"]  # 문자열 대신 숫자 필요
        }

        with pytest.raises(Exception):  # Pydantic ValidationError
            KpiDataInput(kpiData=kpi_data)


class TestAnalysisOptionsInput:
    """AnalysisOptionsInput 모델 테스트"""

    def test_default_values(self):
        """기본값으로 모델 생성 테스트"""
        model = AnalysisOptionsInput()

        assert model.threshold == 0.1
        assert model.sample_size == 20
        assert model.significance_level == 0.05

    def test_custom_values(self):
        """사용자 정의 값으로 모델 생성 테스트"""
        model = AnalysisOptionsInput(
            threshold=0.2,
            sampleSize=30,
            significanceLevel=0.01
        )

        assert model.threshold == 0.2
        assert model.sample_size == 30
        assert model.significance_level == 0.01

    def test_invalid_threshold(self):
        """잘못된 임계값으로 모델 생성 시도"""
        with pytest.raises(Exception):  # Pydantic ValidationError
            AnalysisOptionsInput(threshold=1.5)  # 1.0 초과

    def test_invalid_sample_size(self):
        """잘못된 샘플 크기로 모델 생성 시도"""
        with pytest.raises(Exception):  # Pydantic ValidationError
            AnalysisOptionsInput(sampleSize=0)  # 0 이하


class TestAbnormalKpiDetail:
    """AbnormalKpiDetail 모델 테스트"""

    def test_valid_detail(self):
        """유효한 비정상 KPI 상세 정보로 모델 생성 테스트"""
        detail = AbnormalKpiDetail(
            kpiName="RACH Success Rate",
            n1Value=98.5,
            nValue=94.2,
            changeRate=-0.044,
            severity="warning"
        )

        assert detail.kpi_name == "RACH Success Rate"
        assert detail.n1_value == 98.5
        assert detail.n_value == 94.2
        assert detail.change_rate == -0.044
        assert detail.severity == "warning"

    def test_invalid_severity(self):
        """잘못된 심각도로 모델 생성 시도"""
        with pytest.raises(Exception):  # Pydantic ValidationError
            AbnormalKpiDetail(
                kpiName="Test KPI",
                n1Value=100.0,
                nValue=95.0,
                changeRate=-0.05,
                severity="invalid"  # warning 또는 critical만 허용
            )


class TestAnalysisResult:
    """AnalysisResult 모델 테스트"""

    def test_valid_result(self):
        """유효한 분석 결과로 모델 생성 테스트"""
        abnormal_kpi = AbnormalKpiDetail(
            kpiName="Test KPI",
            n1Value=100.0,
            nValue=95.0,
            changeRate=-0.05,
            severity="warning"
        )

        result = AnalysisResult(
            totalKpis=15,
            abnormalKpis=[abnormal_kpi],
            abnormalScore=0.133,
            alarmLevel="caution",
            analysis={
                "screening": {
                    "status": "caution",
                    "score": 0.133,
                    "threshold": 0.1,
                    "description": "비정상 패턴 감지됨"
                }
            }
        )

        assert result.total_kpis == 15
        assert len(result.abnormal_kpis) == 1
        assert result.abnormal_score == 0.133
        assert result.alarm_level == "caution"
        assert isinstance(result.timestamp, datetime)


class TestStatisticalTestInputs:
    """통계 검정 입력 모델 테스트"""

    def test_mann_whitney_input(self):
        """Mann-Whitney U 검정 입력 모델 테스트"""
        input_data = MannWhitneyTestInput(
            groupA=[98.5, 97.8, 99.2],
            groupB=[94.2, 95.1, 93.8],
            significanceLevel=0.05
        )

        assert len(input_data.group_a) == 3
        assert len(input_data.group_b) == 3
        assert input_data.significance_level == 0.05

    def test_kolmogorov_smirnov_input(self):
        """Kolmogorov-Smirnov 검정 입력 모델 테스트"""
        input_data = KolmogorovSmirnovTestInput(
            groupA=[98.5, 97.8, 99.2],
            groupB=[94.2, 95.1, 93.8],
            significanceLevel=0.01
        )

        assert len(input_data.group_a) == 3
        assert len(input_data.group_b) == 3
        assert input_data.significance_level == 0.01


class TestValidationFunctions:
    """검증 함수 테스트"""

    def test_validate_kpi_data_valid(self):
        """유효한 KPI 데이터 검증 테스트"""
        kpi_data = {
            "RACH Success Rate": [98.5, 97.8, 99.2],
            "RLC DL Throughput": [45.2, 46.1, 44.8]
        }

        assert validate_kpi_data(kpi_data) is True

    def test_validate_kpi_data_empty(self):
        """빈 KPI 데이터 검증 테스트"""
        assert validate_kpi_data({}) is False

    def test_validate_kpi_data_invalid_values(self):
        """잘못된 타입의 KPI 값 검증 테스트"""
        kpi_data = {
            "RACH Success Rate": ["invalid", "values"]
        }

        assert validate_kpi_data(kpi_data) is False

    def test_validate_kpi_data_empty_list(self):
        """빈 리스트의 KPI 값 검증 테스트"""
        kpi_data = {
            "RACH Success Rate": []
        }

        assert validate_kpi_data(kpi_data) is False


if __name__ == "__main__":
    pytest.main([__file__])
