"""
마할라노비스 거리 분석 모델 정의

이 모듈은 마할라노비스 거리 분석을 위한 Pydantic 모델들을 정의합니다.
"""

import logging
import numpy as np
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator, ValidationError
from typing import List, Optional, Dict, Any
from datetime import datetime

# 로깅 설정
logger = logging.getLogger(__name__)


class KpiDataInput(BaseModel):
    """
    KPI 데이터 입력 모델

    프론트엔드에서 전송되는 KPI 데이터를 검증하고 구조화합니다.
    """
    kpi_data: Dict[str, List[float]] = Field(
        ...,
        description="KPI 이름과 값들의 매핑 (예: {'kpi_name_1': [1.0, 2.0, 3.0], ...})",
        alias="kpiData"
    )
    timestamps: Optional[List[str]] = Field(
        None,
        description="데이터 포인트들의 타임스탬프 배열 (ISO 8601 형식, 예: ['2024-01-01T10:00:00Z', ...])",
        alias="timestamps"
    )
    period_labels: Optional[List[str]] = Field(
        None,
        description="기간 레이블 (예: ['N-2', 'N-1', 'N'] 또는 ['2024-01-01', '2024-01-02'])",
        alias="periodLabels"
    )

    @field_validator('kpi_data')
    @classmethod
    def validate_kpi_data_structure(cls, v: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """KPI 데이터 구조 검증"""
        if not v:
            raise ValueError("KPI 데이터가 비어있습니다")

        if len(v) == 0:
            raise ValueError("적어도 하나의 KPI가 있어야 합니다")

        if len(v) > 100:
            raise ValueError("KPI 수가 너무 많습니다 (최대 100개)")

        # 각 KPI별 검증
        for kpi_name, values in v.items():
            if not kpi_name or not kpi_name.strip():
                raise ValueError(f"KPI 이름이 비어있습니다")

            if not isinstance(values, list):
                raise ValueError(f"KPI '{kpi_name}'의 값이 리스트가 아닙니다")

            if len(values) < 2:
                raise ValueError(f"KPI '{kpi_name}'의 데이터 포인트가 부족합니다 (최소 2개 필요)")

            if len(values) > 1000:
                raise ValueError(f"KPI '{kpi_name}'의 데이터 포인트가 너무 많습니다 (최대 1000개)")

            # 수치 값 검증
            for i, value in enumerate(values):
                if not isinstance(value, (int, float)):
                    raise ValueError(f"KPI '{kpi_name}'의 {i+1}번째 값이 숫자가 아닙니다")

                if not np.isfinite(value):
                    raise ValueError(f"KPI '{kpi_name}'의 {i+1}번째 값이 유효하지 않은 숫자입니다")

                # 비합리적인 값 범위 검증 (필요에 따라 조정)
                if abs(value) > 1e10:
                    raise ValueError(f"KPI '{kpi_name}'의 {i+1}번째 값이 너무 큽니다")

        return v

    @field_validator('timestamps')
    @classmethod
    def validate_timestamps(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """타임스탬프 검증"""
        if v is None:
            return v

        if not isinstance(v, list):
            raise ValueError("타임스탬프가 리스트가 아닙니다")

        for i, timestamp in enumerate(v):
            if not isinstance(timestamp, str):
                raise ValueError(f"{i+1}번째 타임스탬프가 문자열이 아닙니다")

            # ISO 8601 형식 검증 (간단한 검증)
            if 'T' not in timestamp:
                raise ValueError(f"{i+1}번째 타임스탬프가 ISO 8601 형식이 아닙니다")

        return v

    @field_validator('period_labels')
    @classmethod
    def validate_period_labels(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """기간 레이블 검증"""
        if v is None:
            return v

        if not isinstance(v, list):
            raise ValueError("기간 레이블이 리스트가 아닙니다")

        for i, label in enumerate(v):
            if not isinstance(label, str):
                raise ValueError(f"{i+1}번째 기간 레이블이 문자열이 아닙니다")

            if not label.strip():
                raise ValueError(f"{i+1}번째 기간 레이블이 비어있습니다")

        return v

    @model_validator(mode='after')
    def validate_data_consistency(self) -> 'KpiDataInput':
        """데이터 일관성 검증"""
        if not self.kpi_data:
            return self

        # 모든 KPI의 데이터 포인트 수가 동일한지 검증
        data_lengths = [len(values) for values in self.kpi_data.values()]
        if len(set(data_lengths)) > 1:
            raise ValueError("모든 KPI의 데이터 포인트 수가 동일해야 합니다")

        data_length = data_lengths[0]

        # 타임스탬프 일관성 검증
        if self.timestamps is not None:
            if len(self.timestamps) != data_length:
                raise ValueError("타임스탬프 수와 데이터 포인트 수가 일치해야 합니다")

        # 기간 레이블 일관성 검증
        if self.period_labels is not None:
            if len(self.period_labels) != data_length:
                raise ValueError("기간 레이블 수와 데이터 포인트 수가 일치해야 합니다")

        # 데이터 품질 검증
        total_data_points = sum(len(values) for values in self.kpi_data.values())
        if total_data_points > 5000:
            logger.warning(f"많은 양의 데이터 포인트가 처리됩니다: {total_data_points}")

        # 이상치 검증 (선택적)
        for kpi_name, values in self.kpi_data.items():
            if len(values) >= 3:  # 최소 3개 이상일 때만 검증
                mean_val = np.mean(values)
                std_val = np.std(values)

                if std_val == 0:
                    logger.warning(f"KPI '{kpi_name}'의 모든 값이 동일합니다")

                # 3시그마 범위 밖의 값들 카운트
                outliers = sum(1 for v in values if abs(v - mean_val) > 3 * std_val)
                if outliers > 0:
                    logger.info(f"KPI '{kpi_name}'에서 {outliers}개의 잠재적 이상치 감지됨")

        return self

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "kpiData": {
                    "RACH Success Rate": [98.5, 97.8, 99.2, 96.7, 98.1],
                    "RLC DL Throughput": [45.2, 46.1, 44.8, 47.3, 45.9],
                    "RLC UL Throughput": [12.3, 13.1, 11.8, 12.9, 12.7]
                },
                "timestamps": [
                    "2024-01-01T10:00:00Z",
                    "2024-01-01T11:00:00Z",
                    "2024-01-01T12:00:00Z",
                    "2024-01-01T13:00:00Z",
                    "2024-01-01T14:00:00Z"
                ],
                "periodLabels": ["N-2", "N-1", "N"]
            }
        }
    )


class AnalysisOptionsInput(BaseModel):
    """
    분석 옵션 입력 모델

    마할라노비스 분석에 필요한 설정값들을 검증합니다.
    """
    threshold: float = Field(
        default=0.1,
        description="비정상 감지 임계값 (0.0 ~ 1.0)",
        ge=0.0,
        le=1.0
    )
    sample_size: int = Field(
        default=20,
        description="샘플 크기 (데이터 포인트 수)",
        gt=0,
        alias="sampleSize"
    )
    significance_level: float = Field(
        default=0.05,
        description="유의 수준 (Mann-Whitney U 검정용)",
        ge=0.0,
        le=1.0,
        alias="significanceLevel"
    )

    @field_validator('threshold')
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """임계값 검증"""
        if not np.isfinite(v):
            raise ValueError("임계값이 유효하지 않은 숫자입니다")

        if v <= 0:
            raise ValueError("임계값은 0보다 커야 합니다")

        if v >= 1:
            raise ValueError("임계값은 1보다 작아야 합니다")

        if v < 0.001:
            logger.warning("임계값이 매우 낮습니다. 많은 거짓 긍정이 발생할 수 있습니다")

        if v > 0.5:
            logger.warning("임계값이 매우 높습니다. 실제 이상을 놓칠 수 있습니다")

        return v

    @field_validator('sample_size')
    @classmethod
    def validate_sample_size(cls, v: int) -> int:
        """샘플 크기 검증"""
        if not isinstance(v, int):
            raise ValueError("샘플 크기는 정수여야 합니다")

        if v < 5:
            raise ValueError("샘플 크기는 최소 5여야 합니다")

        if v > 1000:
            raise ValueError("샘플 크기는 최대 1000을 초과할 수 없습니다")

        if v < 10:
            logger.warning("샘플 크기가 작습니다. 통계적 검정력이 낮아질 수 있습니다")

        if v > 100:
            logger.info("샘플 크기가 큽니다. 처리 시간이 증가할 수 있습니다")

        return v

    @field_validator('significance_level')
    @classmethod
    def validate_significance_level(cls, v: float) -> float:
        """유의 수준 검증"""
        if not np.isfinite(v):
            raise ValueError("유의 수준이 유효하지 않은 숫자입니다")

        if v <= 0:
            raise ValueError("유의 수준은 0보다 커야 합니다")

        if v >= 0.5:
            raise ValueError("유의 수준은 0.5보다 작아야 합니다")

        if v < 0.001:
            logger.warning("유의 수준이 매우 엄격합니다. 귀무가설을 기각하기 어려울 수 있습니다")

        if v > 0.1:
            logger.warning("유의 수준이 관대한 편입니다. 거짓 긍정이 증가할 수 있습니다")

        return v

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "threshold": 0.1,
                "sampleSize": 20,
                "significanceLevel": 0.05
            }
        }
    )


class AbnormalKpiDetail(BaseModel):
    """
    비정상 KPI 상세 정보 모델

    마할라노비스 거리 분석에서 감지된 비정상 KPI의 상세 정보를 포함합니다.
    """
    kpi_name: str = Field(..., description="KPI 이름", alias="kpiName")
    n1_value: float = Field(..., description="N-1 기간 평균값", alias="n1Value")
    n_value: float = Field(..., description="N 기간 평균값", alias="nValue")
    change_rate: float = Field(..., description="변화율", alias="changeRate")
    severity: str = Field(
        default="warning",
        description="심각도 (caution, warning, critical)",
        pattern="^(caution|warning|critical)$"
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "kpiName": "RACH Success Rate",
                "n1Value": 98.5,
                "nValue": 94.2,
                "changeRate": -0.044,
                "severity": "warning"
            }
        }
    )


class ScreeningAnalysis(BaseModel):
    """
    스크리닝 분석 결과 모델

    마할라노비스 거리 분석의 스크리닝 단계 결과를 포함합니다.
    """
    status: str = Field(
        default="normal",
        description="분석 상태 (normal, caution, warning, critical, error)",
        pattern="^(normal|caution|warning|critical|error)$"
    )
    score: float = Field(..., description="비정상 점수 (0.0 ~ 1.0)", ge=0.0, le=1.0)
    threshold: float = Field(..., description="임계값", ge=0.0, le=1.0)
    description: str = Field(..., description="분석 설명")
    total_kpis: int = Field(default=0, description="총 KPI 수")
    abnormal_kpis: List[AbnormalKpiDetail] = Field(
        default_factory=list,
        description="비정상 KPI 목록"
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "status": "caution",
                "score": 0.133,
                "threshold": 0.1,
                "description": "비정상 패턴 감지됨",
                "totalKpis": 15,
                "abnormalKpis": []
            }
        }
    )


class DrilldownAnalysis(BaseModel):
    """
    드릴다운 분석 결과 모델

    마할라노비스 거리 분석의 세부 분석 결과를 포함합니다.
    """
    statistical_analysis: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="통계 분석 결과 목록",
        alias="statisticalAnalysis"
    )
    summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="분석 요약 정보"
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "statisticalAnalysis": [
                    {
                        "test": "Mann-Whitney U",
                        "p_value": 0.023,
                        "significant": True,
                        "description": "두 그룹 간 유의미한 차이가 있음"
                    }
                ],
                "summary": {
                    "total_kpis": 15,
                    "abnormal_kpis": 3,
                    "analysis_duration": 1.2
                }
            }
        }
    )


class AnalysisResult(BaseModel):
    """
    마할라노비스 분석 결과 모델

    전체 분석 결과를 구조화하여 반환합니다.
    """
    total_kpis: int = Field(..., description="총 KPI 수", alias="totalKpis")
    abnormal_kpis: List[AbnormalKpiDetail] = Field(
        default_factory=list,
        description="비정상 KPI 목록",
        alias="abnormalKpis"
    )
    abnormal_score: float = Field(..., description="비정상 점수", alias="abnormalScore")
    alarm_level: str = Field(
        default="normal",
        description="알람 레벨",
        alias="alarmLevel",
        pattern="^(normal|caution|warning|critical|error)$"
    )
    analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="분석 세부 결과"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="분석 시간"
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={
            datetime: lambda v: v.isoformat()
        },
        json_schema_extra={
            "example": {
                "totalKpis": 15,
                "abnormalKpis": [
                    {
                        "kpiName": "RACH Success Rate",
                        "n1Value": 98.5,
                        "nValue": 94.2,
                        "changeRate": -0.044,
                        "severity": "warning"
                    }
                ],
                "abnormalScore": 0.133,
                "alarmLevel": "caution",
                "analysis": {
                    "screening": {
                        "status": "caution",
                        "score": 0.133,
                        "threshold": 0.1,
                        "description": "비정상 패턴 감지됨"
                    },
                    "drilldown": {
                        "statisticalAnalysis": [],
                        "summary": {}
                    }
                },
                "timestamp": "2025-09-15T13:30:00.000Z"
            }
        }
    )


class MahalanobisAnalysisResult(BaseModel):
    """
    마할라노비스 분석 최종 결과 모델

    API 응답으로 사용되는 최종 결과 모델입니다.
    """
    success: bool = Field(default=True, description="요청 성공 여부")
    message: str = Field(default="Analysis completed successfully", description="응답 메시지")
    data: AnalysisResult = Field(..., description="분석 결과 데이터")
    processing_time: Optional[float] = Field(None, description="처리 시간 (초)")

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Analysis completed successfully",
                "data": {
                    "totalKpis": 15,
                    "abnormalKpis": [],
                    "abnormalScore": 0.0,
                    "alarmLevel": "normal",
                    "analysis": {},
                    "timestamp": "2025-09-15T13:30:00.000Z"
                },
                "processingTime": 1.2
            }
        }
    )


class MannWhitneyTestInput(BaseModel):
    """
    Mann-Whitney U 검정 입력 모델

    두 그룹의 데이터를 비교하기 위한 입력 모델입니다.
    """
    group_a: List[float] = Field(..., description="A 그룹 데이터", alias="groupA")
    group_b: List[float] = Field(..., description="B 그룹 데이터", alias="groupB")
    significance_level: float = Field(
        default=0.05,
        description="유의 수준",
        ge=0.0,
        le=1.0
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "groupA": [98.5, 97.8, 99.2],
                "groupB": [94.2, 95.1, 93.8],
                "significanceLevel": 0.05
            }
        }
    )


class KolmogorovSmirnovTestInput(BaseModel):
    """
    Kolmogorov-Smirnov 검정 입력 모델

    두 그룹의 분포를 비교하기 위한 입력 모델입니다.
    """
    group_a: List[float] = Field(..., description="A 그룹 데이터", alias="groupA")
    group_b: List[float] = Field(..., description="B 그룹 데이터", alias="groupB")
    significance_level: float = Field(
        default=0.05,
        description="유의 수준",
        ge=0.0,
        le=1.0,
        alias="significanceLevel"
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "groupA": [98.5, 97.8, 99.2],
                "groupB": [94.2, 95.1, 93.8],
                "significanceLevel": 0.05
            }
        }
    )


class StatisticalTestResult(BaseModel):
    """
    통계 검정 결과 모델

    Mann-Whitney U 검정 또는 Kolmogorov-Smirnov 검정 결과를 포함합니다.
    """
    test_name: str = Field(..., description="검정 이름", alias="testName")
    statistic: float = Field(..., description="검정 통계량")
    p_value: float = Field(..., description="p-값", alias="pValue")
    significant: bool = Field(..., description="유의미한 차이 여부")
    interpretation: str = Field(..., description="결과 해석", alias="interpretation")
    description: Optional[str] = Field(None, description="결과 설명")
    effect_size: Optional[float] = Field(None, description="효과 크기")

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "testName": "Mann-Whitney U",
                "statistic": 15.5,
                "pValue": 0.023,
                "significant": True,
                "description": "두 그룹 간 유의미한 차이가 있음"
            }
        }
    )


class StatisticalTestResponse(BaseModel):
    """
    통계 검정 응답 모델

    API 응답으로 사용되는 통계 검정 결과 모델입니다.
    """
    success: bool = Field(default=True, description="요청 성공 여부")
    message: str = Field(default="Statistical test completed", description="응답 메시지")
    data: StatisticalTestResult = Field(..., description="검정 결과 데이터")
    processing_time: Optional[float] = Field(None, description="처리 시간 (초)")

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Mann-Whitney U test completed",
                "data": {
                    "testName": "Mann-Whitney U",
                    "statistic": 15.5,
                    "pValue": 0.023,
                    "significant": True,
                    "description": "두 그룹 간 유의미한 차이가 있음"
                },
                "processingTime": 0.5
            }
        }
    )


# 검증 및 로깅을 위한 헬퍼 함수들
def validate_kpi_data(kpi_data: Dict[str, List[float]]) -> bool:
    """
    KPI 데이터의 유효성을 검증합니다.

    Args:
        kpi_data: 검증할 KPI 데이터

    Returns:
        bool: 유효성 검증 결과
    """
    if not kpi_data:
        logger.warning("KPI 데이터가 비어 있습니다")
        return False

    for kpi_name, values in kpi_data.items():
        if not isinstance(values, list) or len(values) == 0:
            logger.warning(f"KPI '{kpi_name}'의 데이터가 유효하지 않습니다")
            return False

        if not all(isinstance(v, (int, float)) for v in values):
            logger.warning(f"KPI '{kpi_name}'에 숫자가 아닌 값이 포함되어 있습니다")
            return False

    logger.info(f"KPI 데이터 검증 완료: {len(kpi_data)}개 KPI")
    return True


def log_analysis_request(input_data: KpiDataInput, options: AnalysisOptionsInput):
    """
    분석 요청 정보를 로깅합니다.

    Args:
        input_data: KPI 입력 데이터
        options: 분석 옵션
    """
    kpi_count = len(input_data.kpi_data)
    total_points = sum(len(values) for values in input_data.kpi_data.values())

    logger.info("마할라노비스 분석 요청", extra={
        "kpi_count": kpi_count,
        "total_data_points": total_points,
        "threshold": options.threshold,
        "sample_size": options.sample_size,
        "significance_level": options.significance_level
    })


class StatisticalTestInput(BaseModel):
    """
    통계 테스트 입력 모델

    독립적인 통계 테스트를 위한 입력 데이터 모델
    """
    group_a: List[float] = Field(
        ...,
        description="첫 번째 그룹의 데이터 (숫자 배열)",
        alias="groupA"
    )
    group_b: List[float] = Field(
        ...,
        description="두 번째 그룹의 데이터 (숫자 배열)",
        alias="groupB"
    )
    significance_level: float = Field(
        default=0.05,
        description="유의 수준 (0.001 ~ 0.1)",
        ge=0.001,
        le=0.1,
        alias="significanceLevel"
    )

    @field_validator('group_a', 'group_b')
    @classmethod
    def validate_group_data(cls, v: List[float], info) -> List[float]:
        """그룹 데이터 검증"""
        field_name = info.field_name

        if not v:
            raise ValueError(f"{field_name}: 빈 데이터입니다")

        if len(v) < 3:
            raise ValueError(f"{field_name}: 최소 3개의 데이터 포인트가 필요합니다")

        if len(v) > 10000:
            raise ValueError(f"{field_name}: 데이터 포인트가 너무 많습니다 (최대 10000개)")

        # 수치 값 검증
        for i, val in enumerate(v):
            if not isinstance(val, (int, float)):
                raise ValueError(f"{field_name}: {i+1}번째 값이 숫자가 아닙니다")

            if not np.isfinite(val):
                raise ValueError(f"{field_name}: {i+1}번째 값이 유효하지 않은 숫자입니다")

            # 비합리적인 값 범위 검증
            if abs(val) > 1e15:
                raise ValueError(f"{field_name}: {i+1}번째 값이 너무 큽니다")

        return v

    @field_validator('significance_level')
    @classmethod
    def validate_significance_level(cls, v: float) -> float:
        """유의 수준 검증"""
        if not np.isfinite(v):
            raise ValueError("유의 수준이 유효하지 않은 숫자입니다")

        if v < 0.001:
            logger.warning("유의 수준이 매우 엄격합니다")

        if v > 0.1:
            logger.warning("유의 수준이 관대한 편입니다")

        return v

    @model_validator(mode='after')
    def validate_groups_consistency(self) -> 'StatisticalTestInput':
        """그룹 간 일관성 검증"""
        # 데이터 다양성 검증
        all_data = self.group_a + self.group_b

        if len(set(all_data)) == 1:
            logger.warning("모든 데이터 값이 동일합니다")

        # 데이터 규모 검증
        total_points = len(self.group_a) + len(self.group_b)
        if total_points > 1000:
            logger.info(f"대용량 데이터 처리: {total_points} 포인트")

        return self

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "groupA": [98.5, 97.8, 99.2, 96.7, 98.1],
                "groupB": [85.2, 87.3, 84.9, 86.1, 85.8],
                "significanceLevel": 0.05
            }
        }
    )


class StatisticalTestResponse(BaseModel):
    """
    통계 테스트 응답 모델

    독립적인 통계 테스트 결과를 반환하는 모델
    """
    success: bool = Field(default=True, description="요청 성공 여부")
    test_name: str = Field(..., description="수행된 테스트 이름", alias="testName")
    result: StatisticalTestResult = Field(..., description="테스트 결과")
    processing_time: float = Field(..., description="처리 시간 (초)", alias="processingTime")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="응답 생성 시간")
    message: Optional[str] = Field(None, description="추가 메시지")

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "success": True,
                "testName": "Mann-Whitney U",
                "result": {
                    "testName": "Mann-Whitney U",
                    "statistic": 25.0,
                    "pValue": 0.0234,
                    "significant": True,
                    "interpretation": "통계적으로 유의한 차이 (p=0.0234)"
                },
                "processingTime": 0.0012,
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }
    )
