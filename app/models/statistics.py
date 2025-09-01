"""
Statistics 비교 분석 API를 위한 Pydantic 모델 정의

이 모듈은 Statistics 비교 분석 API의 요청 및 응답 데이터 구조를 정의합니다.
- 두 날짜 구간 비교 요청 모델
- 통계 분석 결과 응답 모델
- 개별 PEG 비교 결과 모델
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator, ConfigDict
import logging

# 로거 설정
logger = logging.getLogger(__name__)

class DateRange(BaseModel):
    """날짜 구간을 나타내는 모델"""
    start_date: datetime = Field(..., description="시작 날짜")
    end_date: datetime = Field(..., description="종료 날짜")
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        """종료 날짜가 시작 날짜보다 이후인지 검증"""
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('종료 날짜는 시작 날짜보다 이후여야 합니다')
        return v
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        },
        json_schema_extra={
            "example": {
                "start_date": "2025-08-01T00:00:00",
                "end_date": "2025-08-07T23:59:59"
            }
        }
    )

class StatisticsCompareRequest(BaseModel):
    """Statistics 비교 분석 API 요청 모델"""
    
    # 비교할 두 날짜 구간
    period1: DateRange = Field(..., description="첫 번째 비교 구간")
    period2: DateRange = Field(..., description="두 번째 비교 구간")
    
    # 분석 대상 PEG 목록
    peg_names: List[str] = Field(
        ..., 
        min_items=1, 
        description="분석할 PEG 이름 목록 (예: ['availability', 'rrc', 'erab'])"
    )
    
    # 필터링 조건 (선택사항)
    ne_filter: Optional[List[str]] = Field(
        None, 
        description="NE 필터 (지정하지 않으면 모든 NE 포함)"
    )
    cell_id_filter: Optional[List[str]] = Field(
        None, 
        description="Cell ID 필터 (지정하지 않으면 모든 Cell ID 포함)"
    )
    
    # 분석 옵션
    include_outliers: bool = Field(
        True, 
        description="이상치 포함 여부 (기본값: True)"
    )
    decimal_places: int = Field(
        4, 
        ge=0, 
        le=10, 
        description="소수점 자릿수 (0-10, 기본값: 4)"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "period1": {
                    "start_date": "2025-08-01T00:00:00",
                    "end_date": "2025-08-07T23:59:59"
                },
                "period2": {
                    "start_date": "2025-08-08T00:00:00",
                    "end_date": "2025-08-14T23:59:59"
                },
                "peg_names": ["availability", "rrc", "erab"],
                "ne_filter": ["nvgnb#10000", "nvgnb#20000"],
                "cell_id_filter": ["2010", "2011"],
                "include_outliers": True,
                "decimal_places": 4
            }
        }
    )

class PegStatistics(BaseModel):
    """개별 PEG의 통계 정보"""
    
    # 기본 통계
    count: int = Field(..., description="데이터 포인트 수")
    mean: float = Field(..., description="평균값")
    std: float = Field(..., description="표준편차")
    min: float = Field(..., description="최솟값")
    max: float = Field(..., description="최댓값")
    
    # 추가 통계 (선택사항)
    median: Optional[float] = Field(None, description="중앙값")
    percentile_25: Optional[float] = Field(None, description="25% 분위수")
    percentile_75: Optional[float] = Field(None, description="75% 분위수")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "count": 168,
                "mean": 99.5,
                "std": 0.3,
                "min": 98.2,
                "max": 99.9,
                "median": 99.6,
                "percentile_25": 99.3,
                "percentile_75": 99.8
            }
        }
    )

class PegComparisonResult(BaseModel):
    """개별 PEG의 비교 분석 결과"""
    
    peg_name: str = Field(..., description="PEG 이름")
    
    # 구간별 통계
    period1_stats: PegStatistics = Field(..., description="첫 번째 구간 통계")
    period2_stats: PegStatistics = Field(..., description="두 번째 구간 통계")
    
    # 비교 지표
    delta: float = Field(..., description="평균값 차이 (period2 - period1)")
    delta_percentage: float = Field(..., description="평균값 차이 백분율")
    rsd_period1: float = Field(..., description="첫 번째 구간 RSD (상대표준편차, %)")
    rsd_period2: float = Field(..., description="두 번째 구간 RSD (상대표준편차, %)")
    
    # 통계적 유의성 (향후 확장)
    t_statistic: Optional[float] = Field(None, description="t-검정 통계량")
    p_value: Optional[float] = Field(None, description="p-값")
    is_significant: Optional[bool] = Field(None, description="통계적 유의성 (p < 0.05)")
    
    # 개선/악화 판정
    improvement_status: str = Field(
        ..., 
        description="개선 상태 ('improved', 'degraded', 'stable')"
    )
    improvement_magnitude: str = Field(
        ..., 
        description="개선 정도 ('significant', 'moderate', 'minor', 'none')"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "peg_name": "availability",
                "period1_stats": {
                    "count": 168,
                    "mean": 99.5,
                    "std": 0.3,
                    "min": 98.2,
                    "max": 99.9
                },
                "period2_stats": {
                    "count": 168,
                    "mean": 99.7,
                    "std": 0.2,
                    "min": 98.8,
                    "max": 99.9
                },
                "delta": 0.2,
                "delta_percentage": 0.201,
                "rsd_period1": 0.302,
                "rsd_period2": 0.201,
                "improvement_status": "improved",
                "improvement_magnitude": "minor"
            }
        }
    )

class StatisticsCompareResponse(BaseModel):
    """Statistics 비교 분석 API 응답 모델"""
    
    # 요청 정보 요약
    request_summary: Dict[str, Any] = Field(..., description="요청 정보 요약")
    
    # 전체 분석 결과
    analysis_results: List[PegComparisonResult] = Field(
        ..., 
        description="PEG별 비교 분석 결과"
    )
    
    # 전체 요약 통계
    summary: Dict[str, Any] = Field(..., description="전체 요약 통계")
    
    # 메타데이터
    metadata: Dict[str, Any] = Field(..., description="분석 메타데이터")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "request_summary": {
                    "period1": "2025-08-01 to 2025-08-07",
                    "period2": "2025-08-08 to 2025-08-14", 
                    "peg_count": 3,
                    "filter_applied": True
                },
                "analysis_results": [
                    {
                        "peg_name": "availability",
                        "period1_stats": {"count": 168, "mean": 99.5},
                        "period2_stats": {"count": 168, "mean": 99.7},
                        "delta": 0.2,
                        "delta_percentage": 0.201,
                        "rsd_period1": 0.302,
                        "rsd_period2": 0.201,
                        "improvement_status": "improved",
                        "improvement_magnitude": "minor"
                    }
                ],
                "summary": {
                    "total_pegs_analyzed": 3,
                    "improved_count": 2,
                    "degraded_count": 1,
                    "stable_count": 0,
                    "avg_improvement": 0.15
                },
                "metadata": {
                    "analysis_timestamp": "2025-08-14T22:30:00Z",
                    "processing_time_ms": 1250,
                    "data_source": "mongodb",
                    "decimal_places": 4
                }
            }
        }
    )

class StatisticsCompareError(BaseModel):
    """Statistics 비교 분석 API 에러 응답 모델"""
    
    error_code: str = Field(..., description="에러 코드")
    error_message: str = Field(..., description="에러 메시지")
    details: Optional[Dict[str, Any]] = Field(None, description="상세 에러 정보")
    suggestions: Optional[List[str]] = Field(None, description="해결 제안사항")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error_code": "INSUFFICIENT_DATA",
                "error_message": "지정된 기간에 분석할 데이터가 부족합니다",
                "details": {
                    "period1_data_count": 0,
                    "period2_data_count": 150,
                    "missing_pegs": ["availability"]
                },
                "suggestions": [
                    "날짜 범위를 확장해보세요",
                    "다른 PEG를 선택해보세요",
                    "필터 조건을 완화해보세요"
                ]
            }
        }
    )

# 분석 결과 개선 상태 열거형
IMPROVEMENT_STATUS = {
    'improved': '개선됨',
    'degraded': '악화됨', 
    'stable': '안정됨'
}

IMPROVEMENT_MAGNITUDE = {
    'significant': '상당한 변화',  # > 5%
    'moderate': '보통 변화',      # 1-5%
    'minor': '미미한 변화',       # 0.1-1%
    'none': '변화 없음'           # < 0.1%
}

def calculate_improvement_magnitude(delta_percentage: float) -> str:
    """델타 백분율을 기반으로 개선 정도를 계산"""
    abs_delta = abs(delta_percentage)
    
    if abs_delta >= 5.0:
        return 'significant'
    elif abs_delta >= 1.0:
        return 'moderate'
    elif abs_delta >= 0.1:
        return 'minor'
    else:
        return 'none'

def calculate_improvement_status(delta: float, peg_name: str) -> str:
    """PEG 특성과 델타 값을 기반으로 개선 상태를 계산"""
    
    # PEG별 개선 방향 정의 (높을수록 좋은 지표 vs 낮을수록 좋은 지표)
    higher_is_better = {
        'availability', 'rrc', 'erab', 'sar', 'mobility_intra', 
        'cqi', 'se', 'dl_thp'  # 일반적으로 높을수록 좋은 KPI들
    }
    
    lower_is_better = {
        'ul_int', 'latency', 'jitter', 'packet_loss'  # 낮을수록 좋은 KPI들
    }
    
    if peg_name.lower() in higher_is_better:
        if delta > 0:
            return 'improved'
        elif delta < 0:
            return 'degraded'
        else:
            return 'stable'
    elif peg_name.lower() in lower_is_better:
        if delta < 0:
            return 'improved'
        elif delta > 0:
            return 'degraded'
        else:
            return 'stable'
    else:
        # 알 수 없는 PEG의 경우 중립적으로 판단
        if abs(delta) < 0.001:  # 매우 작은 차이
            return 'stable'
        elif delta > 0:
            return 'improved'  # 기본적으로 증가를 개선으로 간주
        else:
            return 'degraded'

if __name__ == "__main__":
    # 모델 테스트
    logger.info("Statistics 모델 테스트 시작")
    
    # 샘플 요청 생성
    sample_request = StatisticsCompareRequest(
        period1=DateRange(
            start_date=datetime(2025, 8, 1),
            end_date=datetime(2025, 8, 7)
        ),
        period2=DateRange(
            start_date=datetime(2025, 8, 8),
            end_date=datetime(2025, 8, 14)
        ),
        peg_names=["availability", "rrc", "erab"],
        ne_filter=["nvgnb#10000"],
        decimal_places=4
    )
    
    logger.info(f"샘플 요청 생성 완료: {sample_request.dict()}")
    print("✅ Statistics 모델 정의 완료!")

