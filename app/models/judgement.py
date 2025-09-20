"""
Choi 알고리즘 판정 관련 데이터 모델

이 모듈은 3GPP KPI PEGs 판정 알고리즘(Choi Algorithm)에서 사용되는
모든 데이터 구조와 Enum을 정의합니다.

주요 구성 요소:
- JudgementType: 최종 판정 결과 (OK, POK, NOK, Can't Judge)
- CompareDetail: 세부 비교 결과 (Similar, Delta 계층, Improve/Degrade 등)
- 입력/출력 DTO: 알고리즘 간 데이터 전달용 구조체
- API 응답 모델: 최종 JSON 응답 구조

PRD 참조: 섹션 3.2 (데이터 모델), 3.4.1 (API 응답 구조)
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Enum 정의
# =============================================================================

class JudgementType(str, Enum):
    """
    KPI 판정 결과 타입 (PRD 5.4 참조)
    
    - OK: 정상 상태
    - POK: 부분적 문제 (Main KPI는 OK, Sub KPI 중 일부 NOK)
    - NOK: 문제 상태 (Main KPI NOK 또는 성능 저하/개선 감지)
    - CANT_JUDGE: 판정 불가 (데이터 부족, 높은 변동성 등)
    """
    OK = "OK"
    POK = "POK" 
    NOK = "NOK"
    CANT_JUDGE = "Can't judge"


class CompareDetail(str, Enum):
    """
    세부 비교 판정 결과 (PRD 4.1, 5.2 참조)
    
    이상 탐지 (4장):
    - RANGE: DIMS [min,max] 범위 벗어남
    - NEW: 신규 통계 (이전 PKG 버전에 없던 항목)
    - ND: pre/post 중 한쪽만 ND (No Data)
    - ZERO: pre/post 중 한쪽만 0
    - HIGH_DELTA: δ > β3 (500%)
    
    통계 분석 (5장):
    - SIMILAR: 유사한 수준 (δ ≤ β1/β2, |δ| < β5)
    - LOW_DELTA: 낮은 변화량 (β1/β2 < δ ≤ 2*β1/β2)
    - MEDIUM_DELTA: 중간 변화량 (2*β1/β2 < δ ≤ β3)
    - HIGH_DELTA: 높은 변화량 (δ > β3)
    - IMPROVE: 성능 개선 (positive KPI 증가 또는 negative KPI 감소)
    - DEGRADE: 성능 저하 (positive KPI 감소 또는 negative KPI 증가)
    - HIGH_VARIATION: 높은 변동성 (CV > β4)
    """
    # 이상 탐지 결과
    RANGE = "Range"
    NEW = "New"
    ND = "ND"
    ZERO = "Zero"
    HIGH_DELTA = "High Delta"
    
    # 통계 분석 결과  
    SIMILAR = "Similar"
    LOW_DELTA = "Low Delta"
    MEDIUM_DELTA = "Medium Delta"
    IMPROVE = "Improve"
    DEGRADE = "Degrade"
    HIGH_VARIATION = "High Variation"
    CANT_JUDGE = "Can't judge"
    
    # 최종 요약 결과
    PARTIALLY_OK = "Partially OK"  # POK용


class KPIPositivity(str, Enum):
    """
    KPI 극성 (PRD 5.2 참조)
    
    - POSITIVE: 값이 클수록 좋은 KPI (예: Throughput, Success Rate)
    - NEGATIVE: 값이 작을수록 좋은 KPI (예: Error Rate, Defect Rate)
    """
    POSITIVE = "positive"
    NEGATIVE = "negative"


# =============================================================================
# 기본 데이터 구조
# =============================================================================

class PegSampleSeries(BaseModel):
    """
    PEG 시계열 샘플 데이터 (입력용)
    
    각 PEG, 각 Cell, 각 기간(pre/post)의 샘플 배열을 저장
    """
    peg_name: str = Field(..., description="PEG 이름")
    cell_id: str = Field(..., description="셀 ID")
    pre_samples: List[Optional[float]] = Field(..., description="Pre 기간 샘플들")
    post_samples: List[Optional[float]] = Field(..., description="Post 기간 샘플들")
    unit: Optional[str] = Field(None, description="측정 단위")
    
    @validator('pre_samples', 'post_samples')
    def validate_samples(cls, v):
        """샘플 데이터 유효성 검증"""
        if not v:
            logger.warning("Empty sample data provided")
        return v


class PegPeriodStats(BaseModel):
    """
    PEG 기간별 통계 정보
    
    필터링된 샘플들로부터 계산된 기본 통계량
    """
    mean: Optional[float] = Field(None, description="평균값")
    min: Optional[float] = Field(None, description="최솟값") 
    max: Optional[float] = Field(None, description="최댓값")
    std: Optional[float] = Field(None, description="표준편차")
    cv: Optional[float] = Field(None, description="변동계수 (CV = std/mean)")
    nd_ratio: float = Field(0.0, description="ND 비율", ge=0.0, le=1.0)
    zero_ratio: float = Field(0.0, description="Zero 비율", ge=0.0, le=1.0)
    sample_count: int = Field(0, description="유효 샘플 수", ge=0)


class PegCompareMetrics(BaseModel):
    """
    PEG 비교 지표
    
    pre/post 기간 비교를 위한 계산된 지표들
    """
    delta_pct: Optional[float] = Field(None, description="변화율 δ = (post-pre)/pre*100")
    has_nd: bool = Field(False, description="ND 포함 여부")
    has_zero: bool = Field(False, description="Zero 포함 여부") 
    has_new: bool = Field(False, description="신규 통계 여부")
    out_of_range: bool = Field(False, description="Range 벗어남 여부")
    traffic_volume_class: Optional[str] = Field(None, description="트래픽 볼륨 분류 (low/high)")


class PegCompareDecision(BaseModel):
    """
    PEG 비교 판정 결과
    
    개별 PEG에 대한 최종 판정과 근거
    """
    detail: CompareDetail = Field(..., description="세부 판정 결과")
    reason: str = Field(..., description="판정 근거 설명")
    thresholds_used: Dict[str, float] = Field(default_factory=dict, description="사용된 임계값들")
    confidence: Optional[float] = Field(None, description="판정 신뢰도", ge=0.0, le=1.0)


# =============================================================================
# 중간 결과 구조
# =============================================================================

class FilteringResult(BaseModel):
    """
    필터링 알고리즘 결과 (6장)
    """
    valid_time_slots: Dict[str, List[int]] = Field(default_factory=dict, description="셀별 유효 시간 슬롯 인덱스")
    filter_ratio: float = Field(0.0, description="필터링 비율", ge=0.0, le=1.0)
    warning_message: Optional[str] = Field(None, description="50% 규칙 위반 시 경고 메시지")
    preprocessing_stats: Dict[str, Any] = Field(default_factory=dict, description="전처리 통계")
    median_values: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="PEG별 중앙값 정보")


class AbnormalDetectionResult(BaseModel):
    """
    이상 탐지 결과 (4장)
    """
    range_violations: Dict[str, List[str]] = Field(default_factory=dict, description="Range 위반 (anomaly_type -> cell_list)")
    new_statistics: Dict[str, List[str]] = Field(default_factory=dict, description="신규 통계")
    nd_anomalies: Dict[str, List[str]] = Field(default_factory=dict, description="ND 이상")
    zero_anomalies: Dict[str, List[str]] = Field(default_factory=dict, description="Zero 이상")
    high_delta_anomalies: Dict[str, List[str]] = Field(default_factory=dict, description="High Delta 이상")
    display_results: Dict[str, bool] = Field(default_factory=dict, description="α0 규칙에 따른 표시 여부")


class SimpleKPIJudgement(BaseModel):
    """
    간단한 KPI 판정 결과 (최종 요약용)
    """
    judgement_type: JudgementType = Field(..., description="판정 타입")
    compare_detail: CompareDetail = Field(..., description="판정 세부사항")
    reasoning: str = Field(..., description="판정 근거")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="판정 신뢰도")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="분석 메트릭")
    thresholds_used: Dict[str, Any] = Field(default_factory=dict, description="사용된 임계값")


class MainKPIJudgement(BaseModel):
    """
    Main KPI 판정 결과 (5장)
    """
    main_kpi_name: str = Field(..., description="Main KPI 이름")
    main_result: JudgementType = Field(..., description="Main KPI 판정 결과")
    main_decision: PegCompareDecision = Field(..., description="Main KPI 판정 세부사항")
    
    sub_results: List[Dict[str, Any]] = Field(default_factory=list, description="Sub KPI 판정 결과들")
    final_result: JudgementType = Field(..., description="최종 판정 결과 (Main + Sub 종합)")
    summary_text: str = Field(..., description="판정 요약 문구")
    
    # 통계 정보
    pre_stats: PegPeriodStats = Field(..., description="Pre 기간 통계")
    post_stats: PegPeriodStats = Field(..., description="Post 기간 통계")
    compare_metrics: PegCompareMetrics = Field(..., description="비교 지표")


# =============================================================================
# 최종 API 응답 모델 (PRD 3.4.1)
# =============================================================================

class CellJudgementDetail(BaseModel):
    """
    셀별 판정 세부 정보 (3장 UI 지원)
    """
    cell_id: str = Field(..., description="셀 ID")
    judgement_type: JudgementType = Field(..., description="판정 결과")
    judgement_detail: CompareDetail = Field(..., description="세부 판정")
    tooltip_info: Dict[str, Any] = Field(default_factory=dict, description="툴팁 표시 정보")
    delta_percentage: Optional[float] = Field(None, description="변화율 (%)")
    pre_value: Optional[float] = Field(None, description="Pre 기간 값")
    post_value: Optional[float] = Field(None, description="Post 기간 값")
    reasoning: str = Field("", description="판정 근거")


class KPITopicSummary(BaseModel):
    """
    KPI 토픽 요약 (3장 Main KPI 표시용)
    """
    topic_name: str = Field(..., description="토픽 이름")
    main_kpi_name: str = Field(..., description="Main KPI 이름")
    main_result: JudgementType = Field(..., description="Main KPI 판정")
    final_result: JudgementType = Field(..., description="최종 판정 (Main + Sub 종합)")
    
    # 3장 UI 지원 필드
    summary_text: str = Field(..., description="요약 문구 (5.4 형태)")
    cell_details: List[CellJudgementDetail] = Field(default_factory=list, description="셀별 세부 정보")
    sub_kpi_expandable: bool = Field(True, description="Sub KPI 확장 가능 여부")
    sub_kpi_summary: Dict[str, Any] = Field(default_factory=dict, description="Sub KPI 요약")
    
    # 통계 정보
    affected_cells: int = Field(0, description="영향받은 셀 수")
    total_cells: int = Field(0, description="전체 셀 수")


class ChoiAlgorithmResponse(BaseModel):
    """
    Choi 알고리즘 최종 응답 모델
    
    PRD 3.4.1에 정의된 JSON 구조를 정확히 반영하며,
    3장 UI 표시 요구사항도 함께 지원합니다.
    """
    # 메타데이터
    timestamp: datetime = Field(default_factory=datetime.now, description="처리 시각")
    processing_time_ms: Optional[float] = Field(None, description="처리 시간 (밀리초)")
    algorithm_version: str = Field("1.0.0", description="알고리즘 버전")
    
    # 필터링 결과 (6장)
    filtering: FilteringResult = Field(..., description="필터링 알고리즘 결과")
    
    # 이상 탐지 결과 (4장)
    abnormal_detection: AbnormalDetectionResult = Field(..., description="이상 탐지 결과")
    
    # KPI 판정 결과 (5장)
    kpi_judgement: Dict[str, MainKPIJudgement] = Field(default_factory=dict, description="KPI 토픽별 상세 판정 결과")
    
    # 3장 UI 지원을 위한 요약 정보
    ui_summary: Dict[str, KPITopicSummary] = Field(default_factory=dict, description="UI 표시용 KPI 토픽 요약")
    
    # 전체 요약 정보
    overall_summary: Dict[str, Any] = Field(default_factory=dict, description="전체 요약 통계")
    
    # 설정 정보 (디버깅용)
    config_used: Dict[str, Any] = Field(default_factory=dict, description="사용된 설정값들")
    
    # 추가 메타데이터
    total_cells_analyzed: int = Field(0, description="분석된 총 셀 수")
    total_pegs_analyzed: int = Field(0, description="분석된 총 PEG 수")
    processing_warnings: List[str] = Field(default_factory=list, description="처리 중 발생한 경고들")
    
    class Config:
        """Pydantic 설정"""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        json_schema_extra = {
            "example": {
                "timestamp": "2025-09-20T17:30:00.000Z",
                "processing_time_ms": 1250.5,
                "algorithm_version": "1.0.0",
                "total_cells_analyzed": 5,
                "total_pegs_analyzed": 12,
                "processing_warnings": [],
                "filtering": {
                    "valid_time_slots": {"cell_001": [0, 1, 3, 5, 7], "cell_002": [1, 2, 4, 6]},
                    "filter_ratio": 0.75,
                    "warning_message": None,
                    "preprocessing_stats": {"removed_samples": 3, "total_input": 20},
                    "median_values": {"cell_001": {"AirMacDLThruAvg": 1500.0}}
                },
                "abnormal_detection": {
                    "range_violations": {"Range": ["cell_001", "cell_002"]},
                    "new_statistics": {"New": []},
                    "nd_anomalies": {"ND": ["cell_003"]},
                    "zero_anomalies": {"Zero": []},
                    "high_delta_anomalies": {"High Delta": ["cell_001"]},
                    "display_results": {"Range": True, "High Delta": True, "ND": False}
                },
                "kpi_judgement": {
                    "air_mac_dl_thru": {
                        "main_kpi_name": "AirMacDLThruAvg",
                        "main_result": "OK",
                        "final_result": "POK",
                        "summary_text": "[Main KPI] changed from 1000 to 1100 by 10% (POK)",
                        "sub_results": [{"sub_kpi": "ConnNoAvg", "result": "NOK"}]
                    }
                },
                "ui_summary": {
                    "air_mac_dl_thru": {
                        "topic_name": "Air MAC DL Thru",
                        "main_kpi_name": "AirMacDLThruAvg",
                        "main_result": "OK",
                        "final_result": "POK",
                        "summary_text": "[Main KPI] changed from 1000 to 1100 by 10% (POK)",
                        "cell_details": [
                            {
                                "cell_id": "cell_001",
                                "judgement_type": "POK",
                                "judgement_detail": "Similar",
                                "delta_percentage": 10.0,
                                "pre_value": 1000.0,
                                "post_value": 1100.0,
                                "reasoning": "Main KPI OK, but Sub KPI ConnNoAvg shows NOK"
                            }
                        ],
                        "affected_cells": 1,
                        "total_cells": 5
                    }
                },
                "overall_summary": {
                    "total_ok": 2,
                    "total_pok": 1,
                    "total_nok": 1,
                    "total_cant_judge": 1,
                    "performance_trend": "stable"
                },
                "config_used": {
                    "filtering_thresholds": {"min": 0.87, "max": 1.13},
                    "alpha_0": 2,
                    "beta_values": {"beta_0": 1000, "beta_1": 5}
                }
            }
        }


# =============================================================================
# 유틸리티 함수
# =============================================================================

def create_empty_filtering_result() -> FilteringResult:
    """빈 필터링 결과 객체 생성"""
    return FilteringResult()


def create_empty_abnormal_detection_result() -> AbnormalDetectionResult:
    """빈 이상 탐지 결과 객체 생성"""
    return AbnormalDetectionResult()


def validate_judgement_data(data: Dict[str, Any]) -> bool:
    """
    판정 데이터 유효성 검증
    
    Args:
        data: 검증할 데이터
        
    Returns:
        bool: 유효성 여부
    """
    try:
        # 기본 구조 검증
        required_keys = ['filtering', 'abnormal_detection', 'kpi_judgement']
        for key in required_keys:
            if key not in data:
                logger.error(f"Required key '{key}' missing in judgement data")
                return False
        
        # 각 섹션별 세부 검증은 Pydantic 모델에서 자동 수행
        return True
        
    except Exception as e:
        logger.error(f"Error validating judgement data: {e}")
        return False


# =============================================================================
# 로깅 설정
# =============================================================================

# 모델 관련 로그 설정
logger.info("Choi Algorithm judgement models loaded successfully")
