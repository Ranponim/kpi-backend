"""
LLM 분석 결과 모델 정의

이 모듈은 LLM 분석 결과의 저장, 조회, 관리를 위한 
Pydantic 모델들을 정의합니다.
"""

import logging
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from .common import PyObjectId

# 로깅 설정
logger = logging.getLogger(__name__)


class AnalysisDetail(BaseModel):
    """
    개별 KPI 분석 결과 상세 정보
    
    각 KPI에 대한 분석값, 임계값, 상태 등을 포함합니다.
    """
    kpi_name: Optional[str] = Field(None, description="KPI 이름 (예: RACH Success Rate)")  # ✅ LLM 분석용 Optional
    value: Optional[float] = Field(None, description="측정된 KPI 값")                    # ✅ LLM 분석용 Optional
    threshold: Optional[float] = Field(None, description="임계값 (설정된 경우)")
    status: str = Field(default="normal", description="상태 (normal, warning, critical)")
    unit: Optional[str] = Field(None, description="측정 단위 (%, dB, count 등)")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "kpi_name": "RACH Success Rate",
                "value": 98.5,
                "threshold": 95.0,
                "status": "normal",
                "unit": "%"
            }
        }
    )


class StatDetail(BaseModel):
    """
    통계 분석 결과 상세 정보
    
    특정 기간에 대한 KPI의 통계값들을 포함합니다.
    """
    period: str = Field(..., description="분석 기간 (예: N-1, N)")
    kpi_name: str = Field(..., description="KPI 이름")
    avg: Optional[float] = Field(None, description="평균값")
    std: Optional[float] = Field(None, description="표준편차")
    min: Optional[float] = Field(None, description="최솟값")
    max: Optional[float] = Field(None, description="최댓값")
    count: Optional[int] = Field(None, description="데이터 개수")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "period": "N-1",
                "kpi_name": "RACH Success Rate",
                "avg": 97.8,
                "std": 1.2,
                "min": 95.0,
                "max": 99.5,
                "count": 144
            }
        }
    )


class AnalysisMetadata(BaseModel):
    """
    분석 메타데이터
    
    분석 과정에서 생성되는 추가 정보들을 포함합니다.
    """
    created_at: datetime = Field(default_factory=datetime.utcnow, description="생성 시간")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="수정 시간")
    created_by: Optional[str] = Field(None, description="생성자")
    analysis_type: str = Field(default="llm_analysis", description="분석 유형")
    version: str = Field(default="1.0", description="분석 버전")
    processing_time: Optional[float] = Field(None, description="처리 시간 (초)")


class TargetScope(BaseModel):
    """
    분석 타겟 범위 정의
    
    NE, Cell, Host의 다중 필터링을 지원하는 구조체입니다.
    """
    ne_ids: Optional[List[str]] = Field(None, description="분석 대상 NE ID 목록")
    cell_ids: Optional[List[str]] = Field(None, description="분석 대상 Cell ID 목록")
    host_ids: Optional[List[str]] = Field(None, description="분석 대상 Host ID 목록")
    primary_ne: Optional[str] = Field(None, description="대표 NE ID")
    primary_cell: Optional[str] = Field(None, description="대표 Cell ID")
    primary_host: Optional[str] = Field(None, description="대표 Host ID")
    scope_type: str = Field(default="network_wide", description="분석 범위 타입 (specific_target, network_wide)")
    target_combinations: Optional[List[Dict[str, str]]] = Field(
        None, 
        description="NE-Cell-Host 조합 목록 (예: [{'ne': 'nvgnb#10000', 'cell': '2010', 'host': 'host01'}])"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ne_ids": ["nvgnb#10000", "nvgnb#20000"],
                "cell_ids": ["2010", "2011"],
                "host_ids": ["host01", "192.168.1.10"],
                "primary_ne": "nvgnb#10000",
                "primary_cell": "2010",
                "primary_host": "host01",
                "scope_type": "specific_target",
                "target_combinations": [
                    {"ne": "nvgnb#10000", "cell": "2010", "host": "host01"},
                    {"ne": "nvgnb#10000", "cell": "2011", "host": "host01"}
                ]
            }
        }
    )


class FilterMetadata(BaseModel):
    """
    필터링 메타데이터
    
    적용된 필터의 통계 및 커버리지 정보를 포함합니다.
    """
    applied_ne_count: int = Field(0, description="적용된 NE 수")
    applied_cell_count: int = Field(0, description="적용된 Cell 수")
    applied_host_count: int = Field(0, description="적용된 Host 수")
    data_coverage_ratio: Optional[float] = Field(None, description="필터링된 데이터 비율")
    relationship_coverage: Optional[Dict[str, int]] = Field(
        None,
        description="연관성 커버리지 (ne_cell_matches, host_ne_matches, full_combination_matches)"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "applied_ne_count": 2,
                "applied_cell_count": 4,
                "applied_host_count": 2,
                "data_coverage_ratio": 0.85,
                "relationship_coverage": {
                    "ne_cell_matches": 4,
                    "host_ne_matches": 2,
                    "full_combination_matches": 4
                }
            }
        }
    )


class LLMAnalysisSummary(BaseModel):
    """LLM 분석 요약 DTO"""

    summary: Optional[str] = Field(None, description="요약")
    issues: Optional[List[Any]] = Field(None, description="발견된 이슈 목록")
    recommended_actions: Optional[List[Any]] = Field(None, description="권장 조치")
    peg_insights: Dict[str, Any] = Field(default_factory=dict, description="PEG별 통찰")
    confidence: Optional[float] = Field(None, description="LLM 신뢰도")
    model: Optional[str] = Field(None, description="LLM 모델")


class PegMetricItem(BaseModel):
    """PEG 지표 항목"""

    peg_name: str
    n_minus_1_value: Optional[float]
    n_value: Optional[float]
    absolute_change: Optional[float]
    percentage_change: Optional[float]
    llm_analysis_summary: Optional[str] = None


class PegMetricSummary(BaseModel):
    """PEG 지표 요약 통계"""

    total_pegs: int = 0
    complete_data_pegs: int = 0
    incomplete_data_pegs: int = 0
    positive_changes: int = 0
    negative_changes: int = 0
    no_change: int = 0
    avg_percentage_change: Optional[float] = None


class PegMetricsPayload(BaseModel):
    """PEG 지표 응답 페이로드"""

    items: List[PegMetricItem] = Field(default_factory=list)
    statistics: PegMetricSummary = Field(default_factory=PegMetricSummary)


class AnalysisMetadataPayload(BaseModel):
    """분석 메타데이터"""

    workflow_version: str = "4.0"
    processing_timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str = "unknown"
    analysis_id: Optional[str] = None
    analysis_type: str = "enhanced"
    selected_pegs: List[str] = Field(default_factory=list)
    filters: Dict[str, Any] = Field(default_factory=dict)
    total_pegs: Optional[int] = None
    complete_data_pegs: Optional[int] = None
    source: str = "backend.analysis"
    output_dir: Optional[str] = None


class AnalysisResultBase(BaseModel):
    """분석 결과 기본 모델"""

    status: str = Field(default="success", description="분석 상태")
    time_ranges: Dict[str, Any] = Field(default_factory=dict, description="분석 시간 범위")
    peg_metrics: PegMetricsPayload = Field(default_factory=PegMetricsPayload)
    llm_analysis: LLMAnalysisSummary = Field(default_factory=LLMAnalysisSummary)
    metadata: AnalysisMetadataPayload = Field(default_factory=AnalysisMetadataPayload)
    legacy_payload: Optional[Dict[str, Any]] = Field(
        default=None,
        description="레거시 호환을 위한 원본 데이터",
        alias="legacyPayload"
    )
    
    # MongoDB 저장을 위한 필수 필드들
    ne_id: str = Field(..., description="NE ID")
    cell_id: str = Field(..., description="Cell ID") 
    analysis_date: datetime = Field(default_factory=datetime.utcnow, description="분석 날짜")
    analysis_type: str = Field(default="enhanced", description="분석 유형")

    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            PyObjectId: lambda v: str(v)
        }
    )


class AnalysisResultCreate(AnalysisResultBase):
    """
    분석 결과 생성 요청 모델
    
    새로운 분석 결과를 생성할 때 사용됩니다.
    """
    pass


class AnalysisResultUpdate(BaseModel):
    """분석 결과 업데이트 모델"""

    status: Optional[str] = None
    peg_metrics: Optional[PegMetricsPayload] = None
    llm_analysis: Optional[LLMAnalysisSummary] = None
    metadata: Optional[AnalysisMetadataPayload] = None


class AnalysisResultModel(AnalysisResultBase):
    """MongoDB 저장/조회용 분석 결과 모델"""

    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")

    @classmethod
    def from_mongo(cls, data: dict):
        """Mongo 문서를 최신 DTO에 맞게 변환"""

        if not data:
            return None

        doc = data.copy()

        if "_id" in doc:
            doc["id"] = doc["_id"]

        legacy_payload = {
            "results": doc.pop("results", None),
            "stats": doc.pop("stats", None),
            "results_overview": doc.pop("results_overview", None),
            "analysis_raw_compact": doc.pop("analysis_raw_compact", None),
            "analysis": doc.pop("analysis", None),
            "data": doc.pop("data", None),
        }

        llm_analysis = doc.get("llm_analysis") or {}
        
        # DTO 구조에서 peg_metrics 추출 (우선순위)
        peg_metrics_data = doc.get("peg_metrics", {})
        peg_items = peg_metrics_data.get("items", [])
        peg_statistics = peg_metrics_data.get("statistics", {})
        
        # 레거시 구조 fallback
        if not peg_items:
            peg_items = doc.get("peg_analysis", {}).get("results", [])
        if not peg_statistics:
            peg_statistics = doc.get("peg_analysis", {}).get("statistics", {})

        normalized = {
            "status": doc.get("status", "success"),
            "time_ranges": doc.get("time_ranges") or {},
            "peg_metrics": {
                "items": peg_items,
                "statistics": peg_statistics,
            },
            "llm_analysis": llm_analysis,
            "metadata": doc.get("metadata") or {},
            "legacy_payload": legacy_payload,
        }

        return cls(**{**doc, **normalized})


class AnalysisResultSummary(BaseModel):
    """
    분석 결과 요약 모델
    
    목록 조회 시 사용되는 간소화된 모델입니다.
    Host 정보를 포함한 확장된 요약 정보를 제공합니다.
    """
    id: str = Field(alias="_id")
    analysis_date: datetime = Field(alias="analysisDate")
    
    # 하위 호환성을 위한 기존 필드들 (Optional)
    ne_id: Optional[str] = Field(None, alias="neId", description="대표 NE ID (하위 호환성)")
    cell_id: Optional[str] = Field(None, alias="cellId", description="대표 Cell ID (하위 호환성)")
    
    # 새로운 타겟 범위 요약 정보
    target_summary: Optional[Dict[str, Any]] = Field(
        None, 
        description="타겟 범위 요약 (ne_count, cell_count, host_count, scope_type)"
    )
    
    status: str
    results_count: int = Field(0, description="분석 결과 개수")
    analysis_type: Optional[str] = Field(None, description="분석 유형 (standard, llm_analysis)")
    
    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={
            datetime: lambda v: v.isoformat()
        },
        json_schema_extra={
            "example": {
                "id": "64f8a9b2c3d4e5f6a7b8c9d0",
                "analysis_date": "2025-08-20T10:30:00Z",
                "ne_id": "nvgnb#10000",
                "cell_id": "2010",
                "target_summary": {
                    "ne_count": 2,
                    "cell_count": 4,
                    "host_count": 2,
                    "scope_type": "specific_target"
                },
                "status": "success",
                "results_count": 15,
                "analysis_type": "llm_analysis"
            }
        }
    )


class AnalysisResultFilter(BaseModel):
    """
    분석 결과 필터링 모델
    
    검색 및 필터링 파라미터를 정의합니다.
    Host 필터링 지원을 포함한 확장된 필터링 옵션을 제공합니다.
    """
    # 하위 호환성을 위한 기존 필드들
    ne_id: Optional[str] = Field(None, alias="neId", description="Network Element ID로 필터링 (하위 호환성)")
    cell_id: Optional[str] = Field(None, alias="cellId", description="Cell ID로 필터링 (하위 호환성)")
    
    # 새로운 다중 필터링 지원
    ne_ids: Optional[List[str]] = Field(None, description="NE ID 목록으로 필터링")
    cell_ids: Optional[List[str]] = Field(None, description="Cell ID 목록으로 필터링")
    host_ids: Optional[List[str]] = Field(None, description="Host ID 목록으로 필터링")
    host_id: Optional[str] = Field(None, description="단일 Host ID로 필터링")
    
    # 기존 필터링 옵션들
    status: Optional[str] = Field(None, description="상태로 필터링")
    date_from: Optional[datetime] = Field(None, description="시작 날짜")
    date_to: Optional[datetime] = Field(None, description="종료 날짜")
    analysis_type: Optional[str] = Field(None, description="분석 유형으로 필터링")
    
    # 타겟 범위 기반 필터링
    scope_type: Optional[str] = Field(None, description="분석 범위 타입으로 필터링")
    
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "ne_ids": ["nvgnb#10000", "nvgnb#20000"],
                "cell_ids": ["2010", "2011"],
                "host_ids": ["host01", "192.168.1.10"],
                "status": "success",
                "date_from": "2025-08-08T00:00:00Z",
                "date_to": "2025-08-20T23:59:59Z",
                "analysis_type": "llm_analysis",
                "scope_type": "specific_target"
            }
        }
    )


# 응답 모델들
class AnalysisResultListResponse(BaseModel):
    """분석 결과 목록 응답"""
    items: List[Dict[str, Any]]  # ✅ Pydantic 모델 대신 Dict 사용
    total: int
    page: int
    size: int
    has_next: bool


class AnalysisResultResponse(BaseModel):
    """단일 분석 결과 응답"""
    success: bool = True
    message: str = "Analysis result retrieved successfully"
    data: AnalysisResultModel


class AnalysisResultCreateResponse(BaseModel):
    """분석 결과 생성 응답"""
    success: bool = True
    message: str = "Analysis result created successfully"
    data: AnalysisResultModel

