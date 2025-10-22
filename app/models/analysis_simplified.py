"""
간소화된 LLM 분석 결과 모델

핵심 목적: MCP LLM 분석 결과를 저장하고 프론트엔드에서 조회
설계 원칙: 실용성, 단순성, 검색 효율성
"""

import logging
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime
from .common import PyObjectId

logger = logging.getLogger(__name__)


class PegStatistics(BaseModel):
    """PEG 통계 정보"""
    avg: Optional[float] = Field(None, description="평균값")
    pct_95: Optional[float] = Field(None, description="95 백분위수")
    pct_99: Optional[float] = Field(None, description="99 백분위수")
    min: Optional[float] = Field(None, description="최솟값")
    max: Optional[float] = Field(None, description="최댓값")
    count: Optional[int] = Field(None, description="샘플 수")
    std: Optional[float] = Field(None, description="표준편차")


class PegComparison(BaseModel):
    """PEG 비교 결과"""
    peg_name: str = Field(..., description="PEG 이름")
    n_minus_1: PegStatistics = Field(default_factory=PegStatistics, description="N-1 기간 통계")
    n: PegStatistics = Field(default_factory=PegStatistics, description="N 기간 통계")
    change_absolute: Optional[float] = Field(None, description="절대 변화량")
    change_percentage: Optional[float] = Field(None, description="변화율 (%)")
    llm_insight: Optional[str] = Field(None, description="해당 PEG에 대한 LLM 분석")


class ChoiAlgorithmResult(BaseModel):
    """
    Choi 알고리즘 판정 결과
    
    향후 다른 알고리즘 추가 시 유사한 구조로 확장 가능
    """
    enabled: bool = Field(default=False, description="알고리즘 적용 여부")
    status: Optional[str] = Field(None, description="판정 상태 (normal/warning/critical)")
    score: Optional[float] = Field(None, description="종합 점수")
    details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="상세 결과")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "enabled": True,
                "status": "warning",
                "score": 7.5,
                "details": {
                    "anomaly_count": 3,
                    "anomaly_pegs": ["RACH_SUCCESS_RATE", "UL_THROUGHPUT"],
                    "threshold": 8.0
                }
            }
        }
    )


class LLMAnalysis(BaseModel):
    """LLM 종합 분석 결과"""
    summary: Optional[str] = Field(None, description="종합 요약")
    issues: List[str] = Field(default_factory=list, description="발견된 이슈 목록")
    recommendations: List[str] = Field(default_factory=list, description="권장 조치 사항")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="분석 신뢰도 (0-1)")
    model_name: Optional[str] = Field(None, description="사용된 LLM 모델")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "summary": "전반적으로 개선된 성능을 보이나 RACH 성공률 모니터링 필요",
                "issues": [
                    "RACH 성공률이 5% 감소",
                    "UL Throughput 변동성 증가"
                ],
                "recommendations": [
                    "RACH 파라미터 재조정 검토",
                    "간섭 분석 수행 권장"
                ],
                "confidence": 0.92,
                "model_name": "gemini-2.5-pro"
            }
        }
    )


class AnalysisResultSimplified(BaseModel):
    """
    간소화된 분석 결과 모델
    
    핵심 목적:
    - MCP에서 LLM 분석 결과 저장
    - 프론트엔드에서 결과 조회 및 필터링
    - 시계열 분석 지원
    
    설계 원칙:
    - 중복 제거
    - 검색 효율성 (복합 인덱스: ne_id + cell_id + swname)
    - 명확한 필드명 (host → swname)
    - 확장 가능한 구조 (알고리즘 추가 용이)
    """
    
    # ========== 식별자 (복합 인덱스 대상) ==========
    ne_id: str = Field(..., description="Network Element ID", examples=["nvgnb#10000", "420", "All NEs"])
    cell_id: str = Field(..., description="Cell Identity", examples=["2010", "1100", "All cells"])
    swname: str = Field(..., description="Software Name (구 host)", examples=["host01", "192.168.1.10", "All hosts"])
    rel_ver: Optional[str] = Field(None, description="Release Version", examples=["R23A", "5.2.1"])
    
    # ========== 타임스탬프 ==========
    created_at: datetime = Field(
        default_factory=datetime.utcnow, 
        description="분석 결과 생성 시간"
    )
    analysis_period: Dict[str, str] = Field(
        ...,
        description="분석 기간",
        json_schema_extra={
            "example": {
                "n_minus_1_start": "2025-01-19 00:00:00",
                "n_minus_1_end": "2025-01-19 23:59:59",
                "n_start": "2025-01-20 00:00:00",
                "n_end": "2025-01-20 23:59:59"
            }
        }
    )
    
    # ========== Choi 알고리즘 결과 ==========
    choi_result: Optional[ChoiAlgorithmResult] = Field(
        default=None,
        description="Choi 알고리즘 판정 결과 (사용 시만)"
    )
    
    # ========== LLM 분석 ==========
    llm_analysis: LLMAnalysis = Field(
        default_factory=LLMAnalysis,
        description="LLM 종합 분석 결과"
    )
    
    # ========== PEG 비교 결과 ==========
    peg_comparisons: List[PegComparison] = Field(
        default_factory=list,
        description="PEG별 비교 분석 결과"
    )
    
    # ========== 메타정보 (최소화) ==========
    analysis_id: Optional[str] = Field(None, description="분석 고유 ID (추적용)")
    
    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            PyObjectId: lambda v: str(v)
        },
        json_schema_extra={
            "example": {
                "ne_id": "nvgnb#10000",
                "cell_id": "2010",
                "swname": "host01",
                "rel_ver": "R23A",
                "created_at": "2025-01-20T10:30:00Z",
                "analysis_period": {
                    "n_minus_1_start": "2025-01-19 00:00:00",
                    "n_minus_1_end": "2025-01-19 23:59:59",
                    "n_start": "2025-01-20 00:00:00",
                    "n_end": "2025-01-20 23:59:59"
                },
                "choi_result": {
                    "enabled": True,
                    "status": "normal",
                    "score": 9.2,
                    "details": {}
                },
                "llm_analysis": {
                    "summary": "성능 개선 확인",
                    "issues": [],
                    "recommendations": ["지속 모니터링"],
                    "confidence": 0.95,
                    "model_name": "gemini-2.5-pro"
                },
                "peg_comparisons": [
                    {
                        "peg_name": "RACH_SUCCESS_RATE",
                        "n_minus_1": {"avg": 97.5, "pct_95": 99.0},
                        "n": {"avg": 98.2, "pct_95": 99.5},
                        "change_absolute": 0.7,
                        "change_percentage": 0.72
                    }
                ],
                "analysis_id": "analysis-123"
            }
        }
    )


class AnalysisResultSimplifiedCreate(AnalysisResultSimplified):
    """분석 결과 생성용 모델 (POST /api/analysis/results-v2/)"""
    pass


class AnalysisResultSimplifiedModel(AnalysisResultSimplified):
    """MongoDB 저장/조회용 모델"""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    
    @classmethod
    def from_mongo(cls, data: dict):
        """MongoDB 문서를 Pydantic 모델로 변환"""
        if not data:
            return None
        doc = data.copy()
        if "_id" in doc:
            doc["id"] = doc["_id"]
            # analysis_id가 없거나 None인 경우 _id 값을 analysis_id로 사용
            if not doc.get("analysis_id"):
                doc["analysis_id"] = str(doc["_id"])
        return cls(**doc)


# ========== 응답 모델 ==========

class AnalysisResultSimplifiedResponse(BaseModel):
    """단일 조회 응답"""
    success: bool = True
    message: str = "Analysis result retrieved successfully"
    data: AnalysisResultSimplifiedModel


class AnalysisResultSimplifiedListResponse(BaseModel):
    """목록 조회 응답"""
    items: List[AnalysisResultSimplifiedModel]
    total: int
    page: int
    size: int
    has_next: bool


class AnalysisResultSimplifiedSummary(BaseModel):
    """
    목록 조회 시 요약 정보
    
    전체 PEG 결과를 포함하지 않고 핵심 정보만 반환하여
    목록 조회 성능 최적화
    """
    id: str = Field(alias="_id")
    ne_id: str
    cell_id: str
    swname: str
    rel_ver: Optional[str]
    created_at: datetime
    
    # 요약 정보
    choi_status: Optional[str] = Field(None, description="Choi 알고리즘 판정")
    llm_summary: Optional[str] = Field(None, description="LLM 요약")
    total_pegs: int = Field(0, description="분석된 PEG 개수")
    
    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={datetime: lambda v: v.isoformat()}
    )


# ========== 필터 모델 ==========

class AnalysisResultSimplifiedFilter(BaseModel):
    """
    검색 필터
    
    복합 인덱스 활용을 위한 필터 순서: ne_id → cell_id → swname
    """
    ne_id: Optional[str] = Field(None, description="NE ID 필터")
    cell_id: Optional[str] = Field(None, description="Cell ID 필터")
    swname: Optional[str] = Field(None, description="SW Name 필터")
    rel_ver: Optional[str] = Field(None, description="Release Version 필터")
    
    date_from: Optional[datetime] = Field(None, description="시작 날짜")
    date_to: Optional[datetime] = Field(None, description="종료 날짜")
    
    choi_status: Optional[str] = Field(None, description="Choi 판정 상태 필터")
    analysis_id: Optional[str] = Field(None, description="분석 ID 필터")
    
    model_config = ConfigDict(populate_by_name=True)


__all__ = [
    "AnalysisResultSimplified",
    "AnalysisResultSimplifiedCreate",
    "AnalysisResultSimplifiedModel",
    "AnalysisResultSimplifiedResponse",
    "AnalysisResultSimplifiedListResponse",
    "AnalysisResultSimplifiedSummary",
    "AnalysisResultSimplifiedFilter",
    "PegComparison",
    "PegStatistics",
    "ChoiAlgorithmResult",
    "LLMAnalysis",
]




