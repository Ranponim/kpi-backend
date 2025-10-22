"""
PEG 비교분석 데이터 모델

이 모듈은 PEG(Performance Engineering Guidelines) 비교분석 기능을 위한 
데이터 모델들을 정의합니다.
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from bson import ObjectId

from .common import PyObjectId

logger = logging.getLogger("app.models.peg_comparison")


class PEGComparisonPeriodData(BaseModel):
    """PEG 비교분석 기간별 데이터 모델"""
    
    period_name: str = Field(..., description="기간 이름 (예: N1, N)")
    start_date: datetime = Field(..., description="시작 날짜")
    end_date: datetime = Field(..., description="종료 날짜")
    kpi_data: Dict[str, Any] = Field(..., description="KPI 데이터")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="통계 정보")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            ObjectId: lambda v: str(v)
        }


class PEGComparisonResult(BaseModel):
    """PEG 비교분석 결과 모델"""
    
    peg_name: str = Field(..., description="PEG 이름")
    weight: int = Field(..., ge=1, le=100, description="가중치 (1-100)")
    n1_period_data: PEGComparisonPeriodData = Field(..., description="N1 기간 데이터")
    n_period_data: PEGComparisonPeriodData = Field(..., description="N 기간 데이터")
    comparison_metrics: Dict[str, Any] = Field(..., description="비교 지표")
    performance_score: float = Field(..., ge=0, le=100, description="성능 점수 (0-100)")
    status: str = Field(..., description="분석 상태 (success, warning, error)")
    recommendations: List[str] = Field(default_factory=list, description="개선 권장사항")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            ObjectId: lambda v: str(v)
        }


class PEGComparisonSummary(BaseModel):
    """PEG 비교분석 요약 모델"""
    
    total_pegs: int = Field(..., description="총 PEG 개수")
    analyzed_pegs: int = Field(..., description="분석된 PEG 개수")
    overall_score: float = Field(..., ge=0, le=100, description="전체 성능 점수")
    status_distribution: Dict[str, int] = Field(..., description="상태별 분포")
    top_performers: List[str] = Field(default_factory=list, description="상위 성능 PEG 목록")
    improvement_areas: List[str] = Field(default_factory=list, description="개선 영역 목록")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            ObjectId: lambda v: str(v)
        }


class PEGComparisonAnalysisMetadata(BaseModel):
    """PEG 비교분석 메타데이터 모델"""
    
    analysis_id: str = Field(..., description="분석 ID")
    analysis_type: str = Field(default="peg_comparison", description="분석 타입")
    algorithm_version: str = Field(default="v2.1.0", description="알고리즘 버전")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="생성 시간")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="업데이트 시간")
    processing_time: float = Field(..., description="처리 시간 (초)")
    data_source: str = Field(..., description="데이터 소스")
    user_id: Optional[str] = Field(None, description="사용자 ID")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            ObjectId: lambda v: str(v)
        }


class PEGComparisonAnalysisModel(BaseModel):
    """PEG 비교분석 전체 모델"""
    
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    analysis_id: str = Field(..., description="분석 ID")
    peg_comparison_results: List[PEGComparisonResult] = Field(..., description="PEG 비교분석 결과 목록")
    summary: PEGComparisonSummary = Field(..., description="요약 정보")
    metadata: PEGComparisonAnalysisMetadata = Field(..., description="메타데이터")
    cached: bool = Field(default=False, description="캐시 여부")
    mcp_version: str = Field(default="v2.1.0", description="MCP 서버 버전")
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            ObjectId: lambda v: str(v)
        }
    
    @classmethod
    def from_mongo(cls, data: dict) -> "PEGComparisonAnalysisModel":
        """MongoDB 문서를 모델로 변환"""
        if not data:
            return None
        
        # ObjectId를 문자열로 변환
        if "_id" in data:
            data["id"] = str(data["_id"])
        
        return cls(**data)


class PEGComparisonRequest(BaseModel):
    """PEG 비교분석 요청 모델"""
    
    analysis_id: str = Field(..., description="분석 ID")
    include_metadata: bool = Field(default=True, description="메타데이터 포함 여부")
    cache_ttl: int = Field(default=3600, ge=60, le=86400, description="캐시 TTL (초)")
    async_processing: bool = Field(default=False, description="비동기 처리 여부")
    algorithm_version: str = Field(default="v2.1.0", description="알고리즘 버전")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            ObjectId: lambda v: str(v)
        }


class PEGComparisonResponse(BaseModel):
    """PEG 비교분석 응답 모델"""
    
    success: bool = Field(..., description="성공 여부")
    data: Optional[PEGComparisonAnalysisModel] = Field(None, description="분석 결과 데이터")
    processing_time: float = Field(..., description="처리 시간 (초)")
    cached: bool = Field(default=False, description="캐시 여부")
    mcp_version: str = Field(default="v2.1.0", description="MCP 서버 버전")
    message: str = Field(default="PEG comparison analysis completed successfully", description="응답 메시지")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            ObjectId: lambda v: str(v)
        }


class AsyncTaskStatus(BaseModel):
    """비동기 작업 상태 모델"""
    
    task_id: str = Field(..., description="작업 ID")
    status: str = Field(..., description="작업 상태 (PENDING, PROGRESS, COMPLETED, FAILED)")
    progress: int = Field(default=0, ge=0, le=100, description="진행률 (0-100)")
    estimated_completion: Optional[datetime] = Field(None, description="예상 완료 시간")
    error_message: Optional[str] = Field(None, description="에러 메시지")
    result_data: Optional[Dict[str, Any]] = Field(None, description="결과 데이터")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            ObjectId: lambda v: str(v)
        }


class AsyncTaskResponse(BaseModel):
    """비동기 작업 응답 모델"""
    
    success: bool = Field(..., description="성공 여부")
    task_id: str = Field(..., description="작업 ID")
    status: str = Field(..., description="작업 상태")
    progress: int = Field(default=0, description="진행률")
    estimated_completion: Optional[datetime] = Field(None, description="예상 완료 시간")
    message: str = Field(default="Task status retrieved successfully", description="응답 메시지")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            ObjectId: lambda v: str(v)
        }


class MCPRequest(BaseModel):
    """MCP 서버 요청 모델"""
    
    analysis_id: str = Field(..., description="분석 ID")
    raw_data: Dict[str, Any] = Field(..., description="원시 데이터")
    options: Dict[str, Any] = Field(default_factory=dict, description="옵션")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            ObjectId: lambda v: str(v)
        }


class MCPResponse(BaseModel):
    """MCP 서버 응답 모델"""
    
    success: bool = Field(..., description="성공 여부")
    data: Dict[str, Any] = Field(..., description="응답 데이터")
    processing_time: float = Field(..., description="처리 시간 (초)")
    algorithm_version: str = Field(default="v2.1.0", description="알고리즘 버전")
    error_message: Optional[str] = Field(None, description="에러 메시지")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            ObjectId: lambda v: str(v)
        }


# 에러 모델들
class PEGComparisonError(BaseModel):
    """PEG 비교분석 에러 모델"""
    
    code: str = Field(..., description="에러 코드")
    message: str = Field(..., description="에러 메시지")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="에러 발생 시간")
    details: Optional[Dict[str, Any]] = Field(None, description="에러 상세 정보")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            ObjectId: lambda v: str(v)
        }


class PEGComparisonErrorResponse(BaseModel):
    """PEG 비교분석 에러 응답 모델"""
    
    success: bool = Field(default=False, description="성공 여부")
    error: PEGComparisonError = Field(..., description="에러 정보")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            ObjectId: lambda v: str(v)
        }


















