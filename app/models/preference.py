"""
사용자 설정(Preference) 모델 정의

이 모듈은 사용자의 대시보드 및 통계 설정을 관리하기 위한
Pydantic 모델들을 정의합니다.
"""

import logging
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from .common import PyObjectId

# 로깅 설정
logger = logging.getLogger(__name__)


class AnalysisResultFilterSettings(BaseModel):
    """
    분석 결과 필터 설정
    
    사용자가 분석 결과 페이지에서 자주 사용하는 필터 설정을 저장합니다.
    """
    saved_filters: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="저장된 필터 목록"
    )
    default_filter: Optional[Dict[str, Any]] = Field(
        None,
        description="기본으로 적용할 필터"
    )
    favorite_ne_ids: List[str] = Field(
        default_factory=list,
        description="즐겨찾는 NE ID 목록"
    )
    favorite_cell_ids: List[str] = Field(
        default_factory=list,
        description="즐겨찾는 Cell ID 목록"
    )
    multi_cell_selections: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Multi Cell 선택 저장"
    )
    filter_auto_apply: bool = Field(
        default=False,
        description="페이지 로드시 기본 필터 자동 적용"
    )
    
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "saved_filters": [
                    {
                        "name": "Production Sites",
                        "neId": "NE_001,NE_002",
                        "cellId": "CELL_001",
                        "status": "completed",
                        "analysis_type": "llm_analysis"
                    }
                ],
                "default_filter": {
                    "status": "completed"
                },
                "favorite_ne_ids": ["NE_001", "NE_002", "NE_003"],
                "favorite_cell_ids": ["CELL_001", "CELL_002"],
                "multi_cell_selections": [
                    {
                        "name": "Site Alpha",
                        "cells": ["CELL_001", "CELL_002", "CELL_003"]
                    }
                ]
            }
        }
    )


class DashboardSettings(BaseModel):
    """
    대시보드 설정
    
    사용자가 대시보드에서 선택한 PEG, NE, Cell ID 등의 설정을 관리합니다.
    """
    selected_pegs: List[str] = Field(
        default_factory=list,
        description="선택된 PEG 목록"
    )
    selected_nes: List[str] = Field(
        default_factory=list,
        description="선택된 Network Element ID 목록",
        alias="selectedNEs"
    )
    selected_cell_ids: List[str] = Field(
        default_factory=list,
        description="선택된 Cell ID 목록",
        alias="selectedCellIds"
    )
    layout_config: Optional[Dict[str, Any]] = Field(
        None,
        description="대시보드 레이아웃 설정"
    )
    chart_preferences: Optional[Dict[str, Any]] = Field(
        None,
        description="차트 표시 설정"
    )
    auto_refresh: bool = Field(
        default=True,
        description="자동 새로고침 활성화 여부"
    )
    refresh_interval: int = Field(
        default=30,
        ge=5,
        le=300,
        description="새로고침 간격 (초, 5-300)"
    )
    
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "selected_pegs": ["RACH_SUCCESS_RATE", "RRC_SETUP_SUCCESS"],
                "selected_nes": ["eNB001", "eNB002"],
                "selected_cell_ids": ["CELL001", "CELL002"],
                "auto_refresh": True,
                "refresh_interval": 30
            }
        }
    )


class DateRangeSettings(BaseModel):
    """
    날짜 구간 설정
    
    Statistics 분석에서 사용되는 날짜 구간 설정입니다.
    """
    start_date: Optional[datetime] = Field(None, description="시작 날짜")
    end_date: Optional[datetime] = Field(None, description="종료 날짜")
    preset: Optional[str] = Field(
        None,
        description="사전 정의된 기간 (last_7_days, last_30_days, last_quarter)"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "start_date": "2025-08-01T00:00:00Z",
                "end_date": "2025-08-14T23:59:59Z",
                "preset": "last_14_days"
            }
        }
    )


class ComparisonOptions(BaseModel):
    """
    비교 분석 옵션
    
    Statistics에서 두 구간 비교 시 사용되는 설정입니다.
    """
    show_delta: bool = Field(default=True, description="Delta 값 표시 여부")
    show_rsd: bool = Field(default=True, description="RSD 값 표시 여부")
    show_percentage: bool = Field(default=True, description="백분율 표시 여부")
    decimal_places: int = Field(
        default=2,
        ge=0,
        le=6,
        description="소수점 자릿수 (0-6)"
    )
    chart_type: str = Field(
        default="bar",
        description="차트 유형 (bar, line, area)"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "show_delta": True,
                "show_rsd": True,
                "show_percentage": True,
                "decimal_places": 2,
                "chart_type": "bar"
            }
        }
    )


class DatabaseSettings(BaseModel):
    """
    PostgreSQL Database 설정 (LLM/Statistics 공통)
    """
    host: str = Field(default="", description="DB Host")
    port: int = Field(default=5432, description="DB Port")
    user: str = Field(default="postgres", description="DB User")
    password: str = Field(default="", description="DB Password")
    dbname: str = Field(default="postgres", description="Database Name")
    table: str = Field(default="summary", description="기본 테이블명(분석 파라미터의 테이블명 포함)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "host": "127.0.0.1",
                "port": 5432,
                "user": "postgres",
                "password": "secret",
                "dbname": "netperf",
                "table": "summary"
            }
        }
    )


class StatisticsSettings(BaseModel):
    """
    통계 분석 설정
    
    Statistics 탭에서 사용되는 모든 설정을 관리합니다.
    """
    date_range_1: DateRangeSettings = Field(
        default_factory=DateRangeSettings,
        description="첫 번째 날짜 구간",
        alias="dateRange1"
    )
    date_range_2: DateRangeSettings = Field(
        default_factory=DateRangeSettings,
        description="두 번째 날짜 구간",
        alias="dateRange2"
    )
    comparison_options: ComparisonOptions = Field(
        default_factory=ComparisonOptions,
        description="비교 분석 옵션",
        alias="comparisonOptions"
    )
    default_pegs: List[str] = Field(
        default_factory=list,
        description="기본으로 선택될 PEG 목록"
    )
    default_filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="기본 필터 설정"
    )
    
    model_config = ConfigDict(populate_by_name=True)


class NotificationSettings(BaseModel):
    """
    알림 설정
    
    시스템 알림 및 경고에 대한 사용자 설정입니다.
    """
    email_notifications: bool = Field(
        default=False,
        description="이메일 알림 활성화 여부"
    )
    browser_notifications: bool = Field(
        default=True,
        description="브라우저 알림 활성화 여부"
    )
    threshold_alerts: bool = Field(
        default=True,
        description="임계값 경고 알림 활성화 여부"
    )
    alert_email: Optional[str] = Field(
        None,
        description="알림 수신 이메일 주소"
    )


class UserPreferenceMetadata(BaseModel):
    """
    사용자 설정 메타데이터
    """
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="생성 시간"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="수정 시간"
    )
    version: str = Field(
        default="1.0",
        description="설정 버전"
    )
    last_accessed: Optional[datetime] = Field(
        None,
        description="마지막 접근 시간"
    )


class UserPreferenceBase(BaseModel):
    """
    사용자 설정 기본 모델
    """
    user_id: str = Field(
        ...,
        description="사용자 ID",
        alias="userId"
    )
    dashboard_settings: DashboardSettings = Field(
        default_factory=DashboardSettings,
        description="대시보드 설정",
        alias="dashboardSettings"
    )
    statistics_settings: StatisticsSettings = Field(
        default_factory=StatisticsSettings,
        description="통계 분석 설정",
        alias="statisticsSettings"
    )
    analysis_filter_settings: AnalysisResultFilterSettings = Field(
        default_factory=AnalysisResultFilterSettings,
        description="분석 결과 필터 설정",
        alias="analysisFilterSettings"
    )
    database_settings: DatabaseSettings = Field(
        default_factory=DatabaseSettings,
        description="PostgreSQL Database 설정(LLM/Statistics 공통)",
        alias="databaseSettings"
    )
    notification_settings: NotificationSettings = Field(
        default_factory=NotificationSettings,
        description="알림 설정",
        alias="notificationSettings"
    )
    theme: str = Field(
        default="light",
        description="UI 테마 (light, dark)"
    )
    language: str = Field(
        default="ko",
        description="언어 설정 (ko, en)"
    )
    metadata: UserPreferenceMetadata = Field(
        default_factory=UserPreferenceMetadata,
        description="메타데이터"
    )
    
    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            PyObjectId: lambda v: str(v)
        }
    )


class UserPreferenceCreate(UserPreferenceBase):
    """
    사용자 설정 생성 요청 모델
    """
    user_id: str = Field(
        default="default",
        description="사용자 ID",
        alias="userId"
    )


class UserPreferenceUpdate(BaseModel):
    """
    사용자 설정 업데이트 요청 모델
    
    부분 업데이트를 지원합니다.
    """
    dashboard_settings: Optional[DashboardSettings] = Field(
        None,
        alias="dashboardSettings"
    )
    statistics_settings: Optional[StatisticsSettings] = Field(
        None,
        alias="statisticsSettings"
    )
    analysis_filter_settings: Optional[AnalysisResultFilterSettings] = Field(
        None,
        alias="analysisFilterSettings"
    )
    database_settings: Optional[DatabaseSettings] = Field(
        None,
        alias="databaseSettings"
    )
    notification_settings: Optional[NotificationSettings] = Field(
        None,
        alias="notificationSettings"
    )
    theme: Optional[str] = Field(None, description="UI 테마")
    language: Optional[str] = Field(None, description="언어 설정")
    
    model_config = ConfigDict(populate_by_name=True)


class UserPreferenceModel(UserPreferenceBase):
    """
    사용자 설정 응답 모델
    
    데이터베이스에서 조회된 사용자 설정을 반환할 때 사용됩니다.
    """
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    
    @classmethod
    def from_mongo(cls, data: dict):
        """
        MongoDB 문서를 UserPreferenceModel로 변환
        
        Args:
            data: MongoDB에서 조회된 문서
            
        Returns:
            UserPreferenceModel: 변환된 모델 인스턴스
        """
        if not data:
            return None
            
        # _id를 id로 변환
        if "_id" in data:
            data["id"] = data["_id"]
            
        # metadata 처리 - 없으면 기본값 생성
        if "metadata" not in data:
            data["metadata"] = UserPreferenceMetadata().model_dump()
            
        return cls(**data)


class UserPreferenceImportExport(BaseModel):
    """
    설정 Import/Export 모델
    
    JSON 파일로 설정을 내보내거나 가져올 때 사용됩니다.
    """
    dashboard_settings: DashboardSettings
    statistics_settings: StatisticsSettings
    analysis_filter_settings: AnalysisResultFilterSettings
    database_settings: DatabaseSettings
    notification_settings: NotificationSettings
    theme: str
    language: str
    export_date: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "dashboard_settings": {
                    "selected_pegs": ["RACH_SUCCESS_RATE"],
                    "selected_nes": ["eNB001"],
                    "selected_cell_ids": ["CELL001"]
                },
                "statistics_settings": {
                    "date_range_1": {
                        "preset": "last_7_days"
                    },
                    "date_range_2": {
                        "preset": "last_14_days"
                    }
                },
                "theme": "light",
                "language": "ko"
            }
        }
    )


# 응답 모델들
class UserPreferenceResponse(BaseModel):
    """단일 사용자 설정 응답"""
    success: bool = True
    message: str = "User preference retrieved successfully"
    data: UserPreferenceModel


class UserPreferenceCreateResponse(BaseModel):
    """사용자 설정 생성 응답"""
    success: bool = True
    message: str = "User preference created successfully"
    data: UserPreferenceModel


class UserPreferenceUpdateResponse(BaseModel):
    """사용자 설정 업데이트 응답"""
    success: bool = True
    message: str = "User preference updated successfully"
    data: UserPreferenceModel


class PreferenceExportResponse(BaseModel):
    """설정 내보내기 응답"""
    success: bool = True
    message: str = "Preference exported successfully"
    data: UserPreferenceImportExport
    filename: str


class PreferenceImportResponse(BaseModel):
    """설정 가져오기 응답"""
    success: bool = True
    message: str = "Preference imported successfully"
    imported_settings: List[str] = Field(description="가져온 설정 항목들")
    warnings: List[str] = Field(default_factory=list, description="경고 메시지들")
