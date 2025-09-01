"""
LLM 분석 API 라우터

MCP analysis_llm.py와 연동하여 LLM 기반 분석을 수행합니다.
"""

import json
import logging
import uuid
import os
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

import requests
from ..db import get_database
from ..models.analysis import AnalysisResultCreate, AnalysisResultModel, AnalysisResultBase

# 라우터 및 로거 설정
router = APIRouter()
logger = logging.getLogger(__name__)


class MongoJSONEncoder(json.JSONEncoder):
    """MongoDB ObjectId를 JSON으로 직렬화하기 위한 커스텀 인코더"""
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)


@router.post("/api/analysis/trigger-llm-analysis", response_model=Dict[str, str], status_code=status.HTTP_202_ACCEPTED)
async def trigger_llm_analysis(
    request_data: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """
    LLM 분석을 트리거합니다.
    
    Request body should contain:
    - db_config: PostgreSQL 연결 정보
    - n_minus_1: 이전 기간 (e.g., "2025-01-01_00:00~2025-01-01_23:59")
    - n: 현재 기간 (e.g., "2025-01-02_00:00~2025-01-02_23:59")
    - enable_mock: 테스트 모드 여부
    """
    try:
        # 분석 ID 생성
        analysis_id = str(uuid.uuid4())
        
        logger.info(f"LLM 분석 요청 시작: {analysis_id}")
        logger.info(f"요청 데이터: {request_data}")
        
        # 분석 상태를 MongoDB에 먼저 저장
        db = get_database()
        initial_result = AnalysisResultCreate(
            analysis_id=analysis_id,
            analysis_type="llm_analysis",
            status="processing",
            analysis_date=datetime.utcnow(),
            request_params=request_data,
            results=[]
        )
        
        await db.analysis_results.insert_one(initial_result.dict())

        # 사용자 Preference에서 DB 설정을 조회하여 주입 (요청값이 있으면 병합/덮어쓰기)
        user_id = request_data.get("user_id", "default")
        pref = await db.user_preferences.find_one({"user_id": user_id})
        pref_db = (pref or {}).get("database_settings", {})
        # 요청의 db_config가 있으면 요청값 우선, 없으면 preference 사용
        request_db_config = request_data.get("db_config") or {}
        effective_db_config = {
            "host": request_db_config.get("host", pref_db.get("host")),
            "port": request_db_config.get("port", pref_db.get("port", 5432)),
            "user": request_db_config.get("user", pref_db.get("user", "postgres")),
            "password": request_db_config.get("password", pref_db.get("password")),
            "dbname": request_db_config.get("dbname", pref_db.get("dbname", "postgres")),
            "table": request_data.get("table") or pref_db.get("table", "summary"),
        }

        # 백그라운드에서 LLM 분석 실행
        background_tasks.add_task(
            execute_llm_analysis,
            analysis_id,
            effective_db_config,
            request_data.get("n_minus_1", ""),
            request_data.get("n", ""),
            request_data.get("enable_mock", False)
        )
        
        return {
            "status": "triggered",
            "analysis_id": analysis_id,
            "message": "LLM 분석이 시작되었습니다. 잠시 후 결과를 확인할 수 있습니다."
        }
        
    except Exception as e:
        logger.exception(f"LLM 분석 트리거 실패: {e}")
        raise HTTPException(status_code=500, detail=f"분석 요청 처리 중 오류 발생: {str(e)}")


async def execute_llm_analysis(
    analysis_id: str,
    db_config: Dict[str, Any],
    n_minus_1: str,
    n: str,
    enable_mock: bool
):
    """
    백그라운드에서 LLM 분석을 실행하고 결과를 MongoDB에 저장합니다.
    """
    try:
        logger.info(f"LLM 분석 실행 시작: {analysis_id}")

        analysis_result: Dict[str, Any] | None = None

        # 실제 MCP API 호출 (환경변수 설정 시, enable_mock=False일 때만)
        if not enable_mock:
            mcp_url = os.getenv("MCP_ANALYZER_URL")
            mcp_api_key = os.getenv("MCP_API_KEY")
            try:
                if mcp_url:
                    logger.info(f"MCP 호출 시도: {mcp_url}")
                    payload = {
                        "db": db_config,
                        "n_minus_1": n_minus_1,
                        "n": n
                    }
                    headers = {"Content-Type": "application/json"}
                    if mcp_api_key:
                        headers["Authorization"] = f"Bearer {mcp_api_key}"
                    resp = requests.post(mcp_url, json=payload, headers=headers, timeout=60)
                    resp.raise_for_status()
                    analysis_result = resp.json()
                    logger.info("MCP 실제 분석 결과 수신 완료")
                else:
                    logger.warning("MCP_ANALYZER_URL 미설정. Mock으로 폴백합니다.")
            except Exception as e:
                logger.exception(f"MCP 호출 실패. Mock으로 폴백: {e}")
                analysis_result = None

        # 더 이상 MOCK 생성 금지: 실제 결과가 없다면 오류 처리
        if analysis_result is None:
            raise RuntimeError("LLM 분석 결과를 가져오지 못했습니다. MCP_ANALYZER_URL 설정 또는 실제 분석 로직이 필요합니다.")

        # MongoDB 상태 업데이트 - 원본 스키마 정보 포함
        db = get_database()
        source_meta = analysis_result.get("source_metadata", {})
        
        await db.analysis_results.update_one(
            {"analysis_id": analysis_id},
            {
                "$set": {
                    "status": "completed" if analysis_result.get("status") == "success" else "error",
                    "results": analysis_result,
                    "completed_at": datetime.utcnow(),
                    # 원본 PostgreSQL 스키마에서 추출한 정보 추가
                    "ne_id": source_meta.get("ne_id"),
                    "cell_id": source_meta.get("cell_id"),
                    "source_metadata": source_meta
                }
            }
        )
        
        logger.info(f"LLM 분석 처리 완료: {analysis_id}")
        
    except Exception as e:
        logger.exception(f"LLM 분석 실행 오류: {analysis_id}, {e}")
        await update_analysis_error(analysis_id, str(e))


async def update_analysis_error(analysis_id: str, error_message: str):
    """분석 오류 상태 업데이트"""
    try:
        db = get_database()
        await db.analysis_results.update_one(
            {"analysis_id": analysis_id},
            {
                "$set": {
                    "status": "error",
                    "error_message": error_message,
                    "completed_at": datetime.utcnow()
                }
            }
        )
    except Exception as e:
        logger.exception(f"Error updating analysis error status: {e}")


@router.get("/api/analysis/llm-analysis/{analysis_id}", response_model=AnalysisResultModel)
async def get_llm_analysis_result(analysis_id: str):
    """특정 LLM 분석 결과를 조회합니다."""
    try:
        db = get_database()
        result = await db.analysis_results.find_one({"analysis_id": analysis_id})
        
        if not result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis result not found")
        
        # ObjectId를 문자열로 변환
        if '_id' in result:
            result['_id'] = str(result['_id'])
        
        # results 필드가 dict인 경우 list로 변환 (Pydantic 호환성)
        if 'results' in result and isinstance(result['results'], dict):
            result['results'] = [result['results']]
        
        return AnalysisResultModel(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"LLM 분석 결과 조회 오류: {analysis_id}, {e}")
        raise HTTPException(status_code=500, detail=f"분석 결과 조회 중 오류 발생: {str(e)}")