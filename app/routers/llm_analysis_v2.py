"""
LLM 분석 API 라우터 V2

Frontend Dashboard에서 MCP 분석을 트리거하고 결과를 V2 형식으로 저장합니다.
MCP의 AnalysisService를 직접 import하여 Python 모듈로 사용합니다.
"""

import logging
import uuid
import sys
import os
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, Body, status

from ..db import get_database

# 로깅 설정
logger = logging.getLogger(__name__)

# MCP 모듈 경로 추가
MCP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../3gpp_analysis_mcp"))
if MCP_PATH not in sys.path:
    sys.path.insert(0, MCP_PATH)
    logger.info(f"MCP 경로 추가: {MCP_PATH}")

# MCP 모듈 import
try:
    from analysis_llm.services import AnalysisService, AnalysisServiceError
    from analysis_llm.repositories import PostgreSQLRepository
    from analysis_llm.utils import TimeRangeParser, DataProcessor
    
    MCP_AVAILABLE = True
    logger.info("MCP 모듈 import 성공")
except ImportError as e:
    MCP_AVAILABLE = False
    logger.error(f"MCP 모듈 import 실패: {e}")
    AnalysisService = None
    AnalysisServiceError = None


# 라우터 생성
router = APIRouter()


@router.post(
    "/api/analysis/trigger-llm-analysis-v2",
    status_code=status.HTTP_202_ACCEPTED,
    summary="LLM 분석 트리거 (V2 - MCP 직접 호출)",
    description="Dashboard에서 LLM 분석을 트리거하고 MCP를 직접 호출하여 results-v2에 저장합니다."
)
async def trigger_llm_analysis_v2(
    request_data: Dict[str, Any] = Body(...),
    background_tasks: BackgroundTasks = None
):
    """
    LLM 분석을 트리거합니다 (V2).
    
    Request body:
    {
        "n_minus_1": "2025-01-19_00:00~2025-01-19_23:59",  // Time1 (N-1 기간)
        "n": "2025-01-20_00:00~2025-01-20_23:59",           // Time2 (N 기간)
        "ne_id": "nvgnb#10000",                              // NE 선택 (선택)
        "cell_id": "2010",                                   // Cell ID 선택 (선택)
        "db_config": {                                       // DB 설정
            "host": "...",
            "port": 5432,
            "user": "postgres",
            "password": "...",
            "dbname": "peg_db",
            "table": "summary"
        }
    }
    
    Returns:
        분석 ID와 상태 정보
    """
    if not MCP_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="MCP 모듈을 사용할 수 없습니다. 서버 설정을 확인하세요."
        )
    
    try:
        analysis_id = str(uuid.uuid4())
        
        logger.info(f"🚀 LLM 분석 V2 요청 시작: {analysis_id}")
        logger.info(f"📊 요청 데이터: {request_data.keys()}")
        
        # 사용자 Preference에서 DB 설정 병합
        user_id = request_data.get("user_id", "default")
        db = get_database()
        pref = await db.user_preferences.find_one({"user_id": user_id})
        pref_db = (pref or {}).get("database_settings", {})
        
        request_db_config = request_data.get("db_config") or {}
        effective_db_config = {
            "host": request_db_config.get("host", pref_db.get("host")),
            "port": request_db_config.get("port", pref_db.get("port", 5432)),
            "user": request_db_config.get("user", pref_db.get("user", "postgres")),
            "password": request_db_config.get("password", pref_db.get("password")),
            "dbname": request_db_config.get("dbname", pref_db.get("dbname", "postgres")),
            "table": request_data.get("table") or request_db_config.get("table") or pref_db.get("table", "summary"),
        }
        
        logger.info(f"🔌 DB 설정: host={effective_db_config.get('host')}, "
                   f"dbname={effective_db_config.get('dbname')}, "
                   f"table={effective_db_config.get('table')}")
        
        # 백그라운드에서 MCP 분석 실행
        if background_tasks:
            background_tasks.add_task(
                execute_mcp_analysis_and_save_v2,
                analysis_id,
                effective_db_config,
                request_data
            )
        else:
            # 백그라운드 태스크가 없으면 동기 실행 (테스트용)
            await execute_mcp_analysis_and_save_v2(
                analysis_id,
                effective_db_config,
                request_data
            )
        
        return {
            "status": "triggered",
            "analysis_id": analysis_id,
            "message": "LLM 분석이 시작되었습니다. 결과는 /api/analysis/results-v2/에 저장됩니다.",
            "mcp_method": "direct_import"
        }
        
    except Exception as e:
        logger.exception(f"❌ LLM 분석 V2 트리거 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"분석 요청 처리 중 오류 발생: {str(e)}"
        )


async def execute_mcp_analysis_and_save_v2(
    analysis_id: str,
    db_config: Dict[str, Any],
    request_data: Dict[str, Any]
):
    """
    MCP AnalysisService를 직접 호출하고 결과를 V2 형식으로 저장
    
    Args:
        analysis_id: 분석 ID
        db_config: PostgreSQL 연결 정보
        request_data: 분석 요청 데이터
    """
    try:
        logger.info(f"🔬 MCP 분석 실행 시작: {analysis_id}")
        
        # [1] PostgreSQLRepository 생성
        logger.info("📦 PostgreSQL Repository 생성 중...")
        db_repository = PostgreSQLRepository(
            host=db_config["host"],
            port=db_config["port"],
            user=db_config["user"],
            password=db_config["password"],
            dbname=db_config["dbname"]
        )
        
        # [2] AnalysisService 생성 (MCP의 핵심 서비스)
        logger.info("🧠 AnalysisService 생성 중...")
        analysis_service = AnalysisService(
            database_repository=db_repository,
            time_parser=TimeRangeParser(),
            data_processor=DataProcessor()
        )
        
        # [3] MCP 분석 요청 데이터 구성
        mcp_request = {
            "n_minus_1": request_data.get("n_minus_1"),
            "n": request_data.get("n"),
            "table": db_config.get("table", "summary"),
            "analysis_id": analysis_id,
        }
        
        # ne_id 또는 cell_id가 있으면 filters 추가
        if request_data.get("ne_id") or request_data.get("cell_id"):
            mcp_request["filters"] = {}
            if request_data.get("ne_id"):
                mcp_request["filters"]["ne"] = request_data.get("ne_id")
            if request_data.get("cell_id"):
                mcp_request["filters"]["cellid"] = request_data.get("cell_id")
        
        logger.info(f"📋 MCP 요청 데이터: n_minus_1={mcp_request['n_minus_1']}, "
                   f"n={mcp_request['n']}, table={mcp_request.get('table')}")
        
        # [4] MCP 분석 실행 (동기 함수)
        logger.info("⚡ MCP 분석 실행 중...")
        mcp_result = analysis_service.perform_analysis(mcp_request)
        
        logger.info(f"✅ MCP 분석 완료: {analysis_id}")
        logger.debug(f"📊 MCP 결과 키: {list(mcp_result.keys())}")
        
        # [5] V2 형식으로 변환 및 저장
        logger.info("💾 V2 형식으로 저장 중...")
        db = get_database()
        v2_collection = db.analysis_results_v2
        
        # MCP 결과를 V2 형식으로 변환
        v2_payload = convert_mcp_result_to_v2_format(
            mcp_result,
            analysis_id,
            request_data
        )
        
        # MongoDB에 저장
        result = await v2_collection.insert_one(v2_payload)
        
        logger.info(f"💿 V2 결과 저장 완료: analysis_id={analysis_id}, "
                   f"mongodb_id={result.inserted_id}")
        
        # [6] Repository 정리
        db_repository.close()
        logger.info(f"🏁 분석 완료: {analysis_id}")
        
    except AnalysisServiceError as e:
        logger.error(f"❌ MCP 분석 서비스 오류: {analysis_id}, {e}")
        await save_analysis_error_v2(analysis_id, str(e), "mcp_analysis_error")
    except Exception as e:
        logger.exception(f"❌ MCP 분석 실행 중 예상치 못한 오류: {analysis_id}, {e}")
        await save_analysis_error_v2(analysis_id, str(e), "unexpected_error")


def convert_mcp_result_to_v2_format(
    mcp_result: Dict[str, Any],
    analysis_id: str,
    request_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    MCP 분석 결과를 V2 API 형식으로 변환
    
    Args:
        mcp_result: MCP AnalysisService.perform_analysis() 결과
        analysis_id: 분석 ID
        request_data: 원본 요청 데이터
        
    Returns:
        V2 형식의 페이로드
    """
    logger.debug("🔄 MCP 결과 → V2 형식 변환 시작")
    
    # MCP 결과 구조 확인
    source_metadata = mcp_result.get("source_metadata", {})
    llm_analysis = mcp_result.get("llm_analysis", {})
    peg_comparisons = mcp_result.get("peg_comparisons", [])
    choi_result = mcp_result.get("choi_result", {})
    
    # 시간 범위 파싱
    analysis_period = {}
    if "time_ranges" in mcp_result:
        time_ranges = mcp_result["time_ranges"]
        n_minus_1 = time_ranges.get("n_minus_1", {})
        n = time_ranges.get("n", {})
        
        analysis_period = {
            "n_minus_1_start": n_minus_1.get("start", ""),
            "n_minus_1_end": n_minus_1.get("end", ""),
            "n_start": n.get("start", ""),
            "n_end": n.get("end", "")
        }
    
    # V2 페이로드 구성
    v2_payload = {
        "analysis_id": analysis_id,
        "ne_id": source_metadata.get("ne_id", request_data.get("ne_id", "All NEs")),
        "cell_id": str(source_metadata.get("cell_id", request_data.get("cell_id", "All cells"))),
        "swname": source_metadata.get("swname", request_data.get("swname", "Unknown")),
        "rel_ver": source_metadata.get("rel_ver"),
        "analysis_period": analysis_period,
        "choi_result": {
            "enabled": bool(choi_result),
            "status": choi_result.get("overall", "unknown") if choi_result else "not_run",
            "score": choi_result.get("score"),
            "reasons": choi_result.get("reasons", [])
        } if choi_result else None,
        "llm_analysis": llm_analysis,
        "peg_comparisons": peg_comparisons,
        "created_at": datetime.utcnow(),
        "metadata": mcp_result.get("metadata", {})
    }
    
    logger.debug(f"✅ V2 변환 완료: ne_id={v2_payload['ne_id']}, "
                f"cell_id={v2_payload['cell_id']}, "
                f"peg_count={len(peg_comparisons)}")
    
    return v2_payload


async def save_analysis_error_v2(
    analysis_id: str,
    error_message: str,
    error_type: str = "unknown"
):
    """
    분석 오류를 V2 형식으로 저장
    
    Args:
        analysis_id: 분석 ID
        error_message: 오류 메시지
        error_type: 오류 유형
    """
    try:
        db = get_database()
        v2_collection = db.analysis_results_v2
        
        error_payload = {
            "analysis_id": analysis_id,
            "ne_id": "error",
            "cell_id": "error",
            "swname": "error",
            "rel_ver": None,
            "analysis_period": {},
            "choi_result": None,
            "llm_analysis": {
                "error": error_message,
                "error_type": error_type
            },
            "peg_comparisons": [],
            "created_at": datetime.utcnow(),
            "metadata": {
                "status": "error",
                "error_message": error_message,
                "error_type": error_type
            }
        }
        
        await v2_collection.insert_one(error_payload)
        logger.info(f"🔴 오류 상태 저장 완료: {analysis_id}")
        
    except Exception as e:
        logger.exception(f"오류 상태 저장 실패: {analysis_id}, {e}")

