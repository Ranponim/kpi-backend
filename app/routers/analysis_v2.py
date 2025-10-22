"""
간소화된 분석 결과 API 라우터 (v2)

핵심 목적: MCP LLM 분석 결과 저장 및 프론트엔드 조회
설계 원칙: 단순성, 성능, 실용성
"""

import logging
from fastapi import APIRouter, Body, HTTPException, status, Query
from typing import Optional
from datetime import datetime
from pymongo.errors import PyMongoError
from bson import BSON

from ..db import get_database
from ..models.analysis_simplified import (
    AnalysisResultSimplifiedCreate,
    AnalysisResultSimplifiedModel,
    AnalysisResultSimplifiedResponse,
    AnalysisResultSimplifiedListResponse,
    AnalysisResultSimplifiedSummary,
    AnalysisResultSimplifiedFilter,
)
from ..models.common import PyObjectId
from ..exceptions import (
    AnalysisResultNotFoundException,
    InvalidAnalysisDataException,
    DatabaseConnectionException,
)

# 로깅 설정
logger = logging.getLogger("app.analysis_v2")

# 라우터 생성
router = APIRouter(
    prefix="/api/analysis/results-v2",
    tags=["Analysis Results V2 (Simplified)"],
    responses={
        400: {"description": "Bad Request"},
        404: {"description": "Not Found"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)


@router.post(
    "/",
    response_model=AnalysisResultSimplifiedResponse,
    response_model_by_alias=False,  # ✅ alias 대신 필드명 사용 (id 포함)
    status_code=status.HTTP_201_CREATED,
    summary="분석 결과 생성 (간소화)",
    description="MCP에서 전송한 LLM 분석 결과를 저장합니다."
)
async def create_analysis_result_v2(
    result: AnalysisResultSimplifiedCreate = Body(
        ...,
        examples=[
            {
                "ne_id": "nvgnb#10000",
                "cell_id": "2010",
                "swname": "host01",
                "rel_ver": "R23A",
                "analysis_period": {
                    "n_minus_1_start": "2025-01-19 00:00:00",
                    "n_minus_1_end": "2025-01-19 23:59:59",
                    "n_start": "2025-01-20 00:00:00",
                    "n_end": "2025-01-20 23:59:59"
                },
                "choi_result": {
                    "enabled": True,
                    "status": "normal",
                    "score": 9.2
                },
                "llm_analysis": {
                    "executive_summary": "AirMacDLThruAvg(Kbps)의 급격한 감소와 RandomlySelectedPreamblesLow(count)의 현저한 변동성을 확인했습니다.",
                    "diagnostic_findings": [
                        {
                            "primary_hypothesis": "다운링크 자원 할당 실패 또는 접속 시도 실패로 인한 성능 저하",
                            "supporting_evidence": "AirMacDLThruAvg(Kbps)의 극심한 감소는 다운링크 데이터 전송에 직접적인 영향을 미치므로...",
                            "confounding_factors_assessment": "동일 환경 가정 하에서, 하드웨어 오류 가능성은 낮다고 판단됩니다..."
                        }
                    ],
                    "recommended_actions": [
                        {
                            "priority": "P1",
                            "action": "다운링크 자원 할당 관련 로그 분석 및 스케줄링 파라미터 확인",
                            "details": "2025-09-04_12:30~2025-09-04_13:45 구간과 2025-09-05_12:45~2025-09-05_13:00 구간의 RRC connection setup failure 로그를 비교 분석..."
                        }
                    ],
                    "technical_analysis": "DL Throughput 감소는 PRB 할당 실패와 직접 연관됩니다.",
                    "cells_with_significant_change": ["cell_2010", "cell_2011"],
                    "action_plan": "1단계: 로그 수집 → 2단계: RRC 실패 원인 분석 → 3단계: 파라미터 조정 → 4단계: 모니터링",
                    "key_findings": [
                        "DL Throughput 85% 급감",
                        "RACH Preamble 변동성 200% 증가",
                        "특정 시간대 집중 발생"
                    ],
                    "confidence": 0.92,
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
                ]
            },
            {
                "ne_id": "All NEs",
                "cell_id": "All cells",
                "swname": "All hosts",
                "rel_ver": None,
                "analysis_period": {
                    "n_minus_1_start": "2025-01-19 00:00:00",
                    "n_minus_1_end": "2025-01-19 23:59:59",
                    "n_start": "2025-01-20 00:00:00",
                    "n_end": "2025-01-20 23:59:59"
                },
                "llm_analysis": {
                    "summary": "전체 네트워크 분석 - 모든 셀 집계",
                    "issues": ["일부 셀에서 성능 저하 감지"],
                    "recommendations": ["개별 셀 분석 권장"],
                    "confidence": 0.85,
                    "model_name": "gemini-2.5-pro"
                },
                "peg_comparisons": [
                    {
                        "peg_name": "RACH_SUCCESS_RATE",
                        "n_minus_1": {"avg": 96.5, "pct_95": 98.5},
                        "n": {"avg": 97.0, "pct_95": 99.0},
                        "change_absolute": 0.5,
                        "change_percentage": 0.52
                    }
                ]
            }
        ]
    )
):
    """
    새로운 LLM 분석 결과를 저장합니다.
    
    **필수 필드:**
    - ne_id: Network Element ID
    - cell_id: Cell Identity
    - swname: Software Name
    - analysis_period: 분석 기간 (N-1, N)
    - llm_analysis: LLM 분석 결과
    - peg_comparisons: PEG 비교 결과
    
    **선택 필드:**
    - rel_ver: Release Version
    - choi_result: Choi 알고리즘 결과
    - analysis_id: 분석 고유 ID
    """
    try:
        # 🔍 422 에러 디버깅: 수신된 데이터 구조 로깅
        logger.debug("🔍 [422 디버깅] 수신된 요청 데이터:")
        logger.debug("  - result type: %s", type(result).__name__)
        logger.debug("  - result dict keys: %s", list(result.model_dump().keys()) if hasattr(result, 'model_dump') else "No model_dump method")
        
        # LLM 분석 데이터 상세 로깅
        if hasattr(result, 'llm_analysis') and result.llm_analysis:
            llm_data = result.llm_analysis
            logger.debug("  - llm_analysis type: %s", type(llm_data).__name__)
            logger.debug("  - llm_analysis keys: %s", list(llm_data.model_dump().keys()) if hasattr(llm_data, 'model_dump') else "No model_dump method")
            
            # diagnostic_findings 구조 확인
            if hasattr(llm_data, 'diagnostic_findings'):
                findings = llm_data.diagnostic_findings
                logger.debug("  - diagnostic_findings: %d개", len(findings) if findings else 0)
                if findings and len(findings) > 0:
                    first_finding = findings[0]
                    logger.debug("  - 첫 번째 finding type: %s", type(first_finding).__name__)
                    logger.debug("  - 첫 번째 finding keys: %s", list(first_finding.model_dump().keys()) if hasattr(first_finding, 'model_dump') else "No model_dump method")
            
            # recommended_actions 구조 확인
            if hasattr(llm_data, 'recommended_actions'):
                actions = llm_data.recommended_actions
                logger.debug("  - recommended_actions: %d개", len(actions) if actions else 0)
                if actions and len(actions) > 0:
                    first_action = actions[0]
                    logger.debug("  - 첫 번째 action type: %s", type(first_action).__name__)
                    logger.debug("  - 첫 번째 action keys: %s", list(first_action.model_dump().keys()) if hasattr(first_action, 'model_dump') else "No model_dump method")
        
        db = get_database()
        collection = db.analysis_results_v2
        
        # Pydantic 모델을 dict로 변환
        payload = result.model_dump(by_alias=False, exclude_unset=True)
        
        # created_at 자동 설정
        payload["created_at"] = datetime.utcnow()
        
        logger.info(
            "분석 결과 생성 시도: ne_id=%s, cell_id=%s, swname=%s",
            result.ne_id,
            result.cell_id,
            result.swname
        )
        
        # MongoDB 문서 크기 체크 (16MB 제한)
        try:
            encoded = BSON.encode(payload)
            doc_size = len(encoded)
            max_size = 16 * 1024 * 1024
            
            if doc_size > max_size:
                logger.error(f"문서 크기 초과: {doc_size}B > 16MB")
                raise InvalidAnalysisDataException(
                    f"Document too large ({doc_size} bytes). Reduce PEG count or details."
                )
            
            logger.debug(f"문서 크기: {doc_size / 1024:.2f} KB")
            
        except Exception as e:
            logger.warning(f"문서 크기 체크 실패 (계속 진행): {e}")
        
        # 중복 검사 (같은 NE, Cell, swname, 비슷한 시간의 결과 방지)
        # 최근 1분 이내에 동일한 조합의 결과가 있는지 확인
        one_minute_ago = datetime.utcnow().timestamp() - 60
        existing = await collection.find_one({
            "ne_id": result.ne_id,
            "cell_id": result.cell_id,
            "swname": result.swname,
            "created_at": {"$gte": datetime.fromtimestamp(one_minute_ago)}
        })
        
        if existing:
            logger.warning(
                "최근 중복 결과 발견: ne_id=%s, cell_id=%s, swname=%s",
                result.ne_id,
                result.cell_id,
                result.swname
            )
            # 중복이어도 저장은 허용 (warning만 출력)
        
        # 문서 삽입
        insert_result = await collection.insert_one(payload)
        
        # 생성된 문서 조회
        created_doc = await collection.find_one({"_id": insert_result.inserted_id})
        
        if not created_doc:
            raise DatabaseConnectionException("Failed to retrieve created analysis result")
        
        # 응답 모델로 변환
        analysis_model = AnalysisResultSimplifiedModel.from_mongo(created_doc)
        
        logger.info(
            "분석 결과 생성 완료: id=%s, ne_id=%s, cell_id=%s, swname=%s",
            str(insert_result.inserted_id),
            result.ne_id,
            result.cell_id,
            result.swname
        )
        
        return AnalysisResultSimplifiedResponse(
            message="Analysis result created successfully",
            data=analysis_model
        )
        
    except PyMongoError as e:
        logger.error(f"MongoDB 오류: {e}", exc_info=True)
        raise DatabaseConnectionException(f"Database operation failed: {str(e)}")
        
    except Exception as e:
        logger.error(f"분석 결과 생성 중 오류: {e}", exc_info=True)
        raise InvalidAnalysisDataException(f"Failed to create analysis result: {str(e)}")


@router.get(
    "/",
    response_model=AnalysisResultSimplifiedListResponse,
    response_model_by_alias=False,  # ✅ alias 대신 필드명 사용 (id 포함)
    summary="분석 결과 목록 조회",
    description="필터링 및 페이지네이션을 지원하는 분석 결과 목록 조회"
)
async def list_analysis_results_v2(
    page: int = Query(1, ge=1, description="페이지 번호"),
    size: int = Query(20, ge=1, le=100, description="페이지 크기"),
    ne_id: Optional[str] = Query(None, description="NE ID 필터"),
    cell_id: Optional[str] = Query(None, description="Cell ID 필터"),
    swname: Optional[str] = Query(None, description="SW Name 필터"),
    rel_ver: Optional[str] = Query(None, description="Release Version 필터"),
    date_from: Optional[datetime] = Query(None, description="시작 날짜"),
    date_to: Optional[datetime] = Query(None, description="종료 날짜"),
    choi_status: Optional[str] = Query(None, description="Choi 판정 상태"),
):
    """
    분석 결과 목록을 조회합니다.
    
    **필터 옵션:**
    - ne_id, cell_id, swname: 식별자 기반 필터링
    - rel_ver: Release 버전 필터링
    - date_from, date_to: 시간 범위 필터링
    - choi_status: Choi 알고리즘 판정 결과 필터링
    
    **정렬:** created_at 기준 내림차순 (최신순)
    """
    try:
        db = get_database()
        collection = db.analysis_results_v2
        
        # 필터 조건 구성 (복합 인덱스 활용 순서: ne_id → cell_id → swname)
        filter_dict = {}
        
        if ne_id:
            filter_dict["ne_id"] = ne_id
        if cell_id:
            filter_dict["cell_id"] = cell_id
        if swname:
            filter_dict["swname"] = swname
        if rel_ver:
            filter_dict["rel_ver"] = rel_ver
        if choi_status:
            filter_dict["choi_result.status"] = choi_status
            
        if date_from or date_to:
            filter_dict["created_at"] = {}
            if date_from:
                filter_dict["created_at"]["$gte"] = date_from
            if date_to:
                filter_dict["created_at"]["$lte"] = date_to
        
        logger.info(
            "분석 결과 목록 조회: page=%d, size=%d, filters=%s",
            page,
            size,
            filter_dict
        )
        
        # 전체 개수 조회
        total_count = await collection.count_documents(filter_dict)
        
        # 페이지네이션 계산
        skip = (page - 1) * size
        has_next = (skip + size) < total_count
        
        # 데이터 조회
        cursor = collection.find(filter_dict)
        cursor = cursor.sort("created_at", -1).skip(skip).limit(size)
        
        documents = await cursor.to_list(length=size)
        
        # 응답 모델로 변환
        items = [AnalysisResultSimplifiedModel.from_mongo(doc) for doc in documents]
        
        logger.info(
            "분석 결과 목록 조회 완료: total=%d, returned=%d, page=%d",
            total_count,
            len(items),
            page
        )
        
        return AnalysisResultSimplifiedListResponse(
            items=items,
            total=total_count,
            page=page,
            size=size,
            has_next=has_next
        )
        
    except PyMongoError as e:
        logger.error(f"MongoDB 오류: {e}", exc_info=True)
        raise DatabaseConnectionException(f"Database query failed: {str(e)}")
        
    except Exception as e:
        logger.error(f"분석 결과 목록 조회 중 오류: {e}", exc_info=True)
        raise DatabaseConnectionException(f"Failed to retrieve analysis results: {str(e)}")


@router.get(
    "/{result_id}",
    response_model=AnalysisResultSimplifiedResponse,
    response_model_by_alias=False,  # ✅ alias 대신 필드명 사용 (id 포함)
    summary="분석 결과 상세 조회",
    description="특정 ID의 분석 결과 상세 정보 조회"
)
async def get_analysis_result_v2(result_id: PyObjectId):
    """
    특정 ID의 분석 결과를 상세 조회합니다.
    
    **반환 정보:**
    - 전체 PEG 비교 결과
    - LLM 분석 상세
    - Choi 알고리즘 결과 (있는 경우)
    """
    try:
        db = get_database()
        collection = db.analysis_results_v2
        
        logger.info("분석 결과 상세 조회: id=%s", str(result_id))
        
        # 문서 조회
        document = await collection.find_one({"_id": result_id})
        
        if not document:
            raise AnalysisResultNotFoundException(str(result_id))
        
        # 응답 모델로 변환
        analysis_model = AnalysisResultSimplifiedModel.from_mongo(document)
        
        logger.info(
            "분석 결과 상세 조회 완료: id=%s, ne_id=%s, cell_id=%s",
            str(result_id),
            analysis_model.ne_id,
            analysis_model.cell_id
        )
        
        return AnalysisResultSimplifiedResponse(
            message="Analysis result retrieved successfully",
            data=analysis_model
        )
        
    except AnalysisResultNotFoundException:
        raise
        
    except PyMongoError as e:
        logger.error(f"MongoDB 오류: {e}", exc_info=True)
        raise DatabaseConnectionException(f"Database query failed: {str(e)}")
        
    except Exception as e:
        logger.error(f"분석 결과 조회 중 오류: {e}", exc_info=True)
        raise DatabaseConnectionException(f"Failed to retrieve analysis result: {str(e)}")


@router.delete(
    "/{result_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="분석 결과 삭제",
    description="특정 ID의 분석 결과 삭제"
)
async def delete_analysis_result_v2(result_id: PyObjectId):
    """
    특정 ID의 분석 결과를 삭제합니다.
    """
    try:
        db = get_database()
        collection = db.analysis_results_v2
        
        logger.info("분석 결과 삭제 시도: id=%s", str(result_id))
        
        # 문서 삭제
        delete_result = await collection.delete_one({"_id": result_id})
        
        if delete_result.deleted_count == 0:
            raise AnalysisResultNotFoundException(str(result_id))
        
        logger.info("분석 결과 삭제 완료: id=%s", str(result_id))
        
        # 204 No Content 응답
        return
        
    except AnalysisResultNotFoundException:
        raise
        
    except PyMongoError as e:
        logger.error(f"MongoDB 오류: {e}", exc_info=True)
        raise DatabaseConnectionException(f"Database delete failed: {str(e)}")
        
    except Exception as e:
        logger.error(f"분석 결과 삭제 중 오류: {e}", exc_info=True)
        raise DatabaseConnectionException(f"Failed to delete analysis result: {str(e)}")


@router.get(
    "/stats/summary",
    summary="통계 요약",
    description="분석 결과 통계 요약 (NE, Cell, swname별 집계)"
)
async def get_analysis_stats_v2():
    """
    분석 결과 통계를 조회합니다.
    
    **반환 정보:**
    - 전체 분석 건수
    - NE별 분석 건수
    - Cell별 분석 건수
    - Choi 판정 분포
    - 최근 분석 결과 (10건)
    """
    try:
        db = get_database()
        collection = db.analysis_results_v2
        
        logger.info("분석 통계 요약 조회")
        
        # 전체 개수
        total_count = await collection.count_documents({})
        
        # NE별 집계
        ne_stats = await collection.aggregate([
            {"$group": {"_id": "$ne_id", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]).to_list(length=10)
        
        # Choi 판정 분포
        choi_stats = await collection.aggregate([
            {"$match": {"choi_result.enabled": True}},
            {"$group": {"_id": "$choi_result.status", "count": {"$sum": 1}}}
        ]).to_list(length=None)
        
        # 최근 분석 결과
        recent_results = await collection.find(
            {},
            {"ne_id": 1, "cell_id": 1, "swname": 1, "created_at": 1, "llm_analysis.summary": 1}
        ).sort("created_at", -1).limit(10).to_list(length=10)
        
        summary = {
            "total_count": total_count,
            "by_ne": {stat["_id"]: stat["count"] for stat in ne_stats},
            "choi_distribution": {stat["_id"]: stat["count"] for stat in choi_stats},
            "recent_results": recent_results
        }
        
        logger.info("분석 통계 요약 조회 완료: total=%d", total_count)
        
        return {
            "success": True,
            "data": summary
        }
        
    except PyMongoError as e:
        logger.error(f"MongoDB 오류: {e}", exc_info=True)
        raise DatabaseConnectionException(f"Database aggregation failed: {str(e)}")
        
    except Exception as e:
        logger.error(f"통계 조회 중 오류: {e}", exc_info=True)
        raise DatabaseConnectionException(f"Failed to retrieve statistics: {str(e)}")


# 라우터 export
__all__ = ["router"]




