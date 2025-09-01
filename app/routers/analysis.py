"""
분석 결과 API 라우터

이 모듈은 LLM 분석 결과의 CRUD 작업을 위한 API 엔드포인트들을 정의합니다.
Task 39: Backend LLM 분석 결과 API 및 DB 스키마 구현
"""

import logging
from fastapi import APIRouter, Body, HTTPException, status, Query, Depends
from typing import List, Optional
from datetime import datetime
from pymongo.errors import DuplicateKeyError, PyMongoError
from bson import BSON
import bson

from ..db import get_analysis_collection
from ..models.analysis import (
    AnalysisResultModel,
    AnalysisResultCreate,
    AnalysisResultUpdate,
    AnalysisResultSummary,
    AnalysisResultFilter,
    AnalysisResultListResponse,
    AnalysisResultResponse,
    AnalysisResultCreateResponse
)
from ..models.common import PyObjectId
from ..exceptions import (
    AnalysisResultNotFoundException,
    InvalidAnalysisDataException,
    DatabaseConnectionException,
    DuplicateAnalysisResultException
)

# 로깅 설정
logger = logging.getLogger("app.analysis")

# 요청 추적 임포트
from ..middleware.request_tracing import (
    create_request_context_logger, 
    log_business_event, 
    log_data_event
)
from ..utils.validation import (
    validate_analysis_filters,
    validate_pagination_params,
    validate_analysis_result_data,
    ValidationError
)
from ..utils.data_optimization import (
    optimize_analysis_result,
    restore_analysis_result,
    get_optimization_stats
)
from ..utils.cache_manager import get_cache_manager

# 라우터 생성
router = APIRouter(
    prefix="/api/analysis/results",
    tags=["Analysis Results"],
    responses={
        400: {"description": "Bad Request"},
        404: {"description": "Not Found"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)

# 별도의 라우터 생성 (test-data용)
test_router = APIRouter(
    prefix="/api",
    tags=["Test Data"],
    responses={
        400: {"description": "Bad Request"},
        404: {"description": "Not Found"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)


def _normalize_legacy_keys(doc: dict) -> dict:
    """레거시 camelCase 문서를 snake_case 우선 형태로 정규화합니다."""
    if not isinstance(doc, dict):
        return doc
    mapping = {
        "analysisDate": "analysis_date",
        "neId": "ne_id",
        "cellId": "cell_id",
        "analysisType": "analysis_type",
        "resultsOverview": "results_overview",
        "analysisRawCompact": "analysis_raw_compact",
        "reportPath": "report_path",
        "requestParams": "request_params",
    }
    for old_key, new_key in mapping.items():
        if new_key not in doc and old_key in doc:
            doc[new_key] = doc[old_key]
    return doc


@router.post(
    "/",
    response_model=AnalysisResultCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="분석 결과 생성",
    description="새로운 LLM 분석 결과를 생성합니다."
)
async def create_analysis_result(
    result: AnalysisResultCreate = Body(
        ...,
        example={
            "ne_id": "eNB001",
            "cell_id": "CELL001",
            "analysis_date": "2025-08-14T10:00:00Z",
            "results": [
                {
                    "kpi_name": "RACH Success Rate",
                    "value": 98.5,
                    "threshold": 95.0,
                    "status": "normal",
                    "unit": "%"
                }
            ],
            "status": "success"
        }
    )
):
    """
    새로운 LLM 분석 결과를 생성합니다.
    
    - **ne_id**: 네트워크 장비 ID (예: eNB001)
    - **cell_id**: 셀 ID (예: CELL001)  
    - **analysis_date**: 분석 기준 날짜 (ISO 8601 형식)
    - **results**: KPI별 분석 결과 목록
    - **stats**: 통계 분석 결과 목록 (선택사항)
    - **status**: 분석 상태 (success, warning, error)
    """
    try:
        collection = get_analysis_collection()
        
        # 데이터 준비: DB에는 snake_case 필드명으로 저장 (v2 표준 키)
        # by_alias=False로 덤프하여 camelCase alias가 아닌 원 필드명으로 저장한다
        result_dict = result.model_dump(by_alias=False, exclude_unset=True)

        # MongoDB 문서 크기(16MB) 체크
        try:
            encoded = BSON.encode(result_dict)
            doc_size = len(encoded)
            max_size = 16 * 1024 * 1024
            warn_size = 12 * 1024 * 1024
            if doc_size > max_size:
                logger.error(f"문서 크기 초과: size={doc_size}B > 16MB")
                raise InvalidAnalysisDataException(
                    f"Document too large to store ({doc_size} bytes). Please reduce payload."
                )
            if doc_size > warn_size:
                logger.warning(f"문서 크기 경고: size={doc_size}B > 12MB, 축약 필요 가능")
        except Exception as e:
            logger.warning(f"문서 크기 체크 실패(계속 진행): {e}")
        
        # metadata 업데이트
        if "metadata" in result_dict:
            result_dict["metadata"]["created_at"] = datetime.utcnow()
            result_dict["metadata"]["updated_at"] = datetime.utcnow()
        
        # 요청 컨텍스트 로거 생성
        req_logger = create_request_context_logger("app.analysis.create")
        
        req_logger.info("새 분석 결과 생성 시도", extra={
            "ne_id": result.ne_id,
            "cell_id": result.cell_id,
            "analysis_type": result.analysis_type,
            "status": result.status,
            "document_size_bytes": doc_size if 'doc_size' in locals() else 0
        })
        
        # 비즈니스 이벤트 로깅
        log_business_event("analysis_result_create_started", {
            "ne_id": result.ne_id,
            "cell_id": result.cell_id,
            "analysis_type": result.analysis_type
        })
        
        # 중복 검사 (같은 NE, Cell, 날짜에 대한 분석 결과가 이미 있는지 확인)
        existing = await collection.find_one({
            "ne_id": result.ne_id,
            "cell_id": result.cell_id,
            "analysis_date": result.analysis_date
        })
        
        if existing:
            raise DuplicateAnalysisResultException(
                ne_id=result.ne_id,
                cell_id=result.cell_id,
                analysis_date=result.analysis_date.isoformat()
            )
        
        # 데이터 최적화 적용 (큰 문서인 경우)
        if doc_size > 1024 * 1024:  # 1MB 이상인 경우
            req_logger.info("대용량 문서 감지, 최적화 적용", extra={
                "original_size_bytes": doc_size,
                "ne_id": result.ne_id,
                "cell_id": result.cell_id
            })
            
            try:
                optimized_dict = await optimize_analysis_result(result_dict)
                optimized_size = len(bson.BSON.encode(optimized_dict))
                
                req_logger.info("문서 최적화 완료", extra={
                    "original_size": doc_size,
                    "optimized_size": optimized_size,
                    "compression_ratio": optimized_size / doc_size,
                    "space_saved": doc_size - optimized_size
                })
                
                result_dict = optimized_dict
                
            except Exception as e:
                req_logger.warning(f"문서 최적화 실패, 원본으로 저장: {e}", extra={
                    "error_type": type(e).__name__
                })
        
        # 문서 삽입
        insert_result = await collection.insert_one(result_dict)
        
        # 생성된 문서 조회
        created_result = await collection.find_one({"_id": insert_result.inserted_id})
        
        if not created_result:
            raise DatabaseConnectionException("Failed to retrieve created analysis result")
        
        # 응답 모델로 변환
        analysis_model = AnalysisResultModel.from_mongo(created_result)
        
        # 관련 캐시 무효화
        cache_manager = await get_cache_manager()
        await cache_manager.invalidate_analysis_caches(
            ne_id=result.ne_id,
            cell_id=result.cell_id
        )
        
        req_logger.info("분석 결과 생성 성공", extra={
            "document_id": str(insert_result.inserted_id),
            "ne_id": result.ne_id,
            "cell_id": result.cell_id,
            "analysis_type": result.analysis_type,
            "cache_invalidated": True
        })
        
        # 데이터 이벤트 로깅
        log_data_event("analysis_result_created", {
            "document_id": str(insert_result.inserted_id),
            "ne_id": result.ne_id,
            "cell_id": result.cell_id,
            "collection": "analysis_results"
        })
        
        return AnalysisResultCreateResponse(
            message="Analysis result created successfully",
            data=analysis_model
        )
        
    except (DuplicateAnalysisResultException, DatabaseConnectionException) as e:
        # 이미 정의된 커스텀 예외는 그대로 전파
        raise e
        
    except PyMongoError as e:
        req_logger.error("MongoDB 오류 발생", extra={
            "error_type": type(e).__name__,
            "error_message": str(e),
            "ne_id": result.ne_id,
            "cell_id": result.cell_id
        }, exc_info=True)
        raise DatabaseConnectionException(f"Database operation failed: {str(e)}")
        
    except Exception as e:
        req_logger.error("분석 결과 생성 중 예상치 못한 오류", extra={
            "error_type": type(e).__name__,
            "error_message": str(e),
            "ne_id": result.ne_id,
            "cell_id": result.cell_id
        }, exc_info=True)
        raise InvalidAnalysisDataException(f"Failed to create analysis result: {str(e)}")


@test_router.post(
    "/testdata",
    summary="테스트 데이터 생성",
    description="개발용 테스트 데이터를 생성합니다."
)
async def create_test_data():
    """테스트 데이터를 생성합니다."""
    from datetime import datetime
    from bson import ObjectId

    collection = get_analysis_collection()

    # 기존 오류 데이터 삭제
    await collection.delete_many({'status': 'error'})

    # 새로운 테스트 데이터 생성
    test_data = [
        {
            '_id': ObjectId(),
            'neId': 'TEST_NE_001',
            'cellId': 'TEST_CELL_001',
            'status': 'completed',
            'createdAt': datetime.now(),
            'updatedAt': datetime.now(),
            'results_count': 100,
            'analysis_type': 'performance_analysis',
            'results_overview': '성능 분석이 완료되었습니다.',
            'analysis_summary': {
                'total_analyzed': 100,
                'success_count': 95,
                'error_count': 5,
                'performance_score': 85.5
            }
        },
        {
            '_id': ObjectId(),
            'neId': 'TEST_NE_002',
            'cellId': 'TEST_CELL_002',
            'status': 'completed',
            'createdAt': datetime.now(),
            'updatedAt': datetime.now(),
            'results_count': 150,
            'analysis_type': 'performance_analysis',
            'results_overview': '성능 분석이 완료되었습니다.',
            'analysis_summary': {
                'total_analyzed': 150,
                'success_count': 140,
                'error_count': 10,
                'performance_score': 78.9
            }
        }
    ]

    result = await collection.insert_many(test_data)
    return {
        "message": f"테스트 데이터 {len(result.inserted_ids)}개 생성 완료",
        "created_ids": [str(id) for id in result.inserted_ids]
    }

@router.get(
    "/",
    response_model=AnalysisResultListResponse,
    summary="분석 결과 목록 조회",
    description="분석 결과 목록을 조회합니다. 페이지네이션과 필터링을 지원합니다."
)
async def list_analysis_results(
    page: int = Query(1, ge=1, description="페이지 번호 (1부터 시작)"),
    size: int = Query(20, ge=1, le=100, description="페이지 크기 (1-100)"),
    ne_id: Optional[str] = Query(None, description="Network Element ID로 필터링"),
    cell_id: Optional[str] = Query(None, description="Cell ID로 필터링"),
    status: Optional[str] = Query(None, description="상태로 필터링 (success, warning, error)"),
    date_from: Optional[datetime] = Query(None, description="시작 날짜 (ISO 8601 형식)"),
    date_to: Optional[datetime] = Query(None, description="종료 날짜 (ISO 8601 형식)")
):
    """
    분석 결과 목록을 조회합니다.
    
    페이지네이션을 지원하며, 다양한 조건으로 필터링할 수 있습니다.
    결과는 분석 날짜 기준 내림차순으로 정렬됩니다.
    """
    try:
        collection = get_analysis_collection()
        
        # 필터 조건 구성
        filter_dict = {}
        
        if ne_id:
            filter_dict["ne_id"] = ne_id
        if cell_id:
            filter_dict["cell_id"] = cell_id
        if status:
            filter_dict["status"] = status
        if date_from or date_to:
            filter_dict["analysis_date"] = {}
            if date_from:
                filter_dict["analysis_date"]["$gte"] = date_from
            if date_to:
                filter_dict["analysis_date"]["$lte"] = date_to
        
        # 요청 컨텍스트 로거 생성
        req_logger = create_request_context_logger("app.analysis.list")
        
        # 페이지네이션 검증
        try:
            page, size = validate_pagination_params(page, size)
        except ValidationError as e:
            req_logger.warning(f"페이지네이션 매개변수 검증 실패: {e.message}")
            raise InvalidAnalysisDataException(e.message)
        
        req_logger.info("분석 결과 목록 조회 시작", extra={
            "page": page,
            "size": size,
            "filters": filter_dict,
            "filter_count": len([k for k, v in filter_dict.items() if v is not None])
        })
        
        # 캐시 확인
        cache_manager = await get_cache_manager()
        cached_results = await cache_manager.get_cached_analysis_results(
            filter_dict, page, size
        )
        
        if cached_results:
            req_logger.info("캐시된 분석 결과 반환", extra={
                "page": page,
                "size": size,
                "cache_hit": True,
                "result_count": len(cached_results.get("results", []))
            })
            return cached_results
        
        # 전체 개수 조회
        total_count = await collection.count_documents(filter_dict)
        
        # 페이지네이션 계산
        skip = (page - 1) * size
        has_next = (skip + size) < total_count
        
        # 데이터 조회 (요약 정보만)
        # 기존 camelCase 저장 레거시 문서와 호환을 위해 양쪽 키를 모두 조회
        projection = {
            "_id": 1,
            "analysis_date": 1,
            "analysisDate": 1,
            "ne_id": 1,
            "neId": 1,
            "cell_id": 1,
            "cellId": 1,
            "status": 1,
            "results": 1,
            "analysis_type": 1,
            "analysisType": 1,
            # include overview by default, exclude raw compact
            "results_overview": 1,
            "resultsOverview": 1,
        }
        
        cursor = collection.find(filter_dict, projection)
        cursor = cursor.sort("analysis_date", -1).skip(skip).limit(size)
        
        documents = await cursor.to_list(length=size)
        
        # 응답 데이터 직접 구성 (Pydantic alias 문제 우회)
        items = []
        for doc in documents:
            # results_count 계산
            results_count = len(doc.get("results", []))

            # 레거시 호환: snake_case 우선, 없으면 camelCase 폴백
            analysis_date_val = doc.get("analysis_date") or doc.get("analysisDate")
            ne_id_val = doc.get("ne_id") or doc.get("neId")
            cell_id_val = doc.get("cell_id") or doc.get("cellId")
            analysis_type_val = doc.get("analysis_type") or doc.get("analysisType")

            # 직접 dict로 구성 (FastAPI가 JSON 직렬화 시 필드명 사용)
            item_dict = {
                "id": str(doc["_id"]),  # ✅ _id → id로 변환
                "analysisDate": analysis_date_val.isoformat() if analysis_date_val else None,
                "neId": ne_id_val,
                "cellId": cell_id_val,
                "status": doc.get("status"),
                "results_count": results_count,
                "analysis_type": analysis_type_val,
                "results_overview": doc.get("results_overview") or doc.get("resultsOverview"),
            }
            
            items.append(item_dict)
        
        # 응답 데이터 구성
        response_data = AnalysisResultListResponse(
            items=items,
            total=total_count,
            page=page,
            size=size,
            has_next=has_next
        )
        
        # 결과를 캐시에 저장
        await cache_manager.cache_analysis_results(filter_dict, page, size, response_data.dict())
        
        req_logger.info("분석 결과 목록 조회 완료", extra={
            "returned_items": len(items),
            "total_count": total_count,
            "page": page,
            "size": size,
            "has_next": has_next,
            "applied_filters": len([k for k, v in filter_dict.items() if v is not None]),
            "cached": True
        })
        
        return response_data
        
    except PyMongoError as e:
        logger.error(f"MongoDB 오류: {e}")
        raise DatabaseConnectionException(f"Database query failed: {str(e)}")
        
    except Exception as e:
        logger.error(f"분석 결과 목록 조회 중 오류: {e}")
        raise DatabaseConnectionException(f"Failed to retrieve analysis results: {str(e)}")


@router.get(
    "/{result_id}",
    response_model=AnalysisResultResponse,
    summary="분석 결과 상세 조회",
    description="특정 ID의 분석 결과를 상세 조회합니다."
)
async def get_analysis_result(result_id: PyObjectId, includeRaw: bool = Query(False, description="압축 원본(analysis_raw_compact) 포함 여부")):
    """
    특정 ID의 분석 결과를 상세 조회합니다.
    
    - **result_id**: 조회할 분석 결과의 ObjectId
    """
    try:
        collection = get_analysis_collection()
        
        # 요청 컨텍스트 로거 생성
        req_logger = create_request_context_logger("app.analysis.detail")
        
        req_logger.info("분석 결과 상세 조회 시작", extra={
            "result_id": str(result_id),
            "include_raw": includeRaw
        })
        
        # 캐시 확인 (includeRaw=True인 경우는 캐시하지 않음)
        cache_manager = await get_cache_manager()
        if not includeRaw:
            cached_result = await cache_manager.get_cached_analysis_detail(str(result_id))
            if cached_result:
                req_logger.info("캐시된 분석 결과 상세 반환", extra={
                    "result_id": str(result_id),
                    "cache_hit": True
                })
                return AnalysisResultModel.parse_obj(cached_result)
        
        # 문서 조회
        document = await collection.find_one({"_id": result_id})
        
        if not document:
            raise AnalysisResultNotFoundException(str(result_id))
        
        # 최적화된 문서인 경우 복원
        if "_optimization" in document:
            req_logger.info("최적화된 문서 복원 시작", extra={
                "result_id": str(result_id)
            })
            
            try:
                document = await restore_analysis_result(document)
                req_logger.info("문서 복원 완료", extra={
                    "result_id": str(result_id)
                })
            except Exception as e:
                req_logger.warning(f"문서 복원 실패, 원본 사용: {e}", extra={
                    "error_type": type(e).__name__,
                    "result_id": str(result_id)
                })

        # 레거시 키 정규화 (camelCase → snake_case 우선)
        req_logger.info("레거시 키 정규화 시작", extra={
            "result_id": str(result_id),
            "document_keys": list(document.keys())
        })
        document = _normalize_legacy_keys(document)

        # includeRaw=false인 경우 압축 원본 제외하여 경량 응답 (두 케이스 모두 제거)
        if not includeRaw:
            document.pop("analysis_raw_compact", None)
            document.pop("analysisRawCompact", None)

        # 응답 모델로 변환
        req_logger.info("AnalysisResultModel 변환 시작", extra={
            "result_id": str(result_id),
            "document_keys_after_normalize": list(document.keys())
        })
        analysis_model = AnalysisResultModel.from_mongo(document)
        
        # 캐시에 저장 (includeRaw=False인 경우만)
        if not includeRaw:
            await cache_manager.cache_analysis_detail(str(result_id), analysis_model.dict())
        
        req_logger.info("분석 결과 상세 조회 완료", extra={
            "result_id": str(result_id),
            "ne_id": analysis_model.ne_id,
            "cell_id": analysis_model.cell_id,
            "analysis_type": analysis_model.analysis_type,
            "include_raw": includeRaw,
            "cached": not includeRaw
        })
        
        return AnalysisResultResponse(
            message="Analysis result retrieved successfully",
            data=analysis_model
        )
        
    except AnalysisResultNotFoundException:
        # 이미 정의된 예외는 그대로 전파
        raise
        
    except PyMongoError as e:
        logger.error(f"MongoDB 오류: {e}")
        raise DatabaseConnectionException(f"Database query failed: {str(e)}")
        
    except Exception as e:
        logger.error(f"분석 결과 조회 중 오류: {e}")
        raise DatabaseConnectionException(f"Failed to retrieve analysis result: {str(e)}")


@router.put(
    "/{result_id}",
    response_model=AnalysisResultResponse,
    summary="분석 결과 수정",
    description="특정 ID의 분석 결과를 수정합니다."
)
async def update_analysis_result(
    result_id: PyObjectId,
    update_data: AnalysisResultUpdate = Body(...)
):
    """
    특정 ID의 분석 결과를 수정합니다.
    
    - **result_id**: 수정할 분석 결과의 ObjectId
    - **update_data**: 수정할 데이터 (일부 필드만 포함 가능)
    """
    try:
        collection = get_analysis_collection()
        
        logger.info(f"분석 결과 수정 시도: ID={result_id}")
        
        # 기존 문서 존재 확인
        existing = await collection.find_one({"_id": result_id})
        if not existing:
            raise AnalysisResultNotFoundException(str(result_id))
        
        # 수정 데이터 준비
        update_dict = update_data.model_dump(by_alias=True, exclude_unset=True)
        
        if update_dict:
            # metadata 업데이트
            update_dict["metadata.updated_at"] = datetime.utcnow()
            
            # 문서 업데이트
            await collection.update_one(
                {"_id": result_id},
                {"$set": update_dict}
            )
            
            logger.info(f"분석 결과 수정 완료: ID={result_id}")
        
        # 수정된 문서 조회
        updated_document = await collection.find_one({"_id": result_id})
        analysis_model = AnalysisResultModel.from_mongo(updated_document)
        
        return AnalysisResultResponse(
            message="Analysis result updated successfully",
            data=analysis_model
        )
        
    except AnalysisResultNotFoundException:
        raise
        
    except PyMongoError as e:
        logger.error(f"MongoDB 오류: {e}")
        raise DatabaseConnectionException(f"Database update failed: {str(e)}")
        
    except Exception as e:
        logger.error(f"분석 결과 수정 중 오류: {e}")
        raise InvalidAnalysisDataException(f"Failed to update analysis result: {str(e)}")


@router.delete(
    "/{result_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="분석 결과 삭제",
    description="특정 ID의 분석 결과를 삭제합니다."
)
async def delete_analysis_result(result_id: PyObjectId):
    """
    특정 ID의 분석 결과를 삭제합니다.
    
    - **result_id**: 삭제할 분석 결과의 ObjectId
    """
    try:
        collection = get_analysis_collection()
        
        logger.info(f"분석 결과 삭제 시도: ID={result_id}")
        
        # 문서 삭제
        delete_result = await collection.delete_one({"_id": result_id})
        
        if delete_result.deleted_count == 0:
            raise AnalysisResultNotFoundException(str(result_id))
        
        logger.info(f"분석 결과 삭제 완료: ID={result_id}")
        
        # 204 No Content 응답 (body 없음)
        return
        
    except AnalysisResultNotFoundException:
        raise
        
    except PyMongoError as e:
        logger.error(f"MongoDB 오류: {e}")
        raise DatabaseConnectionException(f"Database delete failed: {str(e)}")
        
    except Exception as e:
        logger.error(f"분석 결과 삭제 중 오류: {e}")
        raise DatabaseConnectionException(f"Failed to delete analysis result: {str(e)}")


@router.get("/optimization/stats", summary="데이터 최적화 통계", tags=["Optimization"])
async def get_optimization_statistics():
    """
    데이터 최적화 통계 조회
    
    압축률, GridFS 사용량, 저장공간 절약 등의 통계를 제공합니다.
    """
    from datetime import datetime
    from ..middleware.request_tracing import get_current_request_id
    
    request_id = get_current_request_id()
    req_logger = create_request_context_logger("app.analysis.optimization_stats")
    
    try:
        req_logger.info("최적화 통계 조회 시작")
        
        stats = await get_optimization_stats()
        
        req_logger.info("최적화 통계 조회 완료", extra={
            "total_documents": stats.get("total_documents", 0),
            "optimized_documents": stats.get("optimized_documents", 0),
            "optimization_rate": stats.get("optimization_rate", 0)
        })
        
        return {
            "status": "success",
            "data": stats,
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id
        }
        
    except Exception as e:
        req_logger.error(f"최적화 통계 조회 실패: {e}", extra={
            "error_type": type(e).__name__
        }, exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"최적화 통계 조회 실패: {str(e)}",
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }
        )


@router.post("/optimization/cleanup", summary="최적화 정리 작업", tags=["Optimization"])
async def cleanup_optimization():
    """
    최적화 관련 정리 작업 수행
    
    고아 GridFS 파일 정리 등의 유지보수 작업을 수행합니다.
    """
    from datetime import datetime
    from ..middleware.request_tracing import get_current_request_id
    from ..utils.data_optimization import get_data_optimizer
    
    request_id = get_current_request_id()
    req_logger = create_request_context_logger("app.analysis.optimization_cleanup")
    
    try:
        req_logger.info("최적화 정리 작업 시작")
        
        optimizer = await get_data_optimizer()
        cleanup_result = await optimizer.cleanup_orphaned_gridfs_files()
        
        req_logger.info("최적화 정리 작업 완료", extra={
            "deleted_files": cleanup_result.get("deleted_files", 0),
            "orphaned_files": cleanup_result.get("orphaned_files", 0)
        })
        
        # 정리 후 업데이트된 통계
        updated_stats = await get_optimization_stats()
        
        return {
            "status": "success",
            "cleanup_result": cleanup_result,
            "updated_stats": updated_stats,
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id
        }
        
    except Exception as e:
        req_logger.error(f"최적화 정리 작업 실패: {e}", extra={
            "error_type": type(e).__name__
        }, exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"최적화 정리 작업 실패: {str(e)}",
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get("/cache/stats", summary="캐시 통계", tags=["Performance"])
async def get_cache_statistics():
    """
    캐시 시스템 통계 조회
    
    Redis와 메모리 캐시의 성능 지표를 제공합니다.
    """
    from datetime import datetime
    from ..middleware.request_tracing import get_current_request_id
    
    request_id = get_current_request_id()
    req_logger = create_request_context_logger("app.analysis.cache_stats")
    
    try:
        req_logger.info("캐시 통계 조회 시작")
        
        cache_manager = await get_cache_manager()
        stats = await cache_manager.get_cache_stats()
        
        req_logger.info("캐시 통계 조회 완료", extra={
            "memory_cache_size": stats.get("memory_cache", {}).get("size", 0),
            "redis_available": stats.get("redis_available", False)
        })
        
        return {
            "status": "success",
            "data": stats,
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id
        }
        
    except Exception as e:
        req_logger.error(f"캐시 통계 조회 실패: {e}", extra={
            "error_type": type(e).__name__
        }, exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"캐시 통계 조회 실패: {str(e)}",
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }
        )


@router.post("/cache/clear", summary="캐시 정리", tags=["Performance"])
async def clear_cache(pattern: str = "*"):
    """
    캐시 정리 작업
    
    지정된 패턴의 캐시를 삭제합니다.
    """
    from datetime import datetime
    from ..middleware.request_tracing import get_current_request_id
    
    request_id = get_current_request_id()
    req_logger = create_request_context_logger("app.analysis.cache_clear")
    
    try:
        req_logger.info("캐시 정리 시작", extra={"pattern": pattern})
        
        cache_manager = await get_cache_manager()
        deleted_count = await cache_manager.delete_pattern(pattern)
        
        req_logger.info("캐시 정리 완료", extra={
            "pattern": pattern,
            "deleted_count": deleted_count
        })
        
        return {
            "status": "success",
            "message": f"캐시 정리 완료: {deleted_count}개 항목 삭제",
            "deleted_count": deleted_count,
            "pattern": pattern,
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id
        }
        
    except Exception as e:
        req_logger.error(f"캐시 정리 실패: {e}", extra={
            "error_type": type(e).__name__,
            "pattern": pattern
        }, exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"캐시 정리 실패: {str(e)}",
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get(
    "/stats/summary",
    summary="분석 결과 통계 요약",
    description="분석 결과의 전체 통계를 조회합니다."
)
async def get_analysis_summary():
    """
    분석 결과의 전체 통계를 조회합니다.
    
    각 상태별 개수, 최근 분석 날짜 등의 요약 정보를 제공합니다.
    """
    try:
        collection = get_analysis_collection()
        
        logger.info("분석 결과 통계 요약 조회")
        
        # 집계 파이프라인 구성
        pipeline = [
            {
                "$group": {
                    "_id": "$status",
                    "count": {"$sum": 1},
                    "latest_date": {"$max": "$analysis_date"},
                    "oldest_date": {"$min": "$analysis_date"}
                }
            },
            {
                "$sort": {"_id": 1}
            }
        ]
        
        # 집계 실행
        status_stats = await collection.aggregate(pipeline).to_list(length=None)
        
        # 전체 개수
        total_count = await collection.count_documents({})
        
        # 최근 분석 결과 (상위 5개)
        recent_results = await collection.find(
            {},
            {"_id": 1, "ne_id": 1, "cell_id": 1, "analysis_date": 1, "status": 1}
        ).sort("analysis_date", -1).limit(5).to_list(length=5)
        
        summary = {
            "total_count": total_count,
            "status_breakdown": {stat["_id"]: stat["count"] for stat in status_stats},
            "date_range": {
                "earliest": min((stat["oldest_date"] for stat in status_stats), default=None),
                "latest": max((stat["latest_date"] for stat in status_stats), default=None)
            },
            "recent_results": recent_results
        }
        
        logger.info("분석 결과 통계 요약 조회 완료")
        
        return {
            "success": True,
            "message": "Analysis summary retrieved successfully",
            "data": summary
        }
        
    except PyMongoError as e:
        logger.error(f"MongoDB 오류: {e}")
        raise DatabaseConnectionException(f"Database aggregation failed: {str(e)}")
        
    except Exception as e:
        logger.error(f"분석 결과 통계 조회 중 오류: {e}")
        raise DatabaseConnectionException(f"Failed to retrieve analysis summary: {str(e)}")

# 라우터 export (main.py에서 사용)
__all__ = ["router", "test_router"]
