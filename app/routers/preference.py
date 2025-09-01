"""
사용자 설정(Preference) API 라우터

이 모듈은 사용자 설정의 CRUD 작업과 Import/Export 기능을 위한 
API 엔드포인트들을 정의합니다.
Task 42: Backend Preference 시스템 API 및 DB 스키마 구현
"""

import logging
import json
from fastapi import APIRouter, Body, Query, HTTPException, status, UploadFile, File
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
from datetime import datetime
from pymongo.errors import DuplicateKeyError, PyMongoError
from io import StringIO

from ..db import get_preference_collection
from ..models.preference import (
    UserPreferenceModel,
    UserPreferenceCreate,
    UserPreferenceUpdate,
    UserPreferenceImportExport,
    UserPreferenceResponse,
    UserPreferenceCreateResponse,
    UserPreferenceUpdateResponse,
    PreferenceExportResponse,
    PreferenceImportResponse
)
from ..exceptions import (
    UserPreferenceNotFoundException,
    InvalidPreferenceDataException,
    DatabaseConnectionException,
    PreferenceImportException
)

# 로깅 설정
logger = logging.getLogger(__name__)

def _deep_merge_dicts(base: Dict[Any, Any], overlay: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    두 딕셔너리를 깊게 병합합니다.
    
    Args:
        base: 기본 딕셔너리 (기본값)
        overlay: 덮어씌울 딕셔너리 (기존 설정)
    
    Returns:
        Dict: 병합된 딕셔너리
    """
    result = base.copy()
    
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # 둘 다 딕셔너리인 경우 재귀적으로 병합
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            # 그렇지 않으면 overlay 값으로 덮어씌움
            result[key] = value
    
    return result


# 라우터 생성
router = APIRouter(
    prefix="/api/preference",
    tags=["User Preferences"],
    responses={
        400: {"description": "Bad Request"},
        404: {"description": "Not Found"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)


@router.get(
    "/settings",
    response_model=UserPreferenceResponse,
    summary="사용자 설정 조회",
    description="특정 사용자의 설정을 조회합니다."
)
async def get_user_preference(
    user_id: str = Query(..., description="사용자 ID")
):
    """
    특정 사용자의 설정을 조회합니다.
    
    - **userId**: 조회할 사용자의 ID
    
    사용자 설정이 없는 경우 기본 설정으로 새로 생성합니다.
    """
    try:
        collection = get_preference_collection()
        
        logger.info(f"사용자 설정 조회: user_id={user_id}")
        
        # 사용자 설정 조회 (MongoDB에서는 userId 필드로 저장됨)
        document = await collection.find_one({"userId": user_id})
        
        if not document:
            # 설정이 없으면 기본 설정으로 새로 생성
            logger.info(f"사용자 설정이 없어 기본 설정 생성: user_id={user_id}")
            
            default_preference = UserPreferenceCreate(user_id=user_id)
            preference_dict = default_preference.model_dump(by_alias=True, exclude_unset=False)
            
            # 메타데이터 설정 - metadata 딕셔너리가 없으면 생성
            if "metadata" not in preference_dict:
                preference_dict["metadata"] = {}
            preference_dict["metadata"]["created_at"] = datetime.utcnow()
            preference_dict["metadata"]["updated_at"] = datetime.utcnow()
            
            # 문서 삽입
            insert_result = await collection.insert_one(preference_dict)
            
            # 생성된 문서 조회
            document = await collection.find_one({"_id": insert_result.inserted_id})
            preference_model = UserPreferenceModel.from_mongo(document)
        else:
            logger.info(f"기존 사용자 설정 로드: user_id={user_id}")
            
            # 기존 문서에 누락된 필드가 있을 수 있으므로 기본값과 병합
            default_preference = UserPreferenceCreate(user_id=user_id)
            default_dict = default_preference.model_dump(by_alias=True, exclude_unset=False)
            
            # 기존 문서의 데이터를 기본값 위에 덮어씌움 (깊은 병합)
            merged_data = _deep_merge_dicts(default_dict, document)
            
            # 마지막 접근 시간 업데이트
            await collection.update_one(
                {"userId": user_id},
                {"$set": {"metadata.last_accessed": datetime.utcnow()}}
            )
            
            # 병합된 데이터로 모델 생성
            preference_model = UserPreferenceModel.from_mongo(merged_data)
        
        logger.info(f"사용자 설정 조회 완료: user_id={user_id}")
        
        return UserPreferenceResponse(
            message="User preference retrieved successfully",
            data=preference_model
        )
        
    except PyMongoError as e:
        logger.error(f"MongoDB 오류: {e}")
        raise DatabaseConnectionException(f"Database query failed: {str(e)}")
        
    except Exception as e:
        logger.error(f"사용자 설정 조회 중 오류: {e}")
        raise DatabaseConnectionException(f"Failed to retrieve user preference: {str(e)}")


@router.put(
    "/settings",
    response_model=UserPreferenceUpdateResponse,
    summary="사용자 설정 업데이트",
    description="특정 사용자의 설정을 업데이트합니다."
)
async def update_user_preference(
    userId: str = Query(..., description="사용자 ID"),
    update_data: UserPreferenceUpdate = Body(
        ...,
        example={
            "dashboard_settings": {
                "selected_pegs": ["RACH_SUCCESS_RATE", "RRC_SETUP_SUCCESS"],
                "selected_nes": ["eNB001"],
                "auto_refresh": True,
                "refresh_interval": 30
            },
            "theme": "dark",
            "language": "ko"
        }
    )
):
    """
    특정 사용자의 설정을 업데이트합니다.
    
    - **userId**: 업데이트할 사용자의 ID
    - **update_data**: 업데이트할 설정 데이터 (부분 업데이트 지원)
    
    설정이 없는 경우 404 오류를 반환합니다.
    """
    try:
        collection = get_preference_collection()
        
        logger.info(f"사용자 설정 업데이트 시도: userId={userId}")
        
        # 기존 설정 존재 확인
        existing = await collection.find_one({"userId": userId})
        if not existing:
            raise UserPreferenceNotFoundException(userId)
        
        # 업데이트 데이터 준비
        update_dict = update_data.model_dump(by_alias=True, exclude_unset=True)
        
        if update_dict:
            # 메타데이터 업데이트
            update_dict["metadata.updated_at"] = datetime.utcnow()
            
            # 문서 업데이트
            await collection.update_one(
                {"userId": userId},
                {"$set": update_dict}
            )
            
            logger.info(f"사용자 설정 업데이트 완료: userId={userId}")
        
        # 업데이트된 문서 조회
        updated_document = await collection.find_one({"userId": userId})
        preference_model = UserPreferenceModel.from_mongo(updated_document)
        
        return UserPreferenceUpdateResponse(
            message="User preference updated successfully",
            data=preference_model
        )
        
    except UserPreferenceNotFoundException:
        raise
        
    except PyMongoError as e:
        logger.error(f"MongoDB 오류: {e}")
        raise DatabaseConnectionException(f"Database update failed: {str(e)}")
        
    except Exception as e:
        logger.error(f"사용자 설정 업데이트 중 오류: {e}")
        raise InvalidPreferenceDataException(f"Failed to update user preference: {str(e)}")


@router.post(
    "/settings",
    response_model=UserPreferenceCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="사용자 설정 생성",
    description="새로운 사용자 설정을 생성합니다."
)
async def create_user_preference(
    preference: UserPreferenceCreate = Body(...)
):
    """
    새로운 사용자 설정을 생성합니다.
    
    - **preference**: 생성할 사용자 설정 데이터
    
    이미 동일한 userId로 설정이 있는 경우 409 오류를 반환합니다.
    """
    try:
        collection = get_preference_collection()
        
        logger.info(f"사용자 설정 생성 시도: userId={preference.userId}")
        
        # 중복 검사
        existing = await collection.find_one({"userId": preference.userId})
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"User preference for user '{preference.userId}' already exists"
            )
        
        # 데이터 준비
        preference_dict = preference.model_dump(by_alias=True, exclude_unset=True)
        
        # 메타데이터 설정
        preference_dict["metadata"]["created_at"] = datetime.utcnow()
        preference_dict["metadata"]["updated_at"] = datetime.utcnow()
        
        # 문서 삽입
        insert_result = await collection.insert_one(preference_dict)
        
        # 생성된 문서 조회
        created_document = await collection.find_one({"_id": insert_result.inserted_id})
        preference_model = UserPreferenceModel.from_mongo(created_document)
        
        logger.info(f"사용자 설정 생성 성공: userId={preference.userId}")
        
        return UserPreferenceCreateResponse(
            message="User preference created successfully",
            data=preference_model
        )
        
    except HTTPException:
        raise
        
    except PyMongoError as e:
        logger.error(f"MongoDB 오류: {e}")
        raise DatabaseConnectionException(f"Database operation failed: {str(e)}")
        
    except Exception as e:
        logger.error(f"사용자 설정 생성 중 오류: {e}")
        raise InvalidPreferenceDataException(f"Failed to create user preference: {str(e)}")


@router.get(
    "/export",
    response_model=PreferenceExportResponse,
    summary="설정 내보내기",
    description="사용자 설정을 JSON 형태로 내보냅니다."
)
async def export_user_preference(
    userId: str = Query(..., description="사용자 ID")
):
    """
    사용자 설정을 JSON 형태로 내보냅니다.
    
    - **userId**: 내보낼 사용자의 ID
    
    내보내기 가능한 설정만 포함되며, 메타데이터는 제외됩니다.
    """
    try:
        collection = get_preference_collection()
        
        logger.info(f"사용자 설정 내보내기: userId={userId}")
        
        # 사용자 설정 조회
        document = await collection.find_one({"userId": userId})
        
        if not document:
            raise UserPreferenceNotFoundException(userId)
        
        # Export 모델로 변환 (민감한 정보 제외)
        export_data = UserPreferenceImportExport(
            dashboard_settings=document.get("dashboard_settings", {}),
            statistics_settings=document.get("statistics_settings", {}),
            notification_settings=document.get("notification_settings", {}),
            theme=document.get("theme", "light"),
            language=document.get("language", "ko"),
            export_date=datetime.utcnow(),
            version="1.0"
        )
        
        # 파일명 생성
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"kpi_dashboard_preferences_{userId}_{timestamp}.json"
        
        logger.info(f"사용자 설정 내보내기 완료: userId={userId}")
        
        return PreferenceExportResponse(
            message="Preference exported successfully",
            data=export_data,
            filename=filename
        )
        
    except UserPreferenceNotFoundException:
        raise
        
    except PyMongoError as e:
        logger.error(f"MongoDB 오류: {e}")
        raise DatabaseConnectionException(f"Database query failed: {str(e)}")
        
    except Exception as e:
        logger.error(f"설정 내보내기 중 오류: {e}")
        raise DatabaseConnectionException(f"Failed to export preference: {str(e)}")


@router.post(
    "/import",
    response_model=PreferenceImportResponse,
    summary="설정 가져오기",
    description="JSON 파일에서 사용자 설정을 가져옵니다."
)
async def import_user_preference(
    userId: str = Query(..., description="사용자 ID"),
    file: UploadFile = File(..., description="가져올 JSON 설정 파일"),
    overwrite: bool = Query(False, description="기존 설정 덮어쓰기 여부")
):
    """
    JSON 파일에서 사용자 설정을 가져옵니다.
    
    - **userId**: 설정을 가져올 사용자의 ID
    - **file**: 설정이 담긴 JSON 파일
    - **overwrite**: 기존 설정 덮어쓰기 여부 (기본값: False)
    
    JSON 파일의 형식이 올바르지 않으면 400 오류를 반환합니다.
    """
    try:
        collection = get_preference_collection()
        
        logger.info(f"사용자 설정 가져오기 시도: userId={userId}, file={file.filename}")
        
        # 파일 확장자 검사
        if not file.filename.endswith('.json'):
            raise PreferenceImportException("Only JSON files are supported")
        
        # 파일 내용 읽기
        try:
            content = await file.read()
            import_data = json.loads(content.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise PreferenceImportException(f"Invalid JSON format: {str(e)}")
        except UnicodeDecodeError as e:
            raise PreferenceImportException(f"File encoding error: {str(e)}")
        
        # Import 데이터 검증
        try:
            validated_import = UserPreferenceImportExport(**import_data)
        except Exception as e:
            raise PreferenceImportException(f"Invalid preference data structure: {str(e)}")
        
        # 기존 설정 확인
        existing = await collection.find_one({"userId": userId})
        
        if existing and not overwrite:
            raise PreferenceImportException(
                "User preference already exists. Use overwrite=true to replace it."
            )
        
        # 가져올 설정 데이터 준비
        update_data = {
            "dashboard_settings": validated_import.dashboard_settings.model_dump(by_alias=True),
            "statistics_settings": validated_import.statistics_settings.model_dump(by_alias=True),
            "notification_settings": validated_import.notification_settings.model_dump(by_alias=True),
            "theme": validated_import.theme,
            "language": validated_import.language,
            "metadata.updated_at": datetime.utcnow()
        }
        
        imported_settings = []
        warnings = []
        
        if existing:
            # 기존 설정 업데이트
            await collection.update_one(
                {"userId": userId},
                {"$set": update_data}
            )
            imported_settings = ["dashboard_settings", "statistics_settings", "notification_settings", "theme", "language"]
            logger.info(f"기존 사용자 설정 업데이트 완료: userId={userId}")
        else:
            # 새 설정 생성
            new_preference = UserPreferenceCreate(
                userId=userId,
                dashboard_settings=validated_import.dashboard_settings,
                statistics_settings=validated_import.statistics_settings,
                notification_settings=validated_import.notification_settings,
                theme=validated_import.theme,
                language=validated_import.language
            )
            
            preference_dict = new_preference.model_dump(by_alias=True, exclude_unset=True)
            preference_dict["metadata"]["created_at"] = datetime.utcnow()
            preference_dict["metadata"]["updated_at"] = datetime.utcnow()
            
            await collection.insert_one(preference_dict)
            imported_settings = ["dashboard_settings", "statistics_settings", "notification_settings", "theme", "language"]
            logger.info(f"새 사용자 설정 생성 완료: userId={userId}")
        
        logger.info(f"사용자 설정 가져오기 완료: userId={userId}")
        
        return PreferenceImportResponse(
            message="Preference imported successfully",
            imported_settings=imported_settings,
            warnings=warnings
        )
        
    except (PreferenceImportException, UserPreferenceNotFoundException):
        raise
        
    except PyMongoError as e:
        logger.error(f"MongoDB 오류: {e}")
        raise DatabaseConnectionException(f"Database operation failed: {str(e)}")
        
    except Exception as e:
        logger.error(f"설정 가져오기 중 오류: {e}")
        raise PreferenceImportException(f"Failed to import preference: {str(e)}")


@router.delete(
    "/settings",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="사용자 설정 삭제",
    description="특정 사용자의 설정을 삭제합니다."
)
async def delete_user_preference(
    userId: str = Query(..., description="삭제할 사용자의 ID")
):
    """
    특정 사용자의 설정을 삭제합니다.
    
    - **userId**: 삭제할 사용자의 ID
    """
    try:
        collection = get_preference_collection()
        
        logger.info(f"사용자 설정 삭제 시도: userId={userId}")
        
        # 설정 삭제
        delete_result = await collection.delete_one({"userId": userId})
        
        if delete_result.deleted_count == 0:
            raise UserPreferenceNotFoundException(userId)
        
        logger.info(f"사용자 설정 삭제 완료: userId={userId}")
        
        # 204 No Content 응답 (body 없음)
        return
        
    except UserPreferenceNotFoundException:
        raise
        
    except PyMongoError as e:
        logger.error(f"MongoDB 오류: {e}")
        raise DatabaseConnectionException(f"Database delete failed: {str(e)}")
        
    except Exception as e:
        logger.error(f"사용자 설정 삭제 중 오류: {e}")
        raise DatabaseConnectionException(f"Failed to delete user preference: {str(e)}")


@router.get(
    "/stats/summary",
    summary="설정 통계 요약",
    description="사용자 설정의 전체 통계를 조회합니다."
)
async def get_preference_summary():
    """
    사용자 설정의 전체 통계를 조회합니다.
    
    테마별, 언어별 사용자 수 등의 통계 정보를 제공합니다.
    """
    try:
        collection = get_preference_collection()
        
        logger.info("설정 통계 요약 조회")
        
        # 집계 파이프라인 구성
        pipeline = [
            {
                "$group": {
                    "_id": {
                        "theme": "$theme",
                        "language": "$language"
                    },
                    "count": {"$sum": 1}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "theme_stats": {
                        "$push": {
                            "theme": "$_id.theme",
                            "language": "$_id.language",
                            "count": "$count"
                        }
                    },
                    "total_users": {"$sum": "$count"}
                }
            }
        ]
        
        # 집계 실행
        stats_result = await collection.aggregate(pipeline).to_list(length=1)
        
        # 전체 사용자 수
        total_count = await collection.count_documents({})
        
        # 최근 활동 사용자 (상위 5명)
        recent_users = await collection.find(
            {},
            {"userId": 1, "metadata.last_accessed": 1, "theme": 1, "language": 1}
        ).sort("metadata.last_accessed", -1).limit(5).to_list(length=5)
        
        # 테마별 통계
        theme_stats = {}
        language_stats = {}
        
        if stats_result:
            for item in stats_result[0].get("theme_stats", []):
                theme = item["theme"]
                language = item["language"]
                count = item["count"]
                
                theme_stats[theme] = theme_stats.get(theme, 0) + count
                language_stats[language] = language_stats.get(language, 0) + count
        
        summary = {
            "total_users": total_count,
            "theme_breakdown": theme_stats,
            "language_breakdown": language_stats,
            "recent_active_users": recent_users
        }
        
        logger.info("설정 통계 요약 조회 완료")
        
        return {
            "success": True,
            "message": "Preference summary retrieved successfully",
            "data": summary
        }
        
    except PyMongoError as e:
        logger.error(f"MongoDB 오류: {e}")
        raise DatabaseConnectionException(f"Database aggregation failed: {str(e)}")
        
    except Exception as e:
        logger.error(f"설정 통계 조회 중 오류: {e}")
        raise DatabaseConnectionException(f"Failed to retrieve preference summary: {str(e)}")
