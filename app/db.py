"""
데이터베이스 연결 및 설정

이 모듈은 MongoDB 연결을 관리하고 데이터베이스 초기화를 담당합니다.
"""

import os
import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo import ASCENDING, DESCENDING
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logger = logging.getLogger(__name__)

# 환경 변수에서 MongoDB 설정 읽기
MONGO_URL = os.getenv("MONGO_URL", "mongodb://mongo:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "kpi")

# 글로벌 변수들
client: AsyncIOMotorClient = None
db: AsyncIOMotorDatabase = None
analysis_collection: AsyncIOMotorCollection = None
preference_collection: AsyncIOMotorCollection = None


async def connect_to_mongo():
    """
    MongoDB에 연결하고 데이터베이스 및 컬렉션을 초기화합니다.
    
    이 함수는 애플리케이션 시작 시 한 번 호출되어야 합니다.
    """
    global client, db, analysis_collection, preference_collection
    
    try:
        logger.info(f"MongoDB 연결 시도: {MONGO_URL}")
        
        # MongoDB 클라이언트 생성
        client = AsyncIOMotorClient(MONGO_URL)
        
        # 연결 테스트
        await client.admin.command('ping')
        logger.info("MongoDB 연결 성공")
        
        # 데이터베이스 및 컬렉션 설정
        db = client[MONGO_DB_NAME]
        analysis_collection = db["analysis_results"]
        preference_collection = db["user_preferences"]
        
        # 인덱스 최적화 수행
        from .utils.db_optimization import optimize_all_collections
        await optimize_all_collections(db)
        
        logger.info(f"데이터베이스 '{MONGO_DB_NAME}' 초기화 완료")
        
    except Exception as e:
        logger.error(f"MongoDB 연결 실패: {e}")
        raise


async def close_mongo_connection():
    """
    MongoDB 연결을 종료합니다.
    
    이 함수는 애플리케이션 종료 시 호출되어야 합니다.
    """
    global client
    
    if client:
        client.close()
        logger.info("MongoDB 연결 종료")


async def create_indexes():
    """
    필요한 인덱스들을 생성합니다.
    
    성능 최적화를 위해 자주 조회되는 필드들에 인덱스를 생성합니다.
    """
    try:
        logger.info("인덱스 생성 시작")
        
        # 1. 분석 날짜 인덱스 (내림차순) - 최신 데이터 조회 최적화
        await analysis_collection.create_index(
            [("analysis_date", DESCENDING)],
            name="idx_analysis_date_desc"
        )
        logger.info("분석 날짜 인덱스 생성 완료")
        
        # 2. NE ID 인덱스 - 특정 네트워크 장비 조회 최적화
        await analysis_collection.create_index(
            [("ne_id", ASCENDING)],
            name="idx_ne_id"
        )
        logger.info("NE ID 인덱스 생성 완료")
        
        # 3. Cell ID 인덱스 - 특정 셀 조회 최적화
        await analysis_collection.create_index(
            [("cell_id", ASCENDING)],
            name="idx_cell_id"
        )
        logger.info("Cell ID 인덱스 생성 완료")
        
        # 4. 복합 인덱스 (NE ID + Cell ID + 분석 날짜) - 필터링 최적화
        await analysis_collection.create_index(
            [
                ("ne_id", ASCENDING),
                ("cell_id", ASCENDING),
                ("analysis_date", DESCENDING)
            ],
            name="idx_ne_cell_date"
        )
        logger.info("복합 인덱스 생성 완료")
        
        # 5. 상태 인덱스 - 상태별 조회 최적화
        await analysis_collection.create_index(
            [("status", ASCENDING)],
            name="idx_status"
        )
        logger.info("상태 인덱스 생성 완료")
        
        # 6. 분석 유형 인덱스 - metadata.analysis_type 조회 최적화
        await analysis_collection.create_index(
            [("metadata.analysis_type", ASCENDING)],
            name="idx_analysis_type"
        )
        logger.info("분석 유형 인덱스 생성 완료")
        
        # 7. 날짜 범위 조회용 복합 인덱스 (Task 46 통계 분석 최적화)
        await analysis_collection.create_index(
            [
                ("analysis_date", ASCENDING),
                ("status", ASCENDING)
            ],
            name="idx_date_status"
        )
        logger.info("날짜-상태 복합 인덱스 생성 완료")
        
        # 8. User Preferences 컬렉션 인덱스들
        logger.info("사용자 설정 인덱스 생성 시작")
        
        # user_id 인덱스 (고유 인덱스)
        await preference_collection.create_index(
            [("user_id", ASCENDING)],
            name="idx_user_id",
            unique=True
        )
        logger.info("사용자 ID 고유 인덱스 생성 완료")
        
        # 업데이트 시간 인덱스
        await preference_collection.create_index(
            [("metadata.updated_at", DESCENDING)],
            name="idx_preference_updated_at"
        )
        logger.info("설정 업데이트 시간 인덱스 생성 완료")
        
        # 생성 시간 인덱스
        await preference_collection.create_index(
            [("metadata.created_at", DESCENDING)],
            name="idx_preference_created_at"
        )
        logger.info("설정 생성 시간 인덱스 생성 완료")
        
        # 테마별 인덱스 (통계 목적)
        await preference_collection.create_index(
            [("theme", ASCENDING)],
            name="idx_theme"
        )
        logger.info("테마 인덱스 생성 완료")
        
        # 언어별 인덱스 (통계 목적)
        await preference_collection.create_index(
            [("language", ASCENDING)],
            name="idx_language"
        )
        logger.info("언어 인덱스 생성 완료")
        
        logger.info("모든 인덱스 생성 완료")
        
    except Exception as e:
        logger.error(f"인덱스 생성 중 오류 발생: {e}")
        # 인덱스 생성 실패는 치명적이지 않으므로 애플리케이션은 계속 실행
        pass


def get_analysis_collection() -> AsyncIOMotorCollection:
    """
    분석 결과 컬렉션을 반환합니다.
    
    Returns:
        AsyncIOMotorCollection: analysis_results 컬렉션
        
    Raises:
        RuntimeError: 데이터베이스가 초기화되지 않은 경우
    """
    if analysis_collection is None:
        raise RuntimeError("데이터베이스가 초기화되지 않았습니다. connect_to_mongo()를 먼저 호출하세요.")
    
    return analysis_collection


def get_preference_collection() -> AsyncIOMotorCollection:
    """
    사용자 설정 컬렉션을 반환합니다.
    
    Returns:
        AsyncIOMotorCollection: user_preferences 컬렉션
        
    Raises:
        RuntimeError: 데이터베이스가 초기화되지 않은 경우
    """
    if preference_collection is None:
        raise RuntimeError("데이터베이스가 초기화되지 않았습니다. connect_to_mongo()를 먼저 호출하세요.")
    
    return preference_collection


def get_database() -> AsyncIOMotorDatabase:
    """
    데이터베이스 인스턴스를 반환합니다.
    
    Returns:
        AsyncIOMotorDatabase: KPI 데이터베이스
        
    Raises:
        RuntimeError: 데이터베이스가 초기화되지 않은 경우
    """
    if db is None:
        raise RuntimeError("데이터베이스가 초기화되지 않았습니다. connect_to_mongo()를 먼저 호출하세요.")
    
    return db


async def get_db_stats():
    """
    데이터베이스 통계 정보를 반환합니다.
    
    Returns:
        dict: 데이터베이스 통계 정보
    """
    if db is None:
        return {"error": "Database not initialized"}
    
    try:
        # 컬렉션 통계
        analysis_stats = await db.command("collStats", "analysis_results")
        preference_stats = await db.command("collStats", "user_preferences")
        
        # 전체 문서 수
        analysis_count = await analysis_collection.count_documents({})
        preference_count = await preference_collection.count_documents({})
        
        return {
            "database": MONGO_DB_NAME,
            "collections": {
                "analysis_results": {
                    "document_count": analysis_count,
                    "avg_obj_size": analysis_stats.get("avgObjSize", 0),
                    "storage_size": analysis_stats.get("storageSize", 0),
                    "total_index_size": analysis_stats.get("totalIndexSize", 0)
                },
                "user_preferences": {
                    "document_count": preference_count,
                    "avg_obj_size": preference_stats.get("avgObjSize", 0),
                    "storage_size": preference_stats.get("storageSize", 0),
                    "total_index_size": preference_stats.get("totalIndexSize", 0)
                }
            }
        }
    except Exception as e:
        logger.error(f"데이터베이스 통계 조회 실패: {e}")
        return {"error": str(e)}
