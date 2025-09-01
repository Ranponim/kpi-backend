"""
데이터베이스 최적화 유틸리티

MongoDB 컬렉션에 적절한 인덱스를 생성하고 쿼리 성능을 최적화합니다.
"""

import logging
from typing import List, Dict, Any
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import IndexModel, ASCENDING, DESCENDING, TEXT

logger = logging.getLogger(__name__)

async def create_analysis_indexes(db: AsyncIOMotorDatabase):
    """
    분석 결과 컬렉션에 최적화된 인덱스 생성
    
    주요 쿼리 패턴에 맞춘 복합 인덱스를 생성합니다.
    """
    collection = db.analysis_results
    
    # 인덱스 정의
    indexes = [
        # 1. 기본 조회용 인덱스 (날짜 역순)
        IndexModel([("analysis_date", DESCENDING)], name="idx_analysis_date_desc"),
        
        # 2. NE/Cell 기반 조회용 복합 인덱스
        IndexModel([
            ("ne_id", ASCENDING),
            ("cell_id", ASCENDING),
            ("analysis_date", DESCENDING)
        ], name="idx_ne_cell_date"),
        
        # 3. 상태별 조회용 인덱스
        IndexModel([
            ("status", ASCENDING),
            ("analysis_date", DESCENDING)
        ], name="idx_status_date"),
        
        # 4. 기간별 검색용 인덱스
        IndexModel([
            ("analysis_date", ASCENDING),
            ("status", ASCENDING)
        ], name="idx_date_status"),
        
        # 5. 텍스트 검색용 인덱스 (NE/Cell ID)
        IndexModel([
            ("ne_id", TEXT),
            ("cell_id", TEXT)
        ], name="idx_text_search"),
        
        # 6. KPI 성능 조회용 인덱스
        IndexModel([
            ("results.kpi_name", ASCENDING),
            ("results.status", ASCENDING),
            ("analysis_date", DESCENDING)
        ], name="idx_kpi_performance"),
        
        # 7. 페이지네이션 최적화 인덱스
        IndexModel([("created_at", DESCENDING)], name="idx_created_at_desc"),
        
        # 8. TTL 인덱스 (6개월 후 자동 삭제)
        IndexModel([("analysis_date", ASCENDING)], 
                  name="idx_analysis_date_ttl",
                  expireAfterSeconds=15552000)  # 6개월 = 180일 * 24시간 * 60분 * 60초
    ]
    
    try:
        # 기존 인덱스 정보 조회
        existing_indexes = await collection.list_indexes().to_list(length=None)
        existing_names = {idx.get('name') for idx in existing_indexes}
        
        # 새로운 인덱스만 생성
        new_indexes = [idx for idx in indexes if idx.document['name'] not in existing_names]
        
        if new_indexes:
            await collection.create_indexes(new_indexes)
            logger.info(f"분석 결과 컬렉션에 {len(new_indexes)}개 인덱스 생성 완료")
            
            for idx in new_indexes:
                logger.info(f"  - {idx.document['name']}: {idx.document['key']}")
        else:
            logger.info("분석 결과 컬렉션: 모든 인덱스가 이미 존재")
            
    except Exception as e:
        logger.error(f"분석 결과 인덱스 생성 실패: {e}")
        raise

async def create_preference_indexes(db: AsyncIOMotorDatabase):
    """
    사용자 설정 컬렉션에 인덱스 생성
    """
    collection = db.user_preferences
    
    # 인덱스 생성 전에 중복 데이터 정리
    try:
        # user_id가 null인 중복 데이터 확인 및 정리
        null_user_docs = await collection.find({"user_id": None}).to_list(length=None)
        if len(null_user_docs) > 1:
            logger.warning(f"user_id가 null인 중복 데이터 {len(null_user_docs)}개 발견, 첫 번째만 유지하고 나머지 삭제")
            # 첫 번째 문서만 유지하고 나머지 삭제
            docs_to_delete = null_user_docs[1:]
            for doc in docs_to_delete:
                await collection.delete_one({"_id": doc["_id"]})
            logger.info(f"중복 데이터 {len(docs_to_delete)}개 삭제 완료")
        
        # user_id가 null이 아닌 중복 데이터도 확인
        pipeline = [
            {"$match": {"user_id": {"$ne": None}}},
            {"$group": {"_id": "$user_id", "count": {"$sum": 1}, "docs": {"$push": "$_id"}}},
            {"$match": {"count": {"$gt": 1}}}
        ]
        
        duplicates = await collection.aggregate(pipeline).to_list(length=None)
        for dup in duplicates:
            user_id = dup["_id"]
            doc_ids = dup["docs"]
            logger.warning(f"user_id '{user_id}'의 중복 데이터 {len(doc_ids)}개 발견, 첫 번째만 유지하고 나머지 삭제")
            # 첫 번째 문서만 유지하고 나머지 삭제
            docs_to_delete = doc_ids[1:]
            for doc_id in docs_to_delete:
                await collection.delete_one({"_id": doc_id})
            logger.info(f"user_id '{user_id}'의 중복 데이터 {len(docs_to_delete)}개 삭제 완료")
            
    except Exception as e:
        logger.error(f"중복 데이터 정리 중 오류 발생: {e}")
        # 중복 데이터 정리 실패해도 인덱스 생성은 시도
    
    indexes = [
        # 1. 사용자 ID 기본 인덱스 (유니크)
        IndexModel([("user_id", ASCENDING)], unique=True, name="idx_user_id_unique"),
        
        # 2. 수정 날짜 인덱스
        IndexModel([("updated_at", DESCENDING)], name="idx_updated_at_desc"),
        
        # 3. 설정 타입별 인덱스
        IndexModel([
            ("dashboard_settings.theme", ASCENDING),
            ("user_id", ASCENDING)
        ], name="idx_theme_user")
    ]
    
    try:
        existing_indexes = await collection.list_indexes().to_list(length=None)
        existing_names = {idx.get('name') for idx in existing_indexes}
        
        new_indexes = [idx for idx in indexes if idx.document['name'] not in existing_names]
        
        if new_indexes:
            await collection.create_indexes(new_indexes)
            logger.info(f"사용자 설정 컬렉션에 {len(new_indexes)}개 인덱스 생성 완료")
        else:
            logger.info("사용자 설정 컬렉션: 모든 인덱스가 이미 존재")
            
    except Exception as e:
        logger.error(f"사용자 설정 인덱스 생성 실패: {e}")
        raise

async def create_statistics_indexes(db: AsyncIOMotorDatabase):
    """
    통계 데이터 컬렉션에 인덱스 생성
    """
    collection = db.kpi_statistics
    
    indexes = [
        # 1. 날짜 기간 조회용 인덱스
        IndexModel([
            ("date", ASCENDING),
            ("ne_id", ASCENDING)
        ], name="idx_date_ne"),
        
        # 2. KPI 타입별 조회 인덱스
        IndexModel([
            ("kpi_type", ASCENDING),
            ("date", DESCENDING)
        ], name="idx_kpi_type_date"),
        
        # 3. 복합 조회 인덱스
        IndexModel([
            ("ne_id", ASCENDING),
            ("kpi_type", ASCENDING),
            ("date", DESCENDING)
        ], name="idx_ne_kpi_date")
    ]
    
    try:
        existing_indexes = await collection.list_indexes().to_list(length=None)
        existing_names = {idx.get('name') for idx in existing_indexes}
        
        new_indexes = [idx for idx in indexes if idx.document['name'] not in existing_names]
        
        if new_indexes:
            await collection.create_indexes(new_indexes)
            logger.info(f"통계 데이터 컬렉션에 {len(new_indexes)}개 인덱스 생성 완료")
        else:
            logger.info("통계 데이터 컬렉션: 모든 인덱스가 이미 존재")
            
    except Exception as e:
        logger.error(f"통계 데이터 인덱스 생성 실패: {e}")
        raise

async def create_master_data_indexes(db: AsyncIOMotorDatabase):
    """
    마스터 데이터 컬렉션에 인덱스 생성
    """
    # PEG 마스터 데이터
    peg_collection = db.peg_master
    peg_indexes = [
        IndexModel([("peg_id", ASCENDING)], unique=True, name="idx_peg_id_unique"),
        IndexModel([("peg_name", TEXT)], name="idx_peg_name_text"),
        IndexModel([("region", ASCENDING)], name="idx_region")
    ]
    
    # Cell 마스터 데이터
    cell_collection = db.cell_master
    cell_indexes = [
        IndexModel([("cell_id", ASCENDING)], unique=True, name="idx_cell_id_unique"),
        IndexModel([("peg_id", ASCENDING)], name="idx_cell_peg"),
        IndexModel([("cell_name", TEXT)], name="idx_cell_name_text")
    ]
    
    # PEG 인덱스 생성
    try:
        existing_indexes = await peg_collection.list_indexes().to_list(length=None)
        existing_names = {idx.get('name') for idx in existing_indexes}
        new_indexes = [idx for idx in peg_indexes if idx.document['name'] not in existing_names]
        
        if new_indexes:
            await peg_collection.create_indexes(new_indexes)
            logger.info(f"PEG 마스터 컬렉션에 {len(new_indexes)}개 인덱스 생성 완료")
        else:
            logger.info("PEG 마스터 컬렉션: 모든 인덱스가 이미 존재")
    except Exception as e:
        logger.error(f"PEG 마스터 인덱스 생성 실패: {e}")
    
    # Cell 인덱스 생성
    try:
        existing_indexes = await cell_collection.list_indexes().to_list(length=None)
        existing_names = {idx.get('name') for idx in existing_indexes}
        new_indexes = [idx for idx in cell_indexes if idx.document['name'] not in existing_names]
        
        if new_indexes:
            await cell_collection.create_indexes(new_indexes)
            logger.info(f"Cell 마스터 컬렉션에 {len(new_indexes)}개 인덱스 생성 완료")
        else:
            logger.info("Cell 마스터 컬렉션: 모든 인덱스가 이미 존재")
    except Exception as e:
        logger.error(f"Cell 마스터 인덱스 생성 실패: {e}")

async def optimize_all_collections(db: AsyncIOMotorDatabase):
    """
    모든 컬렉션에 최적화된 인덱스 생성
    """
    logger.info("🚀 데이터베이스 인덱스 최적화 시작...")
    
    try:
        # 각 컬렉션별 인덱스 생성
        await create_analysis_indexes(db)
        await create_preference_indexes(db)
        await create_statistics_indexes(db)
        await create_master_data_indexes(db)
        
        logger.info("✅ 모든 컬렉션 인덱스 최적화 완료")
        
        # 인덱스 통계 출력
        await print_index_statistics(db)
        
    except Exception as e:
        logger.error(f"❌ 인덱스 최적화 실패: {e}")
        raise

async def print_index_statistics(db: AsyncIOMotorDatabase):
    """
    각 컬렉션의 인덱스 통계 출력
    """
    collections = [
        "analysis_results",
        "user_preferences", 
        "kpi_statistics",
        "peg_master",
        "cell_master"
    ]
    
    logger.info("📊 인덱스 통계:")
    
    for collection_name in collections:
        try:
            collection = db[collection_name]
            indexes = await collection.list_indexes().to_list(length=None)
            index_count = len(indexes)
            
            logger.info(f"  {collection_name}: {index_count}개 인덱스")
            
            for idx in indexes:
                name = idx.get('name', 'unnamed')
                key = idx.get('key', {})
                logger.debug(f"    - {name}: {dict(key)}")
                
        except Exception as e:
            logger.warning(f"  {collection_name}: 통계 조회 실패 - {e}")

async def analyze_query_performance(db: AsyncIOMotorDatabase, collection_name: str, query: Dict[str, Any]):
    """
    특정 쿼리의 성능 분석 (explain 사용)
    """
    try:
        collection = db[collection_name]
        
        # 쿼리 실행 계획 분석
        explain_result = await collection.find(query).explain()
        
        execution_stats = explain_result.get('executionStats', {})
        
        analysis = {
            "query": query,
            "collection": collection_name,
            "execution_time_ms": execution_stats.get('executionTimeMillis', 0),
            "total_docs_examined": execution_stats.get('totalDocsExamined', 0),
            "total_docs_returned": execution_stats.get('totalDocsReturned', 0),
            "index_used": execution_stats.get('indexName'),
            "winning_plan": explain_result.get('queryPlanner', {}).get('winningPlan', {})
        }
        
        # 성능 평가
        if analysis['total_docs_examined'] > analysis['total_docs_returned'] * 10:
            analysis['performance_warning'] = "인덱스 최적화 필요 - 너무 많은 문서를 검사함"
            
        return analysis
        
    except Exception as e:
        logger.error(f"쿼리 성능 분석 실패: {e}")
        return {"error": str(e)}
