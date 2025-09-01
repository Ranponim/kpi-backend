"""
MongoDB 인덱스 최적화 유틸리티

Host 필터링 기능을 위한 MongoDB 인덱스를 생성하고 관리합니다.
이 모듈은 분석 결과 컬렉션의 쿼리 성능을 최적화하기 위해 사용됩니다.
"""

import logging
from typing import Dict, List, Optional
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.errors import OperationFailure

# 로깅 설정
logger = logging.getLogger(__name__)


class MongoDBIndexManager:
    """
    MongoDB 인덱스 관리 클래스
    
    Host 필터링을 포함한 분석 결과 검색 최적화를 위한
    인덱스를 생성하고 관리합니다.
    """
    
    def __init__(self, client: MongoClient, database_name: str, collection_name: str = "analysis_results"):
        """
        인덱스 매니저 초기화
        
        Args:
            client: MongoDB 클라이언트
            database_name: 데이터베이스 이름
            collection_name: 컬렉션 이름 (기본값: analysis_results)
        """
        self.client = client
        self.database = client[database_name]
        self.collection: Collection = self.database[collection_name]
        logger.info(f"MongoDB 인덱스 매니저 초기화: {database_name}.{collection_name}")
    
    def create_host_filtering_indexes(self) -> Dict[str, bool]:
        """
        Host 필터링을 위한 인덱스들을 생성합니다.
        
        Returns:
            Dict[str, bool]: 인덱스명과 생성 성공 여부
        """
        results = {}
        
        # 정의할 인덱스들
        indexes_to_create = [
            # 1. Target Scope 기본 인덱스들
            {
                "name": "idx_target_scope_host_ids",
                "keys": [("target_scope.host_ids", ASCENDING)],
                "description": "Host ID 목록 필터링용"
            },
            {
                "name": "idx_target_scope_primary_host",
                "keys": [("target_scope.primary_host", ASCENDING)],
                "description": "대표 Host ID 필터링용"
            },
            {
                "name": "idx_target_scope_combinations_host",
                "keys": [("target_scope.target_combinations.host", ASCENDING)],
                "description": "Host 조합 필터링용"
            },
            
            # 2. 복합 인덱스들 (시간 + 타겟)
            {
                "name": "idx_analysis_date_host_ids",
                "keys": [
                    ("analysis_date", DESCENDING),
                    ("target_scope.host_ids", ASCENDING)
                ],
                "description": "시간순 + Host ID 필터링용"
            },
            {
                "name": "idx_analysis_date_ne_cell_host",
                "keys": [
                    ("analysis_date", DESCENDING),
                    ("target_scope.primary_ne", ASCENDING),
                    ("target_scope.primary_cell", ASCENDING),
                    ("target_scope.primary_host", ASCENDING)
                ],
                "description": "시간순 + NE-Cell-Host 복합 필터링용"
            },
            
            # 3. 메타데이터 검색용 인덱스들
            {
                "name": "idx_filter_metadata_host_count",
                "keys": [("filter_metadata.applied_host_count", ASCENDING)],
                "description": "Host 개수별 필터링용"
            },
            {
                "name": "idx_scope_type_status",
                "keys": [
                    ("target_scope.scope_type", ASCENDING),
                    ("status", ASCENDING)
                ],
                "description": "범위 타입 + 상태별 필터링용"
            },
            
            # 4. 하위 호환성을 위한 기존 인덱스들 (강화)
            {
                "name": "idx_ne_cell_date_compat",
                "keys": [
                    ("ne_id", ASCENDING),
                    ("cell_id", ASCENDING),
                    ("analysis_date", DESCENDING)
                ],
                "description": "하위 호환성: 기존 NE-Cell 필터링 강화"
            }
        ]
        
        logger.info(f"Host 필터링용 인덱스 {len(indexes_to_create)}개 생성 시작")
        
        for index_def in indexes_to_create:
            try:
                # 인덱스 생성
                index_name = self.collection.create_index(
                    index_def["keys"],
                    name=index_def["name"],
                    background=True  # 백그라운드에서 생성하여 성능 영향 최소화
                )
                
                results[index_def["name"]] = True
                logger.info(f"✅ 인덱스 생성 성공: {index_def['name']} - {index_def['description']}")
                
            except OperationFailure as e:
                if "already exists" in str(e):
                    results[index_def["name"]] = True
                    logger.info(f"⚠️ 인덱스 이미 존재: {index_def['name']}")
                else:
                    results[index_def["name"]] = False
                    logger.error(f"❌ 인덱스 생성 실패: {index_def['name']} - {str(e)}")
            except Exception as e:
                results[index_def["name"]] = False
                logger.error(f"❌ 인덱스 생성 중 예외 발생: {index_def['name']} - {str(e)}")
        
        success_count = sum(1 for success in results.values() if success)
        logger.info(f"Host 필터링용 인덱스 생성 완료: {success_count}/{len(indexes_to_create)} 성공")
        
        return results
    
    def get_existing_indexes(self) -> List[Dict]:
        """
        현재 컬렉션의 모든 인덱스를 조회합니다.
        
        Returns:
            List[Dict]: 인덱스 정보 목록
        """
        try:
            indexes = list(self.collection.list_indexes())
            logger.info(f"현재 인덱스 {len(indexes)}개 조회 완료")
            return indexes
        except Exception as e:
            logger.error(f"인덱스 조회 실패: {str(e)}")
            return []
    
    def drop_index_by_name(self, index_name: str) -> bool:
        """
        지정된 이름의 인덱스를 삭제합니다.
        
        Args:
            index_name: 삭제할 인덱스 이름
            
        Returns:
            bool: 삭제 성공 여부
        """
        try:
            self.collection.drop_index(index_name)
            logger.info(f"✅ 인덱스 삭제 성공: {index_name}")
            return True
        except OperationFailure as e:
            if "index not found" in str(e):
                logger.warning(f"⚠️ 인덱스가 존재하지 않음: {index_name}")
                return True  # 이미 없는 것도 성공으로 간주
            else:
                logger.error(f"❌ 인덱스 삭제 실패: {index_name} - {str(e)}")
                return False
        except Exception as e:
            logger.error(f"❌ 인덱스 삭제 중 예외 발생: {index_name} - {str(e)}")
            return False
    
    def analyze_index_usage(self) -> Dict[str, Any]:
        """
        인덱스 사용률을 분석합니다.
        
        Returns:
            Dict[str, Any]: 인덱스 사용률 통계
        """
        try:
            # 컬렉션 통계 조회
            stats = self.database.command("collStats", self.collection.name)
            
            analysis = {
                "collection_name": self.collection.name,
                "document_count": stats.get("count", 0),
                "total_size": stats.get("size", 0),
                "index_count": stats.get("nindexes", 0),
                "total_index_size": stats.get("totalIndexSize", 0),
                "avg_object_size": stats.get("avgObjSize", 0)
            }
            
            logger.info(f"인덱스 사용률 분석 완료: {analysis['index_count']}개 인덱스, {analysis['document_count']}개 문서")
            return analysis
            
        except Exception as e:
            logger.error(f"인덱스 사용률 분석 실패: {str(e)}")
            return {"error": str(e)}
    
    def validate_host_query_performance(self) -> Dict[str, Any]:
        """
        Host 관련 쿼리 성능을 검증합니다.
        
        Returns:
            Dict[str, Any]: 성능 검증 결과
        """
        test_queries = [
            # 1. Host ID 목록 필터링
            {"target_scope.host_ids": {"$in": ["host01", "192.168.1.10"]}},
            
            # 2. 대표 Host 필터링
            {"target_scope.primary_host": "host01"},
            
            # 3. 시간 + Host 복합 필터링
            {
                "analysis_date": {"$gte": "2025-08-01T00:00:00Z"},
                "target_scope.host_ids": {"$in": ["host01"]}
            },
            
            # 4. NE-Cell-Host 복합 필터링
            {
                "target_scope.primary_ne": "nvgnb#10000",
                "target_scope.primary_cell": "2010",
                "target_scope.primary_host": "host01"
            }
        ]
        
        results = {}
        
        for i, query in enumerate(test_queries, 1):
            try:
                # explain() 실행하여 실행 계획 분석
                explain_result = self.collection.find(query).explain()
                
                execution_stats = explain_result.get("executionStats", {})
                results[f"query_{i}"] = {
                    "query": query,
                    "total_docs_examined": execution_stats.get("totalDocsExamined", 0),
                    "total_keys_examined": execution_stats.get("totalKeysExamined", 0),
                    "execution_time_millis": execution_stats.get("executionTimeMillis", 0),
                    "index_used": execution_stats.get("winningPlan", {}).get("inputStage", {}).get("indexName")
                }
                
                logger.info(f"쿼리 {i} 성능 검증 완료: {results[f'query_{i}']['execution_time_millis']}ms")
                
            except Exception as e:
                results[f"query_{i}"] = {"error": str(e)}
                logger.error(f"쿼리 {i} 성능 검증 실패: {str(e)}")
        
        return results


def setup_mongodb_indexes(
    connection_string: str,
    database_name: str,
    collection_name: str = "analysis_results"
) -> Dict[str, Any]:
    """
    MongoDB Host 필터링 인덱스를 설정합니다.
    
    Args:
        connection_string: MongoDB 연결 문자열
        database_name: 데이터베이스 이름
        collection_name: 컬렉션 이름
        
    Returns:
        Dict[str, Any]: 설정 결과
    """
    try:
        # MongoDB 클라이언트 생성
        client = MongoClient(connection_string)
        
        # 인덱스 매니저 생성
        index_manager = MongoDBIndexManager(client, database_name, collection_name)
        
        # 기존 인덱스 조회
        existing_indexes = index_manager.get_existing_indexes()
        
        # Host 필터링 인덱스 생성
        creation_results = index_manager.create_host_filtering_indexes()
        
        # 인덱스 사용률 분석
        usage_analysis = index_manager.analyze_index_usage()
        
        # 성능 검증
        performance_validation = index_manager.validate_host_query_performance()
        
        result = {
            "success": True,
            "existing_indexes_count": len(existing_indexes),
            "new_indexes_created": creation_results,
            "usage_analysis": usage_analysis,
            "performance_validation": performance_validation
        }
        
        logger.info("MongoDB Host 필터링 인덱스 설정 완료")
        return result
        
    except Exception as e:
        logger.error(f"MongoDB 인덱스 설정 실패: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        try:
            client.close()
        except:
            pass


if __name__ == "__main__":
    # 직접 실행 시 테스트
    import os
    
    # 환경변수에서 MongoDB 연결 정보 읽기
    mongo_url = os.getenv("MONGODB_URL", "mongodb://mongo:27017")
    db_name = os.getenv("MONGODB_DATABASE", "kpi_dashboard")
    
    print("MongoDB Host 필터링 인덱스 설정 시작...")
    result = setup_mongodb_indexes(mongo_url, db_name)
    
    if result.get("success"):
        print("✅ 인덱스 설정 성공!")
        print(f"생성된 인덱스: {result['new_indexes_created']}")
    else:
        print(f"❌ 인덱스 설정 실패: {result.get('error')}")
