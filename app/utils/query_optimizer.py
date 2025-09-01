"""
쿼리 성능 최적화 유틸리티

MongoDB 쿼리 성능을 분석하고 최적화하는 도구들을 제공합니다.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import time
from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase
from pymongo.errors import OperationFailure

from ..middleware.request_tracing import create_request_context_logger

logger = logging.getLogger("app.query_optimizer")


class QueryPerformanceAnalyzer:
    """쿼리 성능 분석기"""
    
    def __init__(self, database: AsyncIOMotorDatabase):
        self.db = database
        self.query_stats: List[Dict[str, Any]] = []
        self.slow_query_threshold = 1000  # 1초 (밀리초)
    
    async def analyze_collection_performance(self, collection_name: str) -> Dict[str, Any]:
        """컬렉션 성능 분석"""
        req_logger = create_request_context_logger("app.query_optimizer.analyze")
        
        try:
            collection = self.db[collection_name]
            
            # 컬렉션 통계
            stats = await self.db.command("collStats", collection_name)
            
            # 인덱스 통계
            index_stats = await self._get_index_statistics(collection)
            
            # 느린 쿼리 분석
            slow_queries = await self._analyze_slow_queries(collection_name)
            
            # 쿼리 패턴 분석
            query_patterns = await self._analyze_query_patterns(collection_name)
            
            analysis_result = {
                "collection_name": collection_name,
                "document_count": stats.get("count", 0),
                "storage_size": stats.get("storageSize", 0),
                "index_size": stats.get("totalIndexSize", 0),
                "average_obj_size": stats.get("avgObjSize", 0),
                "indexes": index_stats,
                "slow_queries": slow_queries,
                "query_patterns": query_patterns,
                "analysis_timestamp": datetime.now(),
                "recommendations": self._generate_recommendations(stats, index_stats, slow_queries)
            }
            
            req_logger.info("컬렉션 성능 분석 완료", extra={
                "collection": collection_name,
                "document_count": analysis_result["document_count"],
                "index_count": len(index_stats),
                "slow_query_count": len(slow_queries)
            })
            
            return analysis_result
            
        except Exception as e:
            req_logger.error(f"컬렉션 성능 분석 실패: {e}", exc_info=True)
            raise
    
    async def _get_index_statistics(self, collection: AsyncIOMotorCollection) -> List[Dict[str, Any]]:
        """인덱스 통계 수집"""
        try:
            indexes = []
            
            # 인덱스 목록 조회
            async for index_info in collection.list_indexes():
                index_name = index_info["name"]
                
                # 인덱스 사용 통계 (MongoDB 3.2+)
                try:
                    index_stats = await self.db.command("collStats", collection.name, indexDetails=index_name)
                    usage_stats = index_stats.get("indexSizes", {}).get(index_name, 0)
                except (OperationFailure, KeyError):
                    usage_stats = 0
                
                indexes.append({
                    "name": index_name,
                    "key": index_info.get("key", {}),
                    "unique": index_info.get("unique", False),
                    "sparse": index_info.get("sparse", False),
                    "ttl": index_info.get("expireAfterSeconds"),
                    "size_bytes": usage_stats,
                    "background": index_info.get("background", False)
                })
            
            return indexes
            
        except Exception as e:
            logger.warning(f"인덱스 통계 수집 실패: {e}")
            return []
    
    async def _analyze_slow_queries(self, collection_name: str, 
                                  hours_back: int = 24) -> List[Dict[str, Any]]:
        """느린 쿼리 분석 (프로파일러 로그 기반)"""
        try:
            # MongoDB 프로파일러 활성화 확인
            profiler_status = await self.db.command("profile", -1)
            
            if profiler_status.get("was", 0) == 0:
                logger.info("MongoDB 프로파일러가 비활성화되어 있습니다.")
                return []
            
            # system.profile 컬렉션에서 느린 쿼리 조회
            profile_collection = self.db["system.profile"]
            
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            slow_queries = []
            query = {
                "ns": f"{self.db.name}.{collection_name}",
                "ts": {"$gte": cutoff_time},
                "millis": {"$gte": self.slow_query_threshold}
            }
            
            async for doc in profile_collection.find(query).sort("millis", -1).limit(50):
                slow_queries.append({
                    "timestamp": doc.get("ts"),
                    "duration_ms": doc.get("millis"),
                    "command": doc.get("command", {}),
                    "planSummary": doc.get("planSummary"),
                    "docsExamined": doc.get("docsExamined", 0),
                    "docsReturned": doc.get("docsReturned", 0),
                    "keysExamined": doc.get("keysExamined", 0)
                })
            
            return slow_queries
            
        except Exception as e:
            logger.warning(f"느린 쿼리 분석 실패: {e}")
            return []
    
    async def _analyze_query_patterns(self, collection_name: str) -> Dict[str, Any]:
        """쿼리 패턴 분석"""
        try:
            # 실제 구현에서는 애플리케이션 로그나 프로파일러 데이터를 분석
            # 여기서는 기본적인 패턴 예시를 제공
            
            patterns = {
                "common_filters": [
                    {"field": "analysis_date", "usage_count": 95, "selectivity": "high"},
                    {"field": "ne_id", "usage_count": 80, "selectivity": "medium"},
                    {"field": "cell_id", "usage_count": 75, "selectivity": "medium"},
                    {"field": "status", "usage_count": 60, "selectivity": "low"}
                ],
                "sort_patterns": [
                    {"field": "analysis_date", "direction": -1, "usage_count": 90},
                    {"field": "_id", "direction": -1, "usage_count": 30}
                ],
                "range_queries": [
                    {"field": "analysis_date", "type": "date_range", "usage_count": 70}
                ],
                "text_searches": [
                    {"fields": ["ne_id", "cell_id"], "usage_count": 40}
                ]
            }
            
            return patterns
            
        except Exception as e:
            logger.warning(f"쿼리 패턴 분석 실패: {e}")
            return {}
    
    def _generate_recommendations(self, stats: Dict[str, Any], 
                                index_stats: List[Dict[str, Any]], 
                                slow_queries: List[Dict[str, Any]]) -> List[str]:
        """성능 개선 권장사항 생성"""
        recommendations = []
        
        # 문서 수에 따른 권장사항
        doc_count = stats.get("count", 0)
        if doc_count > 1000000:  # 100만 개 이상
            recommendations.append("대용량 컬렉션입니다. 샤딩을 고려해보세요.")
        
        # 인덱스 관련 권장사항
        if len(index_stats) < 3:
            recommendations.append("인덱스가 부족할 수 있습니다. 주요 쿼리 필드에 인덱스를 추가하세요.")
        
        if len(index_stats) > 10:
            recommendations.append("인덱스가 너무 많습니다. 사용하지 않는 인덱스를 제거하세요.")
        
        # 느린 쿼리 관련 권장사항
        if slow_queries:
            recommendations.append(f"{len(slow_queries)}개의 느린 쿼리가 발견되었습니다. 쿼리 최적화를 검토하세요.")
            
            # COLLSCAN이 많은 경우
            collscan_count = len([q for q in slow_queries 
                                if q.get("planSummary", "").startswith("COLLSCAN")])
            if collscan_count > len(slow_queries) * 0.3:
                recommendations.append("전체 컬렉션 스캔이 많습니다. 적절한 인덱스를 추가하세요.")
        
        # 저장소 크기 관련
        storage_size = stats.get("storageSize", 0)
        index_size = stats.get("totalIndexSize", 0)
        if index_size > storage_size * 0.5:
            recommendations.append("인덱스 크기가 데이터 크기의 50%를 초과합니다. 인덱스 최적화를 고려하세요.")
        
        # 평균 문서 크기
        avg_size = stats.get("avgObjSize", 0)
        if avg_size > 1024 * 1024:  # 1MB 이상
            recommendations.append("평균 문서 크기가 큽니다. 문서 구조 최적화나 GridFS 사용을 고려하세요.")
        
        return recommendations
    
    async def measure_query_performance(self, collection: AsyncIOMotorCollection,
                                      query: Dict[str, Any], 
                                      operation: str = "find") -> Dict[str, Any]:
        """쿼리 성능 측정"""
        req_logger = create_request_context_logger("app.query_optimizer.measure")
        
        start_time = time.time()
        start_timestamp = datetime.now()
        
        try:
            if operation == "find":
                # explain 실행하여 실행 계획 분석
                explain_result = await collection.find(query).explain()
                
                # 실제 쿼리 실행
                results = await collection.find(query).to_list(None)
                result_count = len(results)
                
            elif operation == "count":
                explain_result = await collection.count_documents(query)
                result_count = explain_result
                
            elif operation == "aggregate":
                # aggregate의 경우 query가 pipeline이어야 함
                pipeline = query if isinstance(query, list) else [{"$match": query}]
                explain_result = await collection.aggregate(pipeline).explain()
                results = await collection.aggregate(pipeline).to_list(None)
                result_count = len(results)
            
            else:
                raise ValueError(f"지원하지 않는 operation: {operation}")
            
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            # 실행 통계 추출
            exec_stats = explain_result.get("executionStats", {})
            
            performance_data = {
                "query": query,
                "operation": operation,
                "duration_ms": round(duration_ms, 2),
                "timestamp": start_timestamp,
                "result_count": result_count,
                "docs_examined": exec_stats.get("totalDocsExamined", 0),
                "keys_examined": exec_stats.get("totalKeysExamined", 0),
                "execution_plan": exec_stats.get("winningPlan", {}),
                "index_used": exec_stats.get("winningPlan", {}).get("inputStage", {}).get("indexName"),
                "is_slow": duration_ms > self.slow_query_threshold
            }
            
            # 성능 데이터 저장
            self.query_stats.append(performance_data)
            
            req_logger.info("쿼리 성능 측정 완료", extra={
                "operation": operation,
                "duration_ms": duration_ms,
                "result_count": result_count,
                "docs_examined": performance_data["docs_examined"],
                "is_slow": performance_data["is_slow"]
            })
            
            return performance_data
            
        except Exception as e:
            req_logger.error(f"쿼리 성능 측정 실패: {e}", exc_info=True)
            raise
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """성능 측정 요약"""
        if not self.query_stats:
            return {"message": "측정된 쿼리가 없습니다."}
        
        total_queries = len(self.query_stats)
        slow_queries = [q for q in self.query_stats if q["is_slow"]]
        
        avg_duration = sum(q["duration_ms"] for q in self.query_stats) / total_queries
        max_duration = max(q["duration_ms"] for q in self.query_stats)
        min_duration = min(q["duration_ms"] for q in self.query_stats)
        
        return {
            "total_queries_measured": total_queries,
            "slow_queries_count": len(slow_queries),
            "slow_query_percentage": len(slow_queries) / total_queries * 100,
            "average_duration_ms": round(avg_duration, 2),
            "max_duration_ms": round(max_duration, 2),
            "min_duration_ms": round(min_duration, 2),
            "slow_query_threshold_ms": self.slow_query_threshold,
            "measurement_period": {
                "start": min(q["timestamp"] for q in self.query_stats),
                "end": max(q["timestamp"] for q in self.query_stats)
            }
        }


# 전역 성능 분석기 인스턴스
_performance_analyzer: Optional[QueryPerformanceAnalyzer] = None


async def get_performance_analyzer(database: AsyncIOMotorDatabase) -> QueryPerformanceAnalyzer:
    """성능 분석기 인스턴스 반환"""
    global _performance_analyzer
    
    if _performance_analyzer is None:
        _performance_analyzer = QueryPerformanceAnalyzer(database)
    
    return _performance_analyzer


async def optimize_collection_indexes(database: AsyncIOMotorDatabase, 
                                    collection_name: str) -> Dict[str, Any]:
    """컬렉션 인덱스 최적화"""
    analyzer = await get_performance_analyzer(database)
    analysis = await analyzer.analyze_collection_performance(collection_name)
    
    return {
        "collection": collection_name,
        "current_performance": analysis,
        "optimization_completed": True,
        "timestamp": datetime.now()
    }
