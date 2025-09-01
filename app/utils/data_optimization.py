"""
데이터 최적화 유틸리티

이 모듈은 대용량 분석 결과의 효율적인 저장을 위한 압축, GridFS, 
그리고 기타 최적화 전략을 제공합니다.
"""

import gzip
import json
import logging
import zlib
from typing import Dict, Any, Optional, Tuple, Union
from datetime import datetime
import pickle
import bson
from motor.motor_asyncio import AsyncIOMotorGridFSBucket, AsyncIOMotorCollection
from gridfs.errors import GridFSError
import hashlib

from ..db import get_database

logger = logging.getLogger("app.data_optimization")

# 압축 레벨 설정
COMPRESSION_LEVEL = 6  # 0-9, 6은 압축률과 속도의 균형점
LARGE_DOCUMENT_THRESHOLD = 1 * 1024 * 1024  # 1MB
GRIDFS_THRESHOLD = 10 * 1024 * 1024  # 10MB


class DataOptimizer:
    """데이터 최적화 관리 클래스"""
    
    def __init__(self):
        self.db = None
        self.gridfs_bucket = None
        self._initialized = False
    
    async def initialize(self):
        """GridFS 초기화"""
        if not self._initialized:
            self.db = await get_database()
            self.gridfs_bucket = AsyncIOMotorGridFSBucket(self.db, bucket_name="analysis_files")
            self._initialized = True
            logger.info("데이터 최적화 시스템 초기화 완료")
    
    async def optimize_document_for_storage(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        문서를 저장 최적화된 형태로 변환
        
        Args:
            document: 원본 문서
            
        Returns:
            최적화된 문서
        """
        await self.initialize()
        
        # 문서 크기 측정
        document_size = len(bson.BSON.encode(document))
        logger.info(f"원본 문서 크기: {document_size:,} bytes")
        
        optimized_doc = document.copy()
        
        # 대용량 필드들 처리
        large_fields = self._identify_large_fields(document)
        
        for field_name, field_data in large_fields.items():
            field_size = len(json.dumps(field_data, ensure_ascii=False).encode('utf-8'))
            
            if field_size > GRIDFS_THRESHOLD:
                # 매우 큰 데이터는 GridFS로
                file_id = await self._store_in_gridfs(field_name, field_data, document.get('_id'))
                optimized_doc[field_name] = {
                    "storage_type": "gridfs",
                    "file_id": file_id,
                    "original_size": field_size,
                    "compressed": False
                }
                logger.info(f"필드 '{field_name}' GridFS 저장 완료: {field_size:,} bytes")
                
            elif field_size > LARGE_DOCUMENT_THRESHOLD:
                # 중간 크기 데이터는 압축
                compressed_data = self._compress_field(field_data)
                optimized_doc[field_name] = {
                    "storage_type": "compressed",
                    "data": compressed_data,
                    "original_size": field_size,
                    "compressed_size": len(compressed_data),
                    "compression_ratio": len(compressed_data) / field_size
                }
                logger.info(f"필드 '{field_name}' 압축 완료: {field_size:,} → {len(compressed_data):,} bytes "
                          f"({len(compressed_data)/field_size:.1%})")
        
        # 최적화 메타데이터 추가
        optimized_doc["_optimization"] = {
            "optimized_at": datetime.now(),
            "original_size": document_size,
            "optimized_size": len(bson.BSON.encode(optimized_doc)),
            "large_fields_count": len(large_fields),
            "gridfs_fields": [f for f in large_fields.keys() 
                            if optimized_doc.get(f, {}).get("storage_type") == "gridfs"],
            "compressed_fields": [f for f in large_fields.keys() 
                                if optimized_doc.get(f, {}).get("storage_type") == "compressed"]
        }
        
        final_size = len(bson.BSON.encode(optimized_doc))
        logger.info(f"최종 문서 크기: {final_size:,} bytes (압축률: {final_size/document_size:.1%})")
        
        return optimized_doc
    
    async def restore_optimized_document(self, optimized_doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        최적화된 문서를 원본 형태로 복원
        
        Args:
            optimized_doc: 최적화된 문서
            
        Returns:
            복원된 원본 문서
        """
        await self.initialize()
        
        restored_doc = optimized_doc.copy()
        optimization_info = restored_doc.pop("_optimization", {})
        
        if not optimization_info:
            # 최적화되지 않은 문서는 그대로 반환
            return restored_doc
        
        # GridFS 필드들 복원
        for field_name in optimization_info.get("gridfs_fields", []):
            field_info = restored_doc.get(field_name, {})
            if field_info.get("storage_type") == "gridfs":
                try:
                    data = await self._retrieve_from_gridfs(field_info["file_id"])
                    restored_doc[field_name] = data
                    logger.debug(f"GridFS 필드 '{field_name}' 복원 완료")
                except Exception as e:
                    logger.error(f"GridFS 필드 '{field_name}' 복원 실패: {e}")
                    restored_doc[field_name] = None
        
        # 압축된 필드들 복원
        for field_name in optimization_info.get("compressed_fields", []):
            field_info = restored_doc.get(field_name, {})
            if field_info.get("storage_type") == "compressed":
                try:
                    data = self._decompress_field(field_info["data"])
                    restored_doc[field_name] = data
                    logger.debug(f"압축 필드 '{field_name}' 복원 완료")
                except Exception as e:
                    logger.error(f"압축 필드 '{field_name}' 복원 실패: {e}")
                    restored_doc[field_name] = None
        
        logger.info("문서 복원 완료")
        return restored_doc
    
    def _identify_large_fields(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """큰 필드들을 식별"""
        large_fields = {}
        
        # 분석 결과에서 큰 필드들
        potential_large_fields = [
            "analysis", "analysisRawCompact", "analysis_raw_compact",
            "results", "stats", "request_params", "results_overview"
        ]
        
        for field_name in potential_large_fields:
            if field_name in document:
                field_data = document[field_name]
                if field_data:  # None이 아닌 경우만
                    try:
                        field_size = len(json.dumps(field_data, ensure_ascii=False).encode('utf-8'))
                        if field_size > LARGE_DOCUMENT_THRESHOLD:
                            large_fields[field_name] = field_data
                    except (TypeError, ValueError):
                        # JSON 직렬화할 수 없는 데이터는 pickle 크기로 측정
                        try:
                            field_size = len(pickle.dumps(field_data))
                            if field_size > LARGE_DOCUMENT_THRESHOLD:
                                large_fields[field_name] = field_data
                        except Exception as e:
                            logger.warning(f"필드 '{field_name}' 크기 측정 실패: {e}")
        
        return large_fields
    
    def _compress_field(self, data: Any) -> str:
        """필드 데이터 압축"""
        try:
            # JSON으로 직렬화 후 압축
            json_data = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
            json_bytes = json_data.encode('utf-8')
            compressed = gzip.compress(json_bytes, compresslevel=COMPRESSION_LEVEL)
            
            # Base64로 인코딩하여 문자열로 저장
            import base64
            return base64.b64encode(compressed).decode('ascii')
            
        except Exception as e:
            logger.error(f"압축 실패: {e}")
            # 압축 실패 시 원본 데이터를 JSON 문자열로 저장
            try:
                return json.dumps(data, ensure_ascii=False)
            except:
                return str(data)
    
    def _decompress_field(self, compressed_data: str) -> Any:
        """압축된 필드 데이터 복원"""
        try:
            import base64
            
            # Base64 디코딩
            compressed_bytes = base64.b64decode(compressed_data.encode('ascii'))
            
            # 압축 해제
            json_bytes = gzip.decompress(compressed_bytes)
            json_data = json_bytes.decode('utf-8')
            
            # JSON 파싱
            return json.loads(json_data)
            
        except Exception as e:
            logger.error(f"압축 해제 실패: {e}")
            # 압축 해제 실패 시 원본을 JSON으로 파싱 시도
            try:
                return json.loads(compressed_data)
            except:
                return compressed_data
    
    async def _store_in_gridfs(self, field_name: str, data: Any, document_id: Optional[str] = None) -> str:
        """GridFS에 데이터 저장"""
        try:
            # 메타데이터 준비
            metadata = {
                "field_name": field_name,
                "document_id": str(document_id) if document_id else None,
                "created_at": datetime.now(),
                "data_type": type(data).__name__,
                "content_hash": self._calculate_hash(data)
            }
            
            # JSON으로 직렬화
            json_data = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
            json_bytes = json_data.encode('utf-8')
            
            # GridFS에 저장
            file_id = await self.gridfs_bucket.upload_from_stream(
                filename=f"{field_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                source=json_bytes,
                metadata=metadata
            )
            
            logger.info(f"GridFS 저장 완료: {field_name} → {file_id}")
            return str(file_id)
            
        except Exception as e:
            logger.error(f"GridFS 저장 실패: {e}")
            raise
    
    async def _retrieve_from_gridfs(self, file_id: str) -> Any:
        """GridFS에서 데이터 조회"""
        try:
            from bson import ObjectId
            
            # GridFS에서 파일 다운로드
            file_data = await self.gridfs_bucket.download_to_stream(ObjectId(file_id))
            
            # JSON 파싱
            json_data = file_data.decode('utf-8')
            return json.loads(json_data)
            
        except Exception as e:
            logger.error(f"GridFS 조회 실패: {e}")
            raise
    
    def _calculate_hash(self, data: Any) -> str:
        """데이터의 해시값 계산"""
        try:
            json_str = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
            return hashlib.md5(json_str.encode('utf-8')).hexdigest()
        except:
            return hashlib.md5(str(data).encode('utf-8')).hexdigest()
    
    async def get_storage_statistics(self) -> Dict[str, Any]:
        """저장소 통계 정보 반환"""
        await self.initialize()
        
        try:
            # GridFS 통계
            gridfs_stats = await self._get_gridfs_stats()
            
            # 문서 최적화 통계
            collection = self.db.analysis_results
            
            # 최적화된 문서 수
            optimized_count = await collection.count_documents({"_optimization": {"$exists": True}})
            total_count = await collection.count_documents({})
            
            # 압축률 통계
            optimization_pipeline = [
                {"$match": {"_optimization": {"$exists": True}}},
                {"$group": {
                    "_id": None,
                    "avg_compression_ratio": {"$avg": {
                        "$divide": ["$_optimization.optimized_size", "$_optimization.original_size"]
                    }},
                    "total_original_size": {"$sum": "$_optimization.original_size"},
                    "total_optimized_size": {"$sum": "$_optimization.optimized_size"},
                    "gridfs_documents": {"$sum": {"$size": "$_optimization.gridfs_fields"}},
                    "compressed_documents": {"$sum": {"$size": "$_optimization.compressed_fields"}}
                }}
            ]
            
            optimization_stats = await collection.aggregate(optimization_pipeline).to_list(1)
            if optimization_stats:
                opt_stats = optimization_stats[0]
            else:
                opt_stats = {}
            
            return {
                "total_documents": total_count,
                "optimized_documents": optimized_count,
                "optimization_rate": optimized_count / total_count if total_count > 0 else 0,
                "compression": {
                    "average_ratio": opt_stats.get("avg_compression_ratio", 1.0),
                    "total_space_saved": opt_stats.get("total_original_size", 0) - opt_stats.get("total_optimized_size", 0),
                    "documents_with_gridfs": opt_stats.get("gridfs_documents", 0),
                    "documents_with_compression": opt_stats.get("compressed_documents", 0)
                },
                "gridfs": gridfs_stats,
                "updated_at": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"저장소 통계 조회 실패: {e}")
            return {"error": str(e)}
    
    async def _get_gridfs_stats(self) -> Dict[str, Any]:
        """GridFS 통계 정보"""
        try:
            # GridFS 컬렉션들 통계
            fs_files = self.db.get_collection("fs.files")
            fs_chunks = self.db.get_collection("fs.chunks")
            
            files_count = await fs_files.count_documents({})
            
            # 파일 크기 통계
            size_pipeline = [
                {"$group": {
                    "_id": None,
                    "total_size": {"$sum": "$length"},
                    "avg_size": {"$avg": "$length"},
                    "max_size": {"$max": "$length"},
                    "min_size": {"$min": "$length"}
                }}
            ]
            
            size_stats = await fs_files.aggregate(size_pipeline).to_list(1)
            if size_stats:
                size_info = size_stats[0]
            else:
                size_info = {"total_size": 0, "avg_size": 0, "max_size": 0, "min_size": 0}
            
            return {
                "files_count": files_count,
                "total_size": size_info["total_size"],
                "average_file_size": size_info["avg_size"],
                "largest_file_size": size_info["max_size"],
                "smallest_file_size": size_info["min_size"]
            }
            
        except Exception as e:
            logger.error(f"GridFS 통계 조회 실패: {e}")
            return {"error": str(e)}
    
    async def cleanup_orphaned_gridfs_files(self) -> Dict[str, int]:
        """고아 GridFS 파일 정리"""
        await self.initialize()
        
        try:
            # 모든 GridFS 파일 ID 조회
            fs_files = self.db.get_collection("fs.files")
            all_files = await fs_files.find({}, {"_id": 1}).to_list(None)
            all_file_ids = {str(f["_id"]) for f in all_files}
            
            # 문서에서 참조하는 GridFS 파일 ID 조회
            collection = self.db.analysis_results
            referenced_files = set()
            
            async for doc in collection.find({"_optimization.gridfs_fields": {"$exists": True}}):
                optimization = doc.get("_optimization", {})
                for field_name in optimization.get("gridfs_fields", []):
                    field_info = doc.get(field_name, {})
                    if field_info.get("storage_type") == "gridfs":
                        referenced_files.add(field_info.get("file_id"))
            
            # 고아 파일 식별
            orphaned_files = all_file_ids - referenced_files
            
            # 고아 파일 삭제
            deleted_count = 0
            for file_id in orphaned_files:
                try:
                    from bson import ObjectId
                    await self.gridfs_bucket.delete(ObjectId(file_id))
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"고아 파일 삭제 실패 {file_id}: {e}")
            
            logger.info(f"고아 GridFS 파일 정리 완료: {deleted_count}개 삭제")
            
            return {
                "total_files": len(all_file_ids),
                "referenced_files": len(referenced_files),
                "orphaned_files": len(orphaned_files),
                "deleted_files": deleted_count
            }
            
        except Exception as e:
            logger.error(f"고아 파일 정리 실패: {e}")
            return {"error": str(e)}


# 전역 인스턴스
_data_optimizer = DataOptimizer()


async def get_data_optimizer() -> DataOptimizer:
    """데이터 최적화 인스턴스 반환"""
    await _data_optimizer.initialize()
    return _data_optimizer


# 편의 함수들
async def optimize_analysis_result(document: Dict[str, Any]) -> Dict[str, Any]:
    """분석 결과 문서 최적화"""
    optimizer = await get_data_optimizer()
    return await optimizer.optimize_document_for_storage(document)


async def restore_analysis_result(optimized_doc: Dict[str, Any]) -> Dict[str, Any]:
    """최적화된 분석 결과 복원"""
    optimizer = await get_data_optimizer()
    return await optimizer.restore_optimized_document(optimized_doc)


async def get_optimization_stats() -> Dict[str, Any]:
    """최적화 통계 조회"""
    optimizer = await get_data_optimizer()
    return await optimizer.get_storage_statistics()
