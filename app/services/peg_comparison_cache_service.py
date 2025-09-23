"""
PEG 비교분석 캐싱 서비스

이 모듈은 PEG 비교분석 결과의 캐싱을 담당하는 서비스입니다.
Redis와 메모리 캐시를 활용하여 성능을 최적화합니다.
"""

import logging
import json
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import asyncio

from ..utils.cache_manager import get_cache_manager
from ..exceptions.peg_comparison_exceptions import CacheError
from ..models.peg_comparison import PEGComparisonAnalysisModel

logger = logging.getLogger("app.services.peg_comparison_cache")


class PEGComparisonCacheService:
    """PEG 비교분석 캐싱 서비스 클래스"""
    
    def __init__(self):
        """캐싱 서비스 초기화"""
        self.cache_ttl = {
            'peg_comparison': 3600,  # 1시간
            'summary_stats': 1800,   # 30분
            'metadata': 7200,        # 2시간
            'async_task': 300        # 5분
        }
        
        self.cache_prefix = "peg_comparison"
        
        logger.info("PEG 비교분석 캐싱 서비스 초기화 완료")
    
    def _generate_cache_key(
        self, 
        analysis_id: str, 
        cache_type: str, 
        additional_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        캐시 키 생성
        
        Args:
            analysis_id: 분석 ID
            cache_type: 캐시 타입
            additional_params: 추가 파라미터
            
        Returns:
            str: 생성된 캐시 키
        """
        # 기본 키 구성
        key_parts = [self.cache_prefix, cache_type, analysis_id]
        
        # 추가 파라미터가 있으면 해시로 추가
        if additional_params:
            params_str = json.dumps(additional_params, sort_keys=True)
            params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
            key_parts.append(params_hash)
        
        return ":".join(key_parts)
    
    async def cache_peg_comparison_result(
        self, 
        analysis_id: str, 
        result: PEGComparisonAnalysisModel,
        cache_type: str = "peg_comparison"
    ) -> bool:
        """
        PEG 비교분석 결과 캐싱
        
        Args:
            analysis_id: 분석 ID
            result: 분석 결과
            cache_type: 캐시 타입
            
        Returns:
            bool: 캐싱 성공 여부
        """
        try:
            cache_manager = await get_cache_manager()
            cache_key = self._generate_cache_key(analysis_id, cache_type)
            ttl = self.cache_ttl.get(cache_type, 3600)
            
            # 캐시 데이터 구성
            cache_data = {
                'data': result.model_dump(),
                'cached_at': datetime.utcnow().isoformat(),
                'ttl': ttl,
                'cache_type': cache_type
            }
            
            # 캐시 저장
            await cache_manager.set(cache_key, cache_data, ttl=ttl)
            
            logger.info(f"PEG 비교분석 결과 캐싱 완료: {analysis_id}", extra={
                "analysis_id": analysis_id,
                "cache_key": cache_key,
                "cache_type": cache_type,
                "ttl": ttl
            })
            
            return True
            
        except Exception as e:
            logger.error(f"PEG 비교분석 결과 캐싱 실패: {analysis_id}", extra={
                "analysis_id": analysis_id,
                "cache_type": cache_type,
                "error_type": type(e).__name__,
                "error_message": str(e)
            }, exc_info=True)
            
            raise CacheError(
                message=f"캐싱 실패: {str(e)}",
                cache_operation="cache_peg_comparison_result",
                details={
                    "analysis_id": analysis_id,
                    "cache_type": cache_type,
                    "error": str(e)
                }
            )
    
    async def get_cached_peg_comparison_result(
        self, 
        analysis_id: str,
        cache_type: str = "peg_comparison"
    ) -> Optional[PEGComparisonAnalysisModel]:
        """
        캐시된 PEG 비교분석 결과 조회
        
        Args:
            analysis_id: 분석 ID
            cache_type: 캐시 타입
            
        Returns:
            Optional[PEGComparisonAnalysisModel]: 캐시된 결과 또는 None
        """
        try:
            cache_manager = await get_cache_manager()
            cache_key = self._generate_cache_key(analysis_id, cache_type)
            
            # 캐시 조회
            cached_data = await cache_manager.get(cache_key)
            
            if cached_data:
                # 캐시 데이터 파싱
                if isinstance(cached_data, dict):
                    result_data = cached_data.get('data')
                    if result_data:
                        result = PEGComparisonAnalysisModel(**result_data)
                        result.cached = True
                        
                        logger.info(f"캐시된 PEG 비교분석 결과 조회 성공: {analysis_id}", extra={
                            "analysis_id": analysis_id,
                            "cache_key": cache_key,
                            "cache_type": cache_type,
                            "cached_at": cached_data.get('cached_at')
                        })
                        
                        return result
            
            logger.debug(f"캐시된 PEG 비교분석 결과 없음: {analysis_id}", extra={
                "analysis_id": analysis_id,
                "cache_key": cache_key,
                "cache_type": cache_type
            })
            
            return None
            
        except Exception as e:
            logger.warning(f"캐시된 PEG 비교분석 결과 조회 실패: {analysis_id}", extra={
                "analysis_id": analysis_id,
                "cache_type": cache_type,
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            
            # 캐시 조회 실패는 에러를 발생시키지 않고 None 반환
            return None
    
    async def cache_async_task_status(
        self, 
        task_id: str, 
        status_data: Dict[str, Any]
    ) -> bool:
        """
        비동기 작업 상태 캐싱
        
        Args:
            task_id: 작업 ID
            status_data: 상태 데이터
            
        Returns:
            bool: 캐싱 성공 여부
        """
        try:
            cache_manager = await get_cache_manager()
            cache_key = self._generate_cache_key(task_id, "async_task")
            ttl = self.cache_ttl.get('async_task', 300)
            
            # 캐시 데이터 구성
            cache_data = {
                'data': status_data,
                'cached_at': datetime.utcnow().isoformat(),
                'ttl': ttl,
                'cache_type': 'async_task'
            }
            
            # 캐시 저장
            await cache_manager.set(cache_key, cache_data, ttl=ttl)
            
            logger.debug(f"비동기 작업 상태 캐싱 완료: {task_id}", extra={
                "task_id": task_id,
                "cache_key": cache_key,
                "status": status_data.get('status')
            })
            
            return True
            
        except Exception as e:
            logger.error(f"비동기 작업 상태 캐싱 실패: {task_id}", extra={
                "task_id": task_id,
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            
            raise CacheError(
                message=f"비동기 작업 상태 캐싱 실패: {str(e)}",
                cache_operation="cache_async_task_status",
                details={
                    "task_id": task_id,
                    "error": str(e)
                }
            )
    
    async def get_cached_async_task_status(
        self, 
        task_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        캐시된 비동기 작업 상태 조회
        
        Args:
            task_id: 작업 ID
            
        Returns:
            Optional[Dict[str, Any]]: 캐시된 상태 데이터 또는 None
        """
        try:
            cache_manager = await get_cache_manager()
            cache_key = self._generate_cache_key(task_id, "async_task")
            
            # 캐시 조회
            cached_data = await cache_manager.get(cache_key)
            
            if cached_data and isinstance(cached_data, dict):
                status_data = cached_data.get('data')
                if status_data:
                    logger.debug(f"캐시된 비동기 작업 상태 조회 성공: {task_id}", extra={
                        "task_id": task_id,
                        "cache_key": cache_key,
                        "status": status_data.get('status')
                    })
                    
                    return status_data
            
            logger.debug(f"캐시된 비동기 작업 상태 없음: {task_id}", extra={
                "task_id": task_id,
                "cache_key": cache_key
            })
            
            return None
            
        except Exception as e:
            logger.warning(f"캐시된 비동기 작업 상태 조회 실패: {task_id}", extra={
                "task_id": task_id,
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            
            return None
    
    async def invalidate_peg_comparison_cache(
        self, 
        analysis_id: str,
        cache_types: Optional[List[str]] = None
    ) -> int:
        """
        PEG 비교분석 캐시 무효화
        
        Args:
            analysis_id: 분석 ID
            cache_types: 무효화할 캐시 타입 목록 (None이면 모든 타입)
            
        Returns:
            int: 무효화된 캐시 항목 수
        """
        try:
            cache_manager = await get_cache_manager()
            
            if cache_types is None:
                cache_types = list(self.cache_ttl.keys())
            
            deleted_count = 0
            
            for cache_type in cache_types:
                cache_key = self._generate_cache_key(analysis_id, cache_type)
                
                # 개별 캐시 키 삭제
                if await cache_manager.delete(cache_key):
                    deleted_count += 1
                    logger.debug(f"PEG 비교분석 캐시 무효화: {cache_key}")
            
            # 패턴 기반 캐시 삭제 (추가 파라미터가 있는 경우)
            pattern = f"{self.cache_prefix}:*:{analysis_id}*"
            pattern_deleted = await cache_manager.delete_pattern(pattern)
            deleted_count += pattern_deleted
            
            logger.info(f"PEG 비교분석 캐시 무효화 완료: {analysis_id}", extra={
                "analysis_id": analysis_id,
                "deleted_count": deleted_count,
                "cache_types": cache_types
            })
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"PEG 비교분석 캐시 무효화 실패: {analysis_id}", extra={
                "analysis_id": analysis_id,
                "cache_types": cache_types,
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            
            raise CacheError(
                message=f"캐시 무효화 실패: {str(e)}",
                cache_operation="invalidate_peg_comparison_cache",
                details={
                    "analysis_id": analysis_id,
                    "cache_types": cache_types,
                    "error": str(e)
                }
            )
    
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """
        캐시 통계 조회
        
        Returns:
            Dict[str, Any]: 캐시 통계 정보
        """
        try:
            cache_manager = await get_cache_manager()
            
            # 전체 캐시 통계
            cache_stats = await cache_manager.get_cache_stats()
            
            # PEG 비교분석 관련 캐시 통계
            peg_pattern = f"{self.cache_prefix}:*"
            peg_keys = await cache_manager.get_keys(peg_pattern)
            
            peg_stats = {
                "total_peg_cache_keys": len(peg_keys),
                "cache_types": {},
                "cache_ttl": self.cache_ttl
            }
            
            # 캐시 타입별 통계
            for cache_type in self.cache_ttl.keys():
                type_pattern = f"{self.cache_prefix}:{cache_type}:*"
                type_keys = await cache_manager.get_keys(type_pattern)
                peg_stats["cache_types"][cache_type] = len(type_keys)
            
            # 통합 통계
            combined_stats = {
                "overall_cache": cache_stats,
                "peg_comparison_cache": peg_stats,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.debug("PEG 비교분석 캐시 통계 조회 완료", extra={
                "total_peg_keys": len(peg_keys),
                "cache_types_count": len(peg_stats["cache_types"])
            })
            
            return combined_stats
            
        except Exception as e:
            logger.error(f"PEG 비교분석 캐시 통계 조회 실패: {e}", extra={
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def cache_async_task_status(
        self, 
        task_id: str, 
        task_data: Dict[str, Any]
    ) -> bool:
        """
        비동기 작업 상태 캐싱
        
        Args:
            task_id: 작업 ID
            task_data: 작업 데이터
            
        Returns:
            bool: 캐싱 성공 여부
        """
        try:
            cache_key = f"{self.cache_prefix}:async_task:{task_id}"
            ttl = self.cache_ttl['async_task']
            
            # 직렬화 가능한 형태로 변환
            serializable_data = self._make_serializable(task_data)
            
            cache_manager = await get_cache_manager()
            await cache_manager.set(cache_key, serializable_data, ttl)
            
            logger.debug(f"비동기 작업 상태 캐싱 완료: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"비동기 작업 상태 캐싱 실패: {task_id}", extra={
                "task_id": task_id,
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            return False
    
    async def get_cached_async_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        캐시된 비동기 작업 상태 조회
        
        Args:
            task_id: 작업 ID
            
        Returns:
            Optional[Dict[str, Any]]: 캐시된 작업 상태 또는 None
        """
        try:
            cache_key = f"{self.cache_prefix}:async_task:{task_id}"
            
            cache_manager = await get_cache_manager()
            cached_data = await cache_manager.get(cache_key)
            
            if cached_data:
                logger.debug(f"캐시된 비동기 작업 상태 조회 성공: {task_id}")
                return cached_data
            
            logger.debug(f"캐시된 비동기 작업 상태 없음: {task_id}")
            return None
            
        except Exception as e:
            logger.error(f"캐시된 비동기 작업 상태 조회 실패: {task_id}", extra={
                "task_id": task_id,
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            return None
    
    def _make_serializable(self, data: Any) -> Any:
        """
        데이터를 직렬화 가능한 형태로 변환
        
        Args:
            data: 변환할 데이터
            
        Returns:
            Any: 직렬화 가능한 데이터
        """
        if isinstance(data, dict):
            return {key: self._make_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._make_serializable(item) for item in data]
        elif isinstance(data, datetime):
            return data.isoformat()
        elif hasattr(data, 'dict'):  # Pydantic 모델
            return data.dict()
        else:
            return data
    
    async def cleanup_expired_cache(self) -> Dict[str, int]:
        """
        만료된 캐시 정리
        
        Returns:
            Dict[str, int]: 정리 결과 통계
        """
        try:
            cache_manager = await get_cache_manager()
            
            cleanup_stats = {
                "total_cleaned": 0,
                "by_type": {}
            }
            
            # 각 캐시 타입별로 만료된 항목 정리
            for cache_type in self.cache_ttl.keys():
                pattern = f"{self.cache_prefix}:{cache_type}:*"
                cleaned_count = await cache_manager.cleanup_expired(pattern)
                
                cleanup_stats["by_type"][cache_type] = cleaned_count
                cleanup_stats["total_cleaned"] += cleaned_count
            
            logger.info("PEG 비교분석 만료 캐시 정리 완료", extra={
                "total_cleaned": cleanup_stats["total_cleaned"],
                "by_type": cleanup_stats["by_type"]
            })
            
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"PEG 비교분석 만료 캐시 정리 실패: {e}", extra={
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            
            return {
                "error": str(e),
                "total_cleaned": 0,
                "by_type": {}
            }


# 전역 캐시 서비스 인스턴스
_peg_cache_service: Optional[PEGComparisonCacheService] = None


async def get_peg_comparison_cache_service() -> PEGComparisonCacheService:
    """
    전역 PEG 비교분석 캐시 서비스 인스턴스 반환
    
    Returns:
        PEGComparisonCacheService: 캐시 서비스 인스턴스
    """
    global _peg_cache_service
    
    if _peg_cache_service is None:
        _peg_cache_service = PEGComparisonCacheService()
        logger.info("전역 PEG 비교분석 캐시 서비스 생성")
    
    return _peg_cache_service
