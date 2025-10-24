#!/usr/bin/env python3
"""
비동기 분석 서비스
LLM 분석을 백그라운드에서 처리하고 상태를 관리합니다.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

import requests
from motor.motor_asyncio import AsyncIOMotorDatabase

from ..db import get_database
from ..models.analysis import AnalysisResultCreate, AnalysisResultModel


def normalize_time_format(time_input):
    """
    다양한 시간 형식을 MCP 형식으로 변환
    
    입력 형식:
    - 문자열: "2025-01-01_12:00~2025-01-01_13:00"
    - 객체: {"start": "2025-01-01 12:00:00", "end": "2025-01-01 13:00:00"}
    """
    if isinstance(time_input, str):
        return time_input  # 이미 올바른 형식
    elif isinstance(time_input, dict):
        # 객체 → 문자열 변환
        start = time_input['start'].replace(' ', '_')[:16]
        end = time_input['end'].replace(' ', '_')[:16]
        return f"{start}~{end}"
    else:
        raise ValueError(f"Unsupported time format: {type(time_input)}")

logger = logging.getLogger(__name__)


class AnalysisStatus(str, Enum):
    """분석 상태 열거형"""
    PENDING = "pending"           # 대기 중
    PROCESSING = "processing"     # 처리 중
    COMPLETED = "completed"       # 완료
    FAILED = "failed"            # 실패
    CANCELLED = "cancelled"      # 취소됨


@dataclass
class AnalysisTask:
    """분석 작업 정보"""
    task_id: str
    analysis_id: str
    status: AnalysisStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: int = 0  # 0-100
    error_message: Optional[str] = None
    request_params: Optional[Dict[str, Any]] = None
    result_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        data = asdict(self)
        # datetime 객체를 ISO 문자열로 변환
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data


class AsyncAnalysisService:
    """비동기 분석 서비스"""
    
    def __init__(self):
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.running_tasks: Dict[str, AnalysisTask] = {}
        self.max_concurrent_tasks = 5  # 최대 동시 처리 작업 수
        
    async def initialize(self):
        """서비스 초기화"""
        self.db = get_database()
        logger.info("비동기 분석 서비스 초기화 완료")
        
    async def create_analysis_task(
        self,
        request_params: Dict[str, Any],
        user_id: str = "default"
    ) -> str:
        """
        새로운 분석 작업을 생성합니다.
        
        Args:
            request_params: 분석 요청 파라미터
            user_id: 사용자 ID
            
        Returns:
            analysis_id: 생성된 분석 ID
        """
        if not self.db:
            await self.initialize()
            
        # 분석 ID 생성
        analysis_id = str(uuid.uuid4())
        task_id = f"task_{analysis_id}"
        
        # 분석 작업 생성
        task = AnalysisTask(
            task_id=task_id,
            analysis_id=analysis_id,
            status=AnalysisStatus.PENDING,
            created_at=datetime.utcnow(),
            request_params=request_params
        )
        
        # MongoDB에 초기 상태 저장
        initial_result = AnalysisResultCreate(
            analysis_id=analysis_id,
            analysis_type="async_llm_analysis",
            status=AnalysisStatus.PENDING,
            analysis_date=datetime.utcnow(),
            request_params=request_params,
            results=[],
            ne_id=request_params.get("ne_id", "unknown"),
            cell_id=request_params.get("cell_id", "unknown")
        )
        
        await self.db.analysis_results.insert_one(initial_result.dict())
        
        # 메모리에 작업 정보 저장
        self.running_tasks[analysis_id] = task
        
        logger.info(f"분석 작업 생성 완료: {analysis_id}")
        return analysis_id
        
    async def start_analysis(self, analysis_id: str) -> bool:
        """
        분석 작업을 시작합니다.
        
        Args:
            analysis_id: 분석 ID
            
        Returns:
            bool: 시작 성공 여부
        """
        if analysis_id not in self.running_tasks:
            logger.error(f"분석 작업을 찾을 수 없습니다: {analysis_id}")
            return False
            
        # 동시 실행 작업 수 확인
        running_count = sum(
            1 for task in self.running_tasks.values() 
            if task.status == AnalysisStatus.PROCESSING
        )
        
        if running_count >= self.max_concurrent_tasks:
            logger.warning(f"최대 동시 작업 수 초과: {running_count}/{self.max_concurrent_tasks}")
            return False
            
        task = self.running_tasks[analysis_id]
        task.status = AnalysisStatus.PROCESSING
        task.started_at = datetime.utcnow()
        task.progress = 10
        
        # MongoDB 상태 업데이트
        await self.db.analysis_results.update_one(
            {"analysis_id": analysis_id},
            {
                "$set": {
                    "status": AnalysisStatus.PROCESSING,
                    "started_at": task.started_at,
                    "progress": task.progress
                }
            }
        )
        
        # 백그라운드에서 분석 실행
        asyncio.create_task(self._execute_analysis_task(analysis_id))
        
        logger.info(f"분석 작업 시작: {analysis_id}")
        return True
        
    async def _execute_analysis_task(self, analysis_id: str):
        """
        백그라운드에서 분석 작업을 실행합니다.
        
        Args:
            analysis_id: 분석 ID
        """
        task = self.running_tasks.get(analysis_id)
        if not task:
            logger.error(f"분석 작업을 찾을 수 없습니다: {analysis_id}")
            return
            
        try:
            logger.info(f"분석 실행 시작: {analysis_id}")
            
            # 진행률 업데이트 (20%)
            await self._update_progress(analysis_id, 20, "데이터 수집 중...")
            
            # MCP API 호출
            analysis_result = await self._call_mcp_api(task.request_params)
            
            # 진행률 업데이트 (80%)
            await self._update_progress(analysis_id, 80, "분석 결과 처리 중...")
            
            # 결과 저장
            await self._save_analysis_result(analysis_id, analysis_result)
            
            # 완료 처리
            task.status = AnalysisStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            task.progress = 100
            task.result_data = analysis_result
            
            # MongoDB 상태 업데이트
            await self.db.analysis_results.update_one(
                {"analysis_id": analysis_id},
                {
                    "$set": {
                        "status": AnalysisStatus.COMPLETED,
                        "completed_at": task.completed_at,
                        "progress": 100,
                        "result_data": analysis_result
                    }
                }
            )
            
            logger.info(f"분석 완료: {analysis_id}")
            
        except Exception as e:
            logger.exception(f"분석 실행 실패: {analysis_id}, 오류: {e}")
            
            # 실패 처리
            task.status = AnalysisStatus.FAILED
            task.completed_at = datetime.utcnow()
            task.error_message = str(e)
            
            # MongoDB 상태 업데이트
            await self.db.analysis_results.update_one(
                {"analysis_id": analysis_id},
                {
                    "$set": {
                        "status": AnalysisStatus.FAILED,
                        "completed_at": task.completed_at,
                        "error_message": str(e)
                    }
                }
            )
            
        finally:
            # 작업 완료 후 정리 (24시간 후)
            asyncio.create_task(self._cleanup_task(analysis_id, delay_hours=24))
            
    async def _call_mcp_api(self, request_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        MCP API를 호출합니다.
        
        Args:
            request_params: 요청 파라미터
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        import os
        
        mcp_url = os.getenv("MCP_ANALYZER_URL")
        mcp_api_key = os.getenv("MCP_API_KEY")
        
        if not mcp_url:
            raise RuntimeError("MCP_ANALYZER_URL이 설정되지 않았습니다")
            
        # ✅ 시간 형식 자동 변환
        n_minus_1 = normalize_time_format(request_params.get("n_minus_1", ""))
        n = normalize_time_format(request_params.get("n", ""))
        
        # 요청 페이로드 구성
        payload = {
            "db": request_params.get("db_config", {}),
            "n_minus_1": n_minus_1,
            "n": n,
            "ne_id": request_params.get("ne_id", ""),
            "cell_id": request_params.get("cell_id", "")
        }
        
        logger.info(f"MCP API 호출 페이로드: n_minus_1={n_minus_1}, n={n}")
        
        headers = {"Content-Type": "application/json"}
        if mcp_api_key:
            headers["Authorization"] = f"Bearer {mcp_api_key}"
            
        # MCP API 호출 (타임아웃 300초로 증가)
        response = requests.post(
            mcp_url, 
            json=payload, 
            headers=headers, 
            timeout=300  # 5분으로 증가
        )
        response.raise_for_status()
        
        return response.json()
        
    async def _update_progress(self, analysis_id: str, progress: int, message: str = ""):
        """
        분석 진행률을 업데이트합니다.
        
        Args:
            analysis_id: 분석 ID
            progress: 진행률 (0-100)
            message: 진행 메시지
        """
        task = self.running_tasks.get(analysis_id)
        if task:
            task.progress = progress
            
        # MongoDB 업데이트
        update_data = {"progress": progress}
        if message:
            update_data["progress_message"] = message
            
        await self.db.analysis_results.update_one(
            {"analysis_id": analysis_id},
            {"$set": update_data}
        )
        
    async def _save_analysis_result(self, analysis_id: str, result_data: Dict[str, Any]):
        """
        분석 결과를 저장합니다.
        
        Args:
            analysis_id: 분석 ID
            result_data: 분석 결과 데이터
        """
        # DTO 구조로 정규화
        normalized_result = {
            "status": result_data.get("status", "success"),
            "time_ranges": result_data.get("time_ranges", {}),
            "peg_metrics": result_data.get("peg_metrics", {}),
            "llm_analysis": result_data.get("llm_analysis", {}),
            "metadata": result_data.get("metadata", {}),
            "legacy_payload": result_data.get("legacy_payload", {})
        }
        
        # MongoDB 업데이트
        await self.db.analysis_results.update_one(
            {"analysis_id": analysis_id},
            {
                "$set": {
                    "status": "completed",
                    "peg_metrics": normalized_result["peg_metrics"],
                    "llm_analysis": normalized_result["llm_analysis"],
                    "metadata": normalized_result["metadata"],
                    "legacy_payload": normalized_result["legacy_payload"],
                    "time_ranges": normalized_result["time_ranges"]
                }
            }
        )
        
    async def _cleanup_task(self, analysis_id: str, delay_hours: int = 24):
        """
        완료된 작업을 정리합니다.
        
        Args:
            analysis_id: 분석 ID
            delay_hours: 지연 시간 (시간)
        """
        await asyncio.sleep(delay_hours * 3600)  # 시간을 초로 변환
        
        if analysis_id in self.running_tasks:
            del self.running_tasks[analysis_id]
            logger.info(f"작업 정리 완료: {analysis_id}")
            
    async def get_task_status(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        작업 상태를 조회합니다.
        
        Args:
            analysis_id: 분석 ID
            
        Returns:
            Dict[str, Any]: 작업 상태 정보
        """
        # 메모리에서 조회
        task = self.running_tasks.get(analysis_id)
        if task:
            return task.to_dict()
            
        # MongoDB에서 조회
        result = await self.db.analysis_results.find_one(
            {"analysis_id": analysis_id}
        )
        
        if result:
            return {
                "task_id": f"task_{analysis_id}",
                "analysis_id": analysis_id,
                "status": result.get("status", "unknown"),
                "created_at": result.get("analysis_date"),
                "started_at": result.get("started_at"),
                "completed_at": result.get("completed_at"),
                "progress": result.get("progress", 0),
                "error_message": result.get("error_message"),
                "request_params": result.get("request_params"),
                "result_data": result.get("result_data")
            }
            
        return None
        
    async def cancel_task(self, analysis_id: str) -> bool:
        """
        작업을 취소합니다.
        
        Args:
            analysis_id: 분석 ID
            
        Returns:
            bool: 취소 성공 여부
        """
        task = self.running_tasks.get(analysis_id)
        if not task:
            return False
            
        if task.status in [AnalysisStatus.COMPLETED, AnalysisStatus.FAILED]:
            return False
            
        task.status = AnalysisStatus.CANCELLED
        task.completed_at = datetime.utcnow()
        
        # MongoDB 업데이트
        await self.db.analysis_results.update_one(
            {"analysis_id": analysis_id},
            {
                "$set": {
                    "status": AnalysisStatus.CANCELLED,
                    "completed_at": task.completed_at
                }
            }
        )
        
        logger.info(f"작업 취소: {analysis_id}")
        return True
        
    async def get_running_tasks(self) -> List[Dict[str, Any]]:
        """
        실행 중인 작업 목록을 조회합니다.
        
        Returns:
            List[Dict[str, Any]]: 실행 중인 작업 목록
        """
        return [
            task.to_dict() 
            for task in self.running_tasks.values() 
            if task.status in [AnalysisStatus.PENDING, AnalysisStatus.PROCESSING]
        ]


# 전역 서비스 인스턴스
async_analysis_service = AsyncAnalysisService()


