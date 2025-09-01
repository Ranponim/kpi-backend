#!/usr/bin/env python3
"""
API 호환성 테스트 스크립트

analysis_llm.py의 result_payload 구조와 /api/analysis/results API의 호환성을 테스트합니다.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any

import httpx
from motor.motor_asyncio import AsyncIOMotorClient

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 테스트 설정
API_BASE_URL = "http://localhost:8000"
MONGO_URL = "mongodb://mongo:27017"
MONGO_DB_NAME = "kpi_test"

# analysis_llm.py 스타일의 샘플 payload
SAMPLE_RESULT_PAYLOAD = {
    "analysis_type": "llm_analysis",
    "analysisDate": datetime.now().isoformat(),
    "neId": "eNB001",
    "cellId": "CELL001",
    "status": "success",
    "report_path": "/reports/analysis_2025-01-15_12-00-00.html",
    "results": [],
    "stats": [
        {
            "period": "N-1",
            "kpi_name": "RACH Success Rate",
            "avg": 97.8
        },
        {
            "period": "N",
            "kpi_name": "RACH Success Rate", 
            "avg": 98.2
        },
        {
            "period": "N-1",
            "kpi_name": "RRC Setup Success Rate",
            "avg": 96.5
        },
        {
            "period": "N",
            "kpi_name": "RRC Setup Success Rate",
            "avg": 97.1
        }
    ],
    "analysis": {
        "executive_summary": "전반적으로 성능이 개선되었습니다.",
        "diagnostic_findings": [
            {
                "primary_hypothesis": "RACH 성공률이 개선됨",
                "supporting_evidence": "N-1 대비 0.4% 향상"
            }
        ],
        "recommended_actions": [
            {
                "action": "현재 설정 유지",
                "priority": "low",
                "details": "성능이 양호하므로 현재 설정을 유지하세요."
            }
        ]
    },
    "resultsOverview": {
        "total_pegs": 2,
        "improved_pegs": 2,
        "declined_pegs": 0,
        "stable_pegs": 0
    },
    "analysisRawCompact": {
        "summary": "압축된 원본 분석 데이터",
        "data_points": 144
    },
    "request_params": {
        "db": {
            "host": "localhost",
            "port": 5432,
            "user": "postgres",
            "dbname": "kpi_db"
        },
        "time_ranges": {
            "n_minus_1": {
                "start": "2025-01-14T00:00:00",
                "end": "2025-01-14T23:59:59"
            },
            "n": {
                "start": "2025-01-15T00:00:00", 
                "end": "2025-01-15T23:59:59"
            }
        },
        "selected_pegs": ["RACH Success Rate", "RRC Setup Success Rate"]
    }
}

async def test_api_compatibility():
    """API 호환성 테스트 실행"""
    logger.info("🚀 API 호환성 테스트 시작")
    
    try:
        # 1. API 엔드포인트 테스트
        async with httpx.AsyncClient() as client:
            logger.info("📡 POST /api/analysis/results 테스트")
            
            response = await client.post(
                f"{API_BASE_URL}/api/analysis/results",
                json=SAMPLE_RESULT_PAYLOAD,
                timeout=30.0
            )
            
            logger.info(f"응답 상태 코드: {response.status_code}")
            
            if response.status_code == 201:
                result = response.json()
                logger.info("✅ API 생성 성공")
                logger.info(f"생성된 문서 ID: {result.get('data', {}).get('id')}")
                return result.get('data', {}).get('id')
            else:
                logger.error(f"❌ API 생성 실패: {response.text}")
                return None
                
    except Exception as e:
        logger.error(f"❌ API 테스트 실패: {e}")
        return None

async def test_data_retrieval(document_id: str):
    """저장된 데이터 조회 테스트"""
    if not document_id:
        logger.warning("문서 ID가 없어 조회 테스트를 건너뜁니다.")
        return
        
    try:
        async with httpx.AsyncClient() as client:
            logger.info(f"📖 GET /api/analysis/results/{document_id} 테스트")
            
            response = await client.get(
                f"{API_BASE_URL}/api/analysis/results/{document_id}",
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("✅ 데이터 조회 성공")
                
                # 데이터 무결성 검증
                data = result.get('data', {})
                if data.get('neId') == SAMPLE_RESULT_PAYLOAD['neId']:
                    logger.info("✅ 데이터 무결성 검증 성공")
                else:
                    logger.warning("⚠️ 데이터 무결성 이슈 발견")
                    
            else:
                logger.error(f"❌ 데이터 조회 실패: {response.text}")
                
    except Exception as e:
        logger.error(f"❌ 데이터 조회 테스트 실패: {e}")

async def test_list_endpoint():
    """목록 조회 엔드포인트 테스트"""
    try:
        async with httpx.AsyncClient() as client:
            logger.info("📋 GET /api/analysis/results 목록 테스트")
            
            response = await client.get(
                f"{API_BASE_URL}/api/analysis/results?page=1&size=5",
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("✅ 목록 조회 성공")
                logger.info(f"총 {result.get('total', 0)}개 항목, {len(result.get('items', []))}개 표시")
            else:
                logger.error(f"❌ 목록 조회 실패: {response.text}")
                
    except Exception as e:
        logger.error(f"❌ 목록 조회 테스트 실패: {e}")

def print_payload_structure():
    """Payload 구조 출력"""
    logger.info("📋 analysis_llm.py 호환 payload 구조:")
    print(json.dumps(SAMPLE_RESULT_PAYLOAD, indent=2, ensure_ascii=False, default=str))

async def main():
    """메인 테스트 실행"""
    logger.info("🔧 API 호환성 테스트 도구")
    
    # Payload 구조 출력
    print_payload_structure()
    
    # API 호환성 테스트
    document_id = await test_api_compatibility()
    
    # 데이터 조회 테스트
    await test_data_retrieval(document_id)
    
    # 목록 조회 테스트
    await test_list_endpoint()
    
    logger.info("🏁 테스트 완료")

if __name__ == "__main__":
    asyncio.run(main())
