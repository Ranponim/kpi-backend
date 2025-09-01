#!/usr/bin/env python3
"""
GET API 기능 테스트

/api/analysis/results GET 엔드포인트들의 기능을 테스트합니다.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

import httpx

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8000"

async def test_list_analysis_results():
    """분석 결과 목록 조회 API 테스트"""
    logger.info("=== 분석 결과 목록 조회 API 테스트 ===")
    
    async with httpx.AsyncClient() as client:
        try:
            # 1. 기본 목록 조회 (페이지네이션)
            logger.info("1. 기본 목록 조회 테스트")
            response = await client.get(f"{API_BASE_URL}/api/analysis/results?page=1&size=10")
            
            logger.info(f"응답 상태: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"총 결과 수: {data.get('total', 0)}")
                logger.info(f"현재 페이지: {data.get('page', 0)}")
                logger.info(f"페이지 크기: {data.get('size', 0)}")
                logger.info(f"다음 페이지 있음: {data.get('has_next', False)}")
                
                items = data.get('items', [])
                logger.info(f"이 페이지 항목 수: {len(items)}")
                
                if items:
                    logger.info("첫 번째 항목 예시:")
                    first_item = items[0]
                    logger.info(f"  ID: {first_item.get('id')}")
                    logger.info(f"  분석 날짜: {first_item.get('analysisDate')}")
                    logger.info(f"  NE ID: {first_item.get('neId')}")
                    logger.info(f"  Cell ID: {first_item.get('cellId')}")
                    logger.info(f"  상태: {first_item.get('status')}")
                    logger.info(f"  결과 수: {first_item.get('results_count')}")
                    
                    # 2. 특정 결과 상세 조회 테스트
                    await test_get_analysis_result_detail(first_item.get('id'))
                    
            else:
                logger.error(f"목록 조회 실패: {response.text}")
                
        except Exception as e:
            logger.error(f"목록 조회 테스트 중 오류: {e}")

async def test_get_analysis_result_detail(result_id: str):
    """분석 결과 상세 조회 API 테스트"""
    if not result_id:
        logger.warning("상세 조회를 위한 ID가 없습니다.")
        return
        
    logger.info(f"\n2. 분석 결과 상세 조회 테스트 (ID: {result_id})")
    
    async with httpx.AsyncClient() as client:
        try:
            # includeRaw=false로 조회 (기본)
            response = await client.get(f"{API_BASE_URL}/api/analysis/results/{result_id}")
            
            logger.info(f"응답 상태: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                result_data = data.get('data', {})
                
                logger.info("상세 결과 정보:")
                logger.info(f"  ID: {result_data.get('id')}")
                logger.info(f"  분석 날짜: {result_data.get('analysis_date')}")
                logger.info(f"  NE ID: {result_data.get('ne_id')}")
                logger.info(f"  Cell ID: {result_data.get('cell_id')}")
                logger.info(f"  상태: {result_data.get('status')}")
                logger.info(f"  분석 타입: {result_data.get('analysis_type')}")
                
                # stats 데이터 확인
                stats = result_data.get('stats', [])
                logger.info(f"  통계 데이터 수: {len(stats)}")
                if stats:
                    logger.info("  첫 번째 통계 예시:")
                    stat = stats[0]
                    logger.info(f"    기간: {stat.get('period')}")
                    logger.info(f"    KPI: {stat.get('kpi_name')}")
                    logger.info(f"    평균값: {stat.get('avg')}")
                
                # analysis 섹션 확인
                analysis = result_data.get('analysis', {})
                if analysis:
                    logger.info(f"  LLM 분석 섹션 키 수: {len(analysis.keys())}")
                    if 'summary' in analysis:
                        summary = analysis['summary']
                        logger.info(f"  분석 요약: {summary[:100] if isinstance(summary, str) else str(summary)[:100]}...")
                
                # results_overview 확인
                overview = result_data.get('results_overview', {})
                if overview:
                    logger.info(f"  결과 개요 필드 수: {len(overview.keys())}")
                    
            else:
                logger.error(f"상세 조회 실패: {response.text}")
                
        except Exception as e:
            logger.error(f"상세 조회 테스트 중 오류: {e}")

async def test_filtering_and_pagination():
    """필터링 및 페이지네이션 고급 테스트"""
    logger.info("\n3. 필터링 및 페이지네이션 고급 테스트")
    
    async with httpx.AsyncClient() as client:
        try:
            # 3.1 상태 필터 테스트
            logger.info("3.1 상태 필터 테스트 (success)")
            response = await client.get(f"{API_BASE_URL}/api/analysis/results?status=success&size=5")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"성공 상태 결과 수: {data.get('total', 0)}")
                
            # 3.2 날짜 범위 필터 테스트
            logger.info("3.2 날짜 범위 필터 테스트 (최근 7일)")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            params = {
                "date_from": start_date.isoformat(),
                "date_to": end_date.isoformat(),
                "size": 10
            }
            
            response = await client.get(f"{API_BASE_URL}/api/analysis/results", params=params)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"최근 7일 결과 수: {data.get('total', 0)}")
                
            # 3.3 페이지 크기 테스트
            logger.info("3.3 다양한 페이지 크기 테스트")
            for size in [1, 5, 20, 50]:
                response = await client.get(f"{API_BASE_URL}/api/analysis/results?page=1&size={size}")
                if response.status_code == 200:
                    data = response.json()
                    actual_items = len(data.get('items', []))
                    logger.info(f"  크기 {size} 요청 → 실제 {actual_items}개 반환")
                    
        except Exception as e:
            logger.error(f"고급 테스트 중 오류: {e}")

async def main():
    """메인 테스트 실행"""
    logger.info("GET API 기능 테스트 시작")
    logger.info(f"API 서버: {API_BASE_URL}")
    
    try:
        await test_list_analysis_results()
        await test_filtering_and_pagination()
        
        logger.info("\n✅ GET API 기능 테스트 완료")
        
    except Exception as e:
        logger.error(f"테스트 실행 중 오류: {e}")

if __name__ == "__main__":
    asyncio.run(main())
