#!/usr/bin/env python3
"""
비동기 분석 서비스 테스트 스크립트
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.async_analysis_service import AsyncAnalysisService, AnalysisStatus

async def test_async_service():
    print('=== 비동기 분석 서비스 테스트 ===')
    
    # 서비스 초기화
    service = AsyncAnalysisService()
    await service.initialize()
    
    # 테스트 요청 파라미터
    request_params = {
        'db_config': {
            'host': 'localhost',
            'port': 5432,
            'user': 'postgres',
            'password': 'password',
            'dbname': 'test_db'
        },
        'n_minus_1': '2025-01-01_00:00~2025-01-01_23:59',
        'n': '2025-01-02_00:00~2025-01-02_23:59',
        'ne_id': 'test_ne',
        'cell_id': 'test_cell',
        'user_id': 'test_user'
    }
    
    print('1. 분석 작업 생성 테스트')
    analysis_id = await service.create_analysis_task(request_params, 'test_user')
    print(f'   ✅ 분석 ID 생성: {analysis_id}')
    
    print('2. 분석 시작 테스트')
    started = await service.start_analysis(analysis_id)
    print(f'   ✅ 분석 시작: {started}')
    
    print('3. 상태 조회 테스트')
    status = await service.get_task_status(analysis_id)
    print(f'   ✅ 상태 조회: {status["status"]} (진행률: {status["progress"]}%)')
    
    print('4. 실행 중인 작업 목록 테스트')
    running_tasks = await service.get_running_tasks()
    print(f'   ✅ 실행 중인 작업 수: {len(running_tasks)}')
    
    print('5. 작업 취소 테스트')
    cancelled = await service.cancel_task(analysis_id)
    print(f'   ✅ 작업 취소: {cancelled}')
    
    print('\n=== 테스트 완료 ===')

if __name__ == '__main__':
    asyncio.run(test_async_service())
