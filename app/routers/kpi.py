"""
KPI 쿼리 API 라우터

이 모듈은 기존 KPI 쿼리 기능을 새로운 모듈 구조로 이식합니다.
Dashboard, Statistics, AdvancedChart 컴포넌트에서 사용하는 /api/kpi/query 엔드포인트를 제공합니다.
"""

import logging
import json
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Body, HTTPException, status
from fastapi.responses import JSONResponse
import math

logger = logging.getLogger(__name__)

# 라우터 생성
router = APIRouter(
    prefix="/api/kpi",
    tags=["KPI Query"],
    responses={
        400: {"description": "Bad Request"},
        404: {"description": "Not Found"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)

def _to_list(raw):
    """문자열이나 배열을 정규화된 리스트로 변환"""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [t.strip() for t in raw.split(',') if t.strip()]
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if str(t).strip()]
    return [str(raw).strip()]

def generate_mock_kpi_data(
    start_dt: datetime,
    end_dt: datetime,
    kpi_type: str,
    entity_ids: List[str],
    ne_filters: List[str] = None,
    cellid_filters: List[str] = None,
    kpi_peg_names: List[str] = None,
    kpi_peg_like: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Mock KPI 데이터 생성기
    기존 로직을 그대로 이식
    """
    try:
        # 기본 설정
        kpi_default_value_map = {
            'availability': 99.5,
            'rrc': 98.2,
            'erab': 97.8,
            'accessibility': 99.1,
            'integrity': 98.9,
            'mobility': 96.5
        }
        
        base_value = kpi_default_value_map.get(kpi_type, 95.0)
        
        # 시간 범위 계산
        total_hours = int((end_dt - start_dt).total_seconds() / 3600) + 1
        if total_hours > 168:  # 1주일 이상이면 일별로 샘플링
            step_hours = max(1, total_hours // 168)
        else:
            step_hours = 1
            
        # Entity ID 처리
        if not entity_ids:
            entity_ids = ['LHK078ML1', 'LHK078MR1']
            
        # NE/Cell ID 필터링 시뮬레이션
        if ne_filters:
            entity_ids = [eid for eid in entity_ids if any(ne in eid for ne in ne_filters)]
        if cellid_filters:
            entity_ids = [eid for eid in entity_ids if any(cell in eid for cell in cellid_filters)]
            
        # 데이터 생성
        data = []
        current_time = start_dt
        
        while current_time <= end_dt:
            for entity_id in entity_ids:
                # 각 entity별로 약간의 변동 추가
                entity_variation = hash(entity_id) % 10 - 5  # -5 to +4
                time_variation = random.uniform(-2, 2)
                
                value = base_value + entity_variation + time_variation
                value = max(0, min(100, value))  # 0-100 범위로 제한
                
                # PEG 이름 생성 (실제 DB에서는 다를 수 있음)
                peg_names = []
                if kpi_peg_names:
                    peg_names.extend(kpi_peg_names)
                if kpi_peg_like:
                    # LIKE 패턴 시뮬레이션
                    for pattern in kpi_peg_like:
                        pattern_clean = pattern.replace('%', '').replace('_', '')
                        peg_names.append(f"{pattern_clean}_{kpi_type}")
                        
                if not peg_names:
                    peg_names = [f"{kpi_type}_rate", f"{kpi_type}_count"]
                
                for peg_name in peg_names[:2]:  # 최대 2개
                    data.append({
                        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "entity_id": entity_id,
                        "kpi_type": kpi_type,
                        "peg_name": peg_name,
                        "value": round(value, 2),
                        "ne": entity_id.split('#')[0] if '#' in entity_id else f"ne_{entity_id[:6]}",
                        "cell_id": entity_id.split('#')[1] if '#' in entity_id else f"cell_{entity_id[-3:]}",
                        "date": current_time.strftime("%Y-%m-%d"),
                        "hour": current_time.hour
                    })
                    
            current_time += timedelta(hours=step_hours)
            
        logger.info(f"Mock 데이터 생성 완료: {len(data)}건 ({kpi_type}, {len(entity_ids)} entities)")
        return data
        
    except Exception as e:
        logger.error(f"Mock 데이터 생성 실패: {e}")
        return []

@router.post("/query", summary="KPI 데이터 쿼리 (개선된)", tags=["KPI Query"])
async def kpi_query(payload: dict = Body(...)):
    """
    KPI 데이터 쿼리 엔드포인트 (여러 KPI 동시 지원)
    
    여러 `kpi_type`을 배열로 받아 한 번의 요청으로 모든 PEG 데이터를 반환합니다.
    
    Request Body:
    ```json
    {
        "start_date": "2025-08-07",
        "end_date": "2025-08-14", 
        "kpi_types": ["availability", "rrc", "erab"],
        "entity_ids": "LHK078ML1,LHK078MR1",
        "ne": "nvgnb#10000",
        "cellid": "2010"
    }
    ```
    
    Response:
    ```json
    {
        "success": true,
        "data": {
            "availability": [...],
            "rrc": [...],
            "erab": [...]
        },
        "metadata": {
            "kpi_types": ["availability", "rrc", "erab"],
            "date_range": "2025-08-07 ~ 2025-08-14",
            ...
        }
    }
    ```
    """
    try:
        # 필수 매개변수 검증
        start_date = payload.get("start_date")
        end_date = payload.get("end_date")
        kpi_types = _to_list(payload.get("kpi_types") or payload.get("kpi_type"))
        
        if not start_date or not end_date or not kpi_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="start_date, end_date, kpi_types는 필수 매개변수입니다"
            )

        # 날짜 파싱 및 검증 - ISO 형식과 일반 날짜 형식 모두 지원
        try:
            # ISO 형식 (YYYY-MM-DDTHH:MM:SS) 시도
            if 'T' in start_date or 'T' in end_date:
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            else:
                # 일반 날짜 형식 (YYYY-MM-DD)
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1) - timedelta(seconds=1)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"날짜 형식 오류: {str(e)}. 지원 형식: YYYY-MM-DD 또는 YYYY-MM-DDTHH:MM:SS"
            )

        # 날짜를 PostgreSQL 쿼리용 형식으로 변환
        start_date_formatted = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        end_date_formatted = end_dt.strftime("%Y-%m-%d %H:%M:%S")

        # 선택적 매개변수 처리
        entity_ids = payload.get("entity_ids", "LHK078ML1,LHK078MR1")
        ids = [t.strip() for t in str(entity_ids).split(",") if t.strip()]
        
        ne_filters = _to_list(payload.get("ne"))
        cellid_filters = _to_list(payload.get("cellid") or payload.get("cell"))
        kpi_peg_names = _to_list(payload.get("kpi_peg_names") or payload.get("peg_names"))
        kpi_peg_like = _to_list(payload.get("kpi_peg_like") or payload.get("peg_patterns"))

        # 로깅
        logger.info(
            f"/api/kpi/query 매개변수: kpi_types={kpi_types}, "
            f"ids={len(ids)}, ne={len(ne_filters)}, cellid={len(cellid_filters)}, "
            f"기간={start_date}~{end_date}"
        )

        # NOTE: 실제 DB 조회와 Mock 데이터 생성 분기
        # 환경변수 등을 이용해 제어할 수 있습니다. (예: USE_MOCK_DATA=true)
        # 현재는 요구사항에 따라 실제 DB 조회를 우선합니다.
        from ..utils.postgresql_db import query_kpi_data

        data_by_kpi = query_kpi_data(
            start_date=start_date_formatted,
            end_date=end_date_formatted,
            kpi_types=kpi_types,
            ne_filters=ne_filters,
            cellid_filters=cellid_filters
        )

        total_records = sum(len(v) for v in data_by_kpi.values())

        # 응답 생성
        result = {
            "success": True,
            "data": data_by_kpi,
            "metadata": {
                "total_records": total_records,
                "kpi_types": kpi_types,
                "date_range": f"{start_date} ~ {end_date}",
                "entity_count": len(ids), # Note: This might not be accurate with real data
                "ne_filters": ne_filters,
                "cellid_filters": cellid_filters,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_source": "PostgreSQL"
            }
        }

        logger.info(f"/api/kpi/query PostgreSQL 응답: rows={total_records} for {len(kpi_types)} KPIs")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"KPI 쿼리 처리 중 오류 발생: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"KPI 쿼리 처리 중 오류가 발생했습니다: {str(e)}"
        )

@router.post("/timeseries", summary="KPI 시계열 쿼리", tags=["KPI Query"])
async def kpi_timeseries(payload: dict = Body(...)):
    """
    PEG/NE/CellID/시간을 포함한 시계열 데이터 쿼리 엔드포인트.
    - 기본: 지금으로부터 1시간 전 ~ 현재
    - 최대: 14일 이전까지 허용
    - NE만 있고 CellID가 없으면 동일 NE 내 Cell 평균으로 집계하여 반환
    
    Request Body 예시:
    {
      "kpi_types": ["availability", "rrc"],
      "ne": ["NVGNB#101086"],
      "cellid": [],
      "start": "2025-08-14T00:00:00",
      "end": "2025-08-14T01:00:00"
    }
    """
    try:
        kpi_types = _to_list(payload.get("kpi_types") or payload.get("kpi_type"))
        ne_filters = _to_list(payload.get("ne"))
        cellid_filters = _to_list(payload.get("cellid") or payload.get("cell"))

        # 시간 파라미터 처리 (기본 1시간)
        end_raw = payload.get("end")
        start_raw = payload.get("start")
        now = datetime.utcnow()
        end_dt = datetime.fromisoformat(end_raw) if end_raw else now
        start_dt = datetime.fromisoformat(start_raw) if start_raw else (end_dt - timedelta(hours=1))

        # 최대 14일 제한
        if end_dt - start_dt > timedelta(days=14):
            start_dt = end_dt - timedelta(days=14)

        # 포맷팅 (postgres 함수는 날짜 문자열 사용)
        start_date = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        end_date = end_dt.strftime("%Y-%m-%d %H:%M:%S")

        from ..utils.postgresql_db import query_kpi_time_series

        data_by_kpi = query_kpi_time_series(
            start_date=start_date,
            end_date=end_date,
            kpi_types=kpi_types,
            ne_filters=ne_filters,
            cellid_filters=cellid_filters,
            aggregate_cells_if_missing=True,
        )

        return {
            "success": True,
            "data": data_by_kpi,
            "metadata": {
                "kpi_types": kpi_types,
                "ne_filters": ne_filters,
                "cellid_filters": cellid_filters,
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat(),
                "aggregated": len(cellid_filters) == 0,
            }
        }
    except Exception as e:
        logger.error("/api/kpi/timeseries 오류: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"KPI 시계열 쿼리 오류: {e}")


@router.get("/info", summary="KPI API 정보", tags=["KPI Query"])
async def kpi_info():
    """
    KPI API 정보 조회
    """
    return {
        "api": "KPI Query API",
        "version": "1.0.0",
        "description": "3GPP KPI 데이터 쿼리 API",
        "endpoints": {
            "query": "/api/kpi/query",
            "info": "/api/kpi/info",
            "timeseries": "/api/kpi/timeseries"
        },
        "supported_kpi_types": [
            "availability",
            "rrc", 
            "erab",
            "accessibility",
            "integrity",
            "mobility"
        ],
        "features": [
            "Mock 데이터 생성",
            "날짜 범위 필터링",
            "Entity ID 필터링", 
            "NE/Cell ID 필터링",
            "PEG 이름/패턴 필터링",
            "확장 가능한 DB 연동 구조"
        ]
    }

