from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import random
from datetime import datetime, timedelta
import os
from fastapi import Body
from dotenv import load_dotenv
import logging
import time
import math
from pymongo import MongoClient, DESCENDING
from bson import ObjectId

app = FastAPI(title="3GPP KPI Management API", version="1.0.0")

# 환경 변수 로드 (.env)
load_dotenv()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 origin 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 데이터 모델 정의
class KPIData(BaseModel):
    timestamp: str
    entity_id: str
    kpi_type: str
    value: float

class PreferenceModel(BaseModel):
    id: Optional[int] = None
    name: str
    description: Optional[str] = None
    config: dict

class SummaryReport(BaseModel):
    id: int
    title: str
    content: str
    generated_at: str

# 분석 결과 모델
class AnalysisResult(BaseModel):
    id: int
    status: str
    n_minus_1: Optional[str] = None
    n: Optional[str] = None
    analysis: Dict[str, Any]
    stats: List[Dict[str, Any]] = []
    chart_overall_base64: Optional[str] = None
    report_path: Optional[str] = None
    created_at: str

# 인메모리 저장소 (MVP)
analysis_results: List[AnalysisResult] = []
analysis_counter: int = 1

"""MongoDB 연결 (NoSQL 저장소)
- 환경변수:
  - MONGO_URL (예: mongodb://mongo:27017)
  - MONGO_DB_NAME (기본: kpi)
"""
MONGO_URL = os.getenv("MONGO_URL", "mongodb://mongo:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "kpi")
mongo_client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
db = mongo_client[MONGO_DB_NAME]
analysis_col = db["analysis_results"]
prefs_col = db["preferences"]
# 인덱스 권장
try:
    analysis_col.create_index([("created_at", DESCENDING)])
    prefs_col.create_index([("created_at", DESCENDING)])
except Exception as e:
    logging.warning("Mongo 인덱스 생성 경고: %s", e)

def _sanitize_for_json(value: Any) -> Any:
    """JSON 직렬화 호환을 위해 NaN/Infinity 값을 None으로 정규화한다.
    dict/list 등 중첩 구조를 재귀적으로 순회한다.
    """
    try:
        if isinstance(value, float):
            # 비유한수 값(NaN, +/-Infinity)은 JSON 표준 미준수 → None으로 교체
            return value if math.isfinite(value) else None
        if isinstance(value, dict):
            return {k: _sanitize_for_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_sanitize_for_json(v) for v in value]
        return value
    except Exception:
        # 예외 시 원본을 안전하게 반환 (로깅은 호출부에서 수행)
        return value

def _doc_to_analysis_model(doc: Dict[str, Any]) -> AnalysisResult:
    try:
        raw_analysis = doc.get("analysis") or {}
        raw_stats = doc.get("stats") or []
    except Exception:
        raw_analysis, raw_stats = {}, []

    safe_analysis = _sanitize_for_json(raw_analysis)
    safe_stats = _sanitize_for_json(raw_stats)

    created = doc.get("created_at")
    created_iso = created.isoformat() if hasattr(created, "isoformat") else str(created)
    return AnalysisResult(
        id=int(doc.get("_numeric_id", 0)) if isinstance(doc.get("_numeric_id"), int) else 0,
        status=str(doc.get("status", "success")),
        n_minus_1=doc.get("n_minus_1"),
        n=doc.get("n"),
        analysis=safe_analysis,
        stats=safe_stats,
        chart_overall_base64=doc.get("chart_overall_base64"),
        report_path=doc.get("report_path"),
        created_at=created_iso,
    )

# 가상 데이터 생성 함수
def generate_mock_kpi_data(start_date: str, end_date: str, kpi_type: str, entity_ids: List[str], interval_minutes: int = 60) -> List[KPIData]:
    data = []
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    
    step = timedelta(minutes=max(1, int(interval_minutes)))
    current = start
    while current <= end:
        for entity_id in entity_ids:
            # KPI 타입별로 다른 범위의 값 생성
            if kpi_type == "availability":
                base_value = 99.0
                value = base_value + random.uniform(-2.0, 1.0)
            elif kpi_type == "rrc":
                base_value = 98.5
                value = base_value + random.uniform(-1.5, 1.0)
            elif kpi_type == "erab":
                base_value = 99.2
                value = base_value + random.uniform(-1.0, 0.8)
            elif kpi_type == "sar":
                base_value = 2.5
                value = base_value + random.uniform(-0.5, 1.0)
            elif kpi_type == "mobility_intra":
                base_value = 95.0
                value = base_value + random.uniform(-5.0, 3.0)
            elif kpi_type == "payload":
                base_value = 500.0
                value = base_value + random.uniform(-100.0, 200.0)
            elif kpi_type == "cqi":
                base_value = 8.0
                value = base_value + random.uniform(-2.0, 2.0)
            elif kpi_type == "se":
                base_value = 2.0
                value = base_value + random.uniform(-0.5, 1.0)
            elif kpi_type == "dl_thp":
                base_value = 10000.0
                value = base_value + random.uniform(-2000.0, 5000.0)
            elif kpi_type == "ul_int":
                base_value = -110.0
                value = base_value + random.uniform(-5.0, 5.0)
            else:
                value = random.uniform(0, 100)
            
            data.append(KPIData(
                timestamp=current.isoformat(),
                entity_id=entity_id,
                kpi_type=kpi_type,
                value=round(value, 2)
            ))
        
        current += step  # 가변 간격 (기본 60분, 5/15분 등)
    
    return data

def _pref_record_to_model(rec: "PreferenceRecord") -> PreferenceModel:
    try:
        cfg = json.loads(rec.config_json or "{}")
    except Exception as e:
        logging.warning("Preference JSON 로드 실패(id=%s): %s", rec.id, e)
        cfg = {}
    return PreferenceModel(
        id=rec.id,
        name=rec.name,
        description=rec.description,
        config=cfg,
    )

# 가상 리포트 데이터
mock_reports = [
    SummaryReport(
        id=1,
        title="주간 네트워크 성능 분석 리포트",
        content="""
# 주간 네트워크 성능 분석 리포트

## 요약
이번 주 네트워크 성능은 전반적으로 안정적이었으며, 주요 KPI들이 목표치를 달성했습니다.

## 주요 발견사항
- **가용성(Availability)**: 평균 99.1%로 목표치(99.0%) 달성
- **RRC 성공률**: 평균 98.7%로 양호한 수준 유지
- **ERAB 성공률**: 평균 99.3%로 우수한 성능
- **다운링크 처리량**: 평균 12.5 Mbps로 증가 추세

## 권장사항
1. 특정 셀에서 간헐적인 성능 저하 모니터링 필요
2. 피크 시간대 용량 증설 검토
3. 인터페어런스 최적화 작업 수행
        """,
        generated_at="2024-08-13T10:00:00"
    )
]

# API 엔드포인트
@app.get("/")
async def root():
    """헬스 체크 및 간단한 루트 응답."""
    logging.info("GET / 호출")
    return {"message": "3GPP KPI Management API"}

@app.post("/api/analysis-result")
async def post_analysis_result(payload: dict):
    try:
        created_dt = datetime.utcnow()
        logging.info("POST /api/analysis-result 호출: created_at=%s", created_dt.isoformat())
        analysis_obj = _sanitize_for_json(payload.get("analysis") or {})
        stats_obj = _sanitize_for_json(payload.get("stats") or [])
        # allow_nan=False 정책 유지: 직렬화 가능 여부 사전 확인
        try:
            json.dumps(analysis_obj, ensure_ascii=False, allow_nan=False)
            json.dumps(stats_obj, ensure_ascii=False, allow_nan=False)
        except ValueError as ve:
            logging.error("분석결과 직렬화 실패(NaN 포함 가능): %s", ve)
            raise HTTPException(status_code=400, detail=f"Invalid numeric values in analysis/stats: {ve}")

        doc = {
            "status": str(payload.get("status", "success")),
            "n_minus_1": payload.get("n_minus_1"),
            "n": payload.get("n"),
            "analysis": analysis_obj,
            "stats": stats_obj,
            "chart_overall_base64": payload.get("chart_overall_base64"),
            "report_path": payload.get("report_path"),
            "created_at": created_dt,
        }
        result = analysis_col.insert_one(doc)
        logging.info("분석결과 저장 성공: _id=%s", result.inserted_id)
        return {"id": str(result.inserted_id), "created_at": created_dt.isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("분석결과 저장 실패")
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")

@app.get("/api/analysis-result/latest")
async def get_latest_analysis_result():
    logging.info("GET /api/analysis-result/latest 호출")
    doc = analysis_col.find_one(sort=[("created_at", DESCENDING)])
    if not doc:
        raise HTTPException(status_code=404, detail="No analysis results")
    return _doc_to_analysis_model(doc)

@app.get("/api/analysis-result/{result_id}")
async def get_analysis_result(result_id: str):
    logging.info("GET /api/analysis-result/%s 호출", result_id)
    try:
        oid = ObjectId(result_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid id")
    doc = analysis_col.find_one({"_id": oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Result not found")
    return _doc_to_analysis_model(doc)

@app.get("/api/analysis-result")
async def list_analysis_results():
    logging.info("GET /api/analysis-result 호출 (list)")
    docs = list(analysis_col.find().sort("created_at", DESCENDING))
    return {"results": [_doc_to_analysis_model(d) for d in docs]}

@app.post("/api/kpi/query")
async def kpi_query(payload: dict = Body(...)):
    """
    MVP: DB 설정을 입력으로 받아 KPI 통계를 반환하는 프록시.
    현재 단계에서는 기존 mock 생성기를 사용해 프론트 연동을 우선 보장한다.
    기대 입력 예시:
    {
      "db": {"host":"...","port":5432,"user":"...","password":"...","dbname":"..."},
      "table": "summary",
      "start_date": "YYYY-MM-DD",
      "end_date": "YYYY-MM-DD",
      "kpi_type": "availability",
      "entity_ids": "LHK078ML1,LHK078MR1"
    }
    """
    try:
        start_date = payload.get("start_date")
        end_date = payload.get("end_date")
        kpi_type = payload.get("kpi_type")
        entity_ids = payload.get("entity_ids", "LHK078ML1,LHK078MR1")
        if not start_date or not end_date or not kpi_type:
            raise ValueError("start_date, end_date, kpi_type는 필수입니다")

        # 날짜 문자열(YYYY-MM-DD) → 하루 범위
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1) - timedelta(seconds=1)
        except Exception:
            raise ValueError("start_date/end_date 형식은 YYYY-MM-DD 이어야 합니다")

        ids = [t.strip() for t in str(entity_ids).split(",") if t.strip()]
        # 확장 필터: ne, cellid (문자열 또는 배열)
        def _to_list(raw):
            if raw is None:
                return []
            if isinstance(raw, str):
                return [t.strip() for t in raw.split(',') if t.strip()]
            if isinstance(raw, list):
                return [str(t).strip() for t in raw if str(t).strip()]
            return [str(raw).strip()]

        ne_filters = _to_list(payload.get("ne"))
        cellid_filters = _to_list(payload.get("cellid") or payload.get("cell"))
        # KPI 매핑: exact peg_names, like patterns (ILIKE)
        kpi_peg_names = _to_list(payload.get("kpi_peg_names") or payload.get("peg_names"))
        kpi_peg_like = _to_list(payload.get("kpi_peg_like") or payload.get("peg_patterns"))
        logging.info("/api/kpi/query 매개변수: kpi_type=%s, ids=%d, ne=%d, cellid=%d, 기간=%s~%s",
                     kpi_type, len(ids), len(ne_filters), len(cellid_filters), start_date, end_date)

        # NoSQL 전환에 따라 외부 SQL 프록시는 비활성화하고 mock 데이터 제공
        data = generate_mock_kpi_data(start_date, end_date, kpi_type, ids)
        logging.info("/api/kpi/query mock 응답: rows=%d", len(data))
        return {"data": data, "source": "proxy-mock"}
    except Exception:
        # 실패 시 mock 으로 폴백하여 프론트 사용성 보장
        start_date = payload.get("start_date")
        end_date = payload.get("end_date")
        kpi_type = payload.get("kpi_type")
        entity_ids = payload.get("entity_ids", "LHK078ML1,LHK078MR1")
        logging.warning("KPI 프록시 실패, mock 데이터로 폴백")
        data = generate_mock_kpi_data(start_date, end_date, kpi_type, entity_ids.split(","))
        logging.info("/api/kpi/query 폴백 성공: rows=%d (proxy-mock)", len(data))
        return {"data": data, "source": "proxy-mock"}

@app.get("/api/kpi/statistics")
async def get_kpi_statistics(
    start_date: str = Query(..., description="시작 날짜 (YYYY-MM-DD)"),
    end_date: str = Query(..., description="종료 날짜 (YYYY-MM-DD)"),
    kpi_type: str = Query(..., description="KPI 타입"),
    entity_ids: str = Query("LHK078ML1,LHK078MR1", description="엔티티 ID 목록 (쉼표로 구분)"),
    interval_minutes: int = Query(60, ge=1, le=60*24, description="샘플링 간격(분). 최소 5/15 등의 간격 지원")
):
    logging.info("GET /api/kpi/statistics: kpi_type=%s, ids=%s, interval=%s, 기간=%s~%s", kpi_type, entity_ids, interval_minutes, start_date, end_date)
    entity_list = entity_ids.split(",")
    data = generate_mock_kpi_data(start_date, end_date, kpi_type, entity_list, interval_minutes=interval_minutes)
    logging.info("/api/kpi/statistics 응답 rows=%d", len(data))
    return {"data": data}

@app.get("/api/kpi/trends")
async def get_kpi_trends(
    start_date: str = Query(..., description="시작 날짜 (YYYY-MM-DD)"),
    end_date: str = Query(..., description="종료 날짜 (YYYY-MM-DD)"),
    kpi_type: str = Query(..., description="KPI 타입"),
    entity_id: str = Query(..., description="엔티티 ID"),
    interval_minutes: int = Query(60, ge=1, le=60*24, description="샘플링 간격(분)")
):
    logging.info("GET /api/kpi/trends: kpi_type=%s, entity_id=%s, interval=%s, 기간=%s~%s", kpi_type, entity_id, interval_minutes, start_date, end_date)
    data = generate_mock_kpi_data(start_date, end_date, kpi_type, [entity_id], interval_minutes=interval_minutes)
    logging.info("/api/kpi/trends 응답 rows=%d", len(data))
    return {"data": data}

@app.post("/api/kpi/statistics/batch")
async def get_kpi_statistics_batch(payload: dict = Body(...)):
    """
    다수 KPI 타입을 한 번에 조회하는 배치 엔드포인트.
    현재 단계에서는 mock 생성기를 사용해 프론트엔드 대량 KPI 차트 표시를 지원한다.

    기대 입력 예시:
    {
      "start_date": "YYYY-MM-DD",
      "end_date": "YYYY-MM-DD",
      "kpi_types": ["availability","rrc",...],
      "entity_ids": "LHK078ML1,LHK078MR1",
      "interval_minutes": 60
    }
    """
    try:
        start_date = payload.get("start_date")
        end_date = payload.get("end_date")
        kpi_types = payload.get("kpi_types") or []
        entity_ids = payload.get("entity_ids", "LHK078ML1,LHK078MR1")
        interval_minutes = int(payload.get("interval_minutes") or 60)
        if not start_date or not end_date or not kpi_types:
            raise HTTPException(status_code=400, detail="start_date, end_date, kpi_types는 필수입니다")

        entities = [t.strip() for t in str(entity_ids).split(",") if t.strip()]
        # 확장 필터 전달(향후 DB 프록시 통합 시 사용할 수 있도록 수집)
        ne_filters = payload.get("ne")
        cellid_filters = payload.get("cellid") or payload.get("cell")
        if ne_filters or cellid_filters:
            logging.info("배치 필터: ne=%s, cellid=%s", ne_filters, cellid_filters)
        result: Dict[str, List[Dict[str, Any]]] = {}
        logging.info("POST /api/kpi/statistics/batch: types=%d, ids=%d, interval=%s, 기간=%s~%s", len(kpi_types), len(entities), interval_minutes, start_date, end_date)
        for kt in kpi_types:
            result[str(kt)] = generate_mock_kpi_data(start_date, end_date, str(kt), entities, interval_minutes=interval_minutes)
        total = sum(len(v) for v in result.values())
        logging.info("/api/kpi/statistics/batch 응답 합계 rows=%d", total)
        return {"data": result, "source": "proxy-mock"}
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("배치 KPI 통계 생성 실패")
        raise HTTPException(status_code=400, detail=f"failed: {e}")

@app.get("/api/reports/summary")
async def get_summary_reports(report_id: Optional[int] = None):
    if report_id:
        report = next((r for r in mock_reports if r.id == report_id), None)
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        return report
    return {"reports": mock_reports}

@app.get("/api/preferences")
async def get_preferences():
    logging.info("GET /api/preferences 호출(DB)")
    docs = list(prefs_col.find().sort("created_at", DESCENDING))
    items = []
    for d in docs:
        items.append(PreferenceModel(
            id=str(d.get("_id")),
            name=d.get("name", ""),
            description=d.get("description"),
            config=(d.get("config") or {}),
        ))
    logging.info("/api/preferences 응답 count=%d", len(items))
    return {"preferences": items}

@app.get("/api/preferences/{preference_id}")
async def get_preference(preference_id: str):
    logging.info("GET /api/preferences/%s 호출(DB)", preference_id)
    try:
        oid = ObjectId(preference_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid id")
    d = prefs_col.find_one({"_id": oid})
    if not d:
        raise HTTPException(status_code=404, detail="Preference not found")
    return _pref_record_to_model(type("_R", (), {
        "id": preference_id,
        "name": d.get("name", ""),
        "description": d.get("description"),
        "config_json": json.dumps(d.get("config") or {}, ensure_ascii=False),
    }))

@app.post("/api/preferences")
async def create_preference(preference: PreferenceModel):
    logging.info("POST /api/preferences 호출(DB): name=%s", preference.name)
    created_dt = datetime.utcnow()
    try:
        json.dumps(preference.config or {}, ensure_ascii=False, allow_nan=False)
    except ValueError as ve:
        logging.error("Preference 직렬화 실패: %s", ve)
        raise HTTPException(status_code=400, detail=f"Invalid config: {ve}")
    d = {
        "name": preference.name,
        "description": preference.description,
        "config": preference.config or {},
        "created_at": created_dt,
    }
    res = prefs_col.insert_one(d)
    logging.info("Preference 저장 성공: id=%s", res.inserted_id)
    return {"id": str(res.inserted_id), "message": "Preference created successfully"}

@app.put("/api/preferences/{preference_id}")
async def update_preference(preference_id: str, preference: PreferenceModel):
    logging.info("PUT /api/preferences/%s 호출(DB)", preference_id)
    try:
        json.dumps(preference.config or {}, ensure_ascii=False, allow_nan=False)
    except ValueError as ve:
        logging.error("Preference 직렬화 실패: %s", ve)
        raise HTTPException(status_code=400, detail=f"Invalid config: {ve}")
    try:
        oid = ObjectId(preference_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid id")
    result = prefs_col.update_one({"_id": oid}, {"$set": {
        "name": preference.name,
        "description": preference.description,
        "config": preference.config or {},
    }})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Preference not found")
    return {"message": "Preference updated successfully"}

@app.delete("/api/preferences/{preference_id}")
async def delete_preference(preference_id: str):
    logging.info("DELETE /api/preferences/%s 호출(DB)", preference_id)
    try:
        oid = ObjectId(preference_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid id")
    result = prefs_col.delete_one({"_id": oid})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Preference not found")
    return {"message": "Preference deleted successfully"}

@app.get("/api/preferences/{preference_id}/derived-pegs")
async def get_preference_derived_pegs(preference_id: str):
    logging.info("GET /api/preferences/%s/derived-pegs 호출(DB)", preference_id)
    try:
        oid = ObjectId(preference_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid id")
    d = prefs_col.find_one({"_id": oid})
    if not d:
        raise HTTPException(status_code=404, detail="Preference not found")
    cfg = d.get("config") or {}
    derived = (cfg or {}).get("derived_pegs") or {}
    return {"derived_pegs": derived}

@app.put("/api/preferences/{preference_id}/derived-pegs")
async def update_preference_derived_pegs(preference_id: str, payload: dict = Body(...)):
    logging.info("PUT /api/preferences/%s/derived-pegs 호출(DB)", preference_id)
    if not isinstance(payload.get("derived_pegs"), dict):
        raise HTTPException(status_code=400, detail="derived_pegs must be an object {name: expr}")
    try:
        oid = ObjectId(preference_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid id")
    d = prefs_col.find_one({"_id": oid})
    if not d:
        raise HTTPException(status_code=404, detail="Preference not found")
    cfg = d.get("config") or {}
    cfg["derived_pegs"] = payload["derived_pegs"]
    try:
        json.dumps(cfg, ensure_ascii=False, allow_nan=False)
    except ValueError as ve:
        logging.error("Preference 직렬화 실패: %s", ve)
        raise HTTPException(status_code=400, detail=f"Invalid derived_pegs: {ve}")
    prefs_col.update_one({"_id": oid}, {"$set": {"config": cfg}})
    return {"message": "Derived PEGs updated", "derived_pegs": cfg["derived_pegs"]}

@app.get("/api/master/pegs")
async def get_pegs():
    return {
        "pegs": [
            {"id": "PEG001", "name": "Seoul Central"},
            {"id": "PEG002", "name": "Busan North"},
            {"id": "PEG003", "name": "Daegu West"}
        ]
    }

@app.get("/api/master/cells")
async def get_cells():
    return {
        "cells": [
            {"id": "LHK078ML1", "name": "Seoul-Gangnam-001"},
            {"id": "LHK078MR1", "name": "Seoul-Gangnam-002"},
            {"id": "LHK078ML1_SIMPANGRAMBONGL04", "name": "Seoul-Gangnam-003"}
        ]
    }

@app.post("/api/db/ping")
async def db_ping(payload: dict = Body(...)):
    """MongoDB 연결 확인.
    - 입력: {"mongo_url": "mongodb://..."} (옵션)
    - 성공: {"ok": true}
    """
    logging.info("POST /api/db/ping 호출 (Mongo)")
    mongo_url = (payload or {}).get("mongo_url") or MONGO_URL
    try:
        test_client = MongoClient(mongo_url, serverSelectionTimeoutMS=3000)
        test_client.admin.command("ping")
        return {"ok": True}
    except Exception as e:
        logging.warning("Mongo ping 실패: %s", e)
        raise HTTPException(status_code=400, detail=f"DB connection failed: {e}")

def _valid_ident(name: str) -> bool:
    return bool(name) and name.replace('_','a').replace('0','0').replace('1','1').isalnum() and all(c.isalnum() or c=='_' for c in name)

@app.post("/api/master/ne-list")
async def list_ne(payload: dict = Body(...)):
    """NoSQL 모드: 외부 DB 조회 비활성화. 빈 목록 반환."""
    return {"items": []}

@app.post("/api/master/cellid-list")
async def list_cellid(payload: dict = Body(...)):
    """NoSQL 모드: 외부 DB 조회 비활성화. 빈 목록 반환."""
    return {"items": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

