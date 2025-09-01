"""
마스터 데이터 API 라우터

PEG 목록, Cell 목록 등 기준 정보를 제공하는 API 엔드포인트들을 정의합니다.
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import psycopg2
from motor.motor_asyncio import AsyncIOMotorDatabase

from ..db import get_database
from ..exceptions import DatabaseConnectionException

# 로거 설정
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/master", tags=["master"])


@router.get("/pegs", response_model=List[Dict[str, Any]])
async def get_pegs(db: AsyncIOMotorDatabase = Depends(get_database)):
    """
    PEG 마스터 데이터 목록을 조회합니다.
    
    Returns:
        List[Dict]: PEG 마스터 데이터 목록
        - peg_name: PEG 이름
        - description: PEG 설명
        - unit: 단위
    """
    try:
        logger.info("PEG 마스터 데이터 조회 시작")
        
        # MongoDB에서 peg_master 컬렉션 조회
        cursor = db.peg_master.find({}, {"_id": 0})  # _id 필드 제외
        pegs = await cursor.to_list(length=None)
        
        logger.info(f"PEG 마스터 데이터 조회 완료: {len(pegs)}개")
        
        # 데이터가 없으면 기본 하드코딩된 PEG 목록 반환
        if not pegs:
            logger.warning("DB에 PEG 마스터 데이터가 없어 기본 목록 반환")
            pegs = [
                {"peg_name": "availability", "description": "가용성", "unit": "%"},
                {"peg_name": "rrc_success_rate", "description": "RRC 성공률", "unit": "%"},
                {"peg_name": "erab_success_rate", "description": "ERAB 성공률", "unit": "%"},
                {"peg_name": "handover_success_rate", "description": "핸드오버 성공률", "unit": "%"},
                {"peg_name": "throughput_dl", "description": "하향 처리량", "unit": "Mbps"},
                {"peg_name": "throughput_ul", "description": "상향 처리량", "unit": "Mbps"}
            ]
        
        return pegs
        
    except Exception as e:
        logger.error(f"PEG 마스터 데이터 조회 오류: {str(e)}")
        raise DatabaseConnectionException(f"PEG 마스터 데이터 조회 실패: {str(e)}")


@router.get("/cells", response_model=List[Dict[str, Any]])
async def get_cells(db: AsyncIOMotorDatabase = Depends(get_database)):
    """
    Cell 마스터 데이터 목록을 조회합니다.
    
    Returns:
        List[Dict]: Cell 마스터 데이터 목록
        - ne: NE 이름
        - cell_id: Cell ID
        - description: Cell 설명
    """
    try:
        logger.info("Cell 마스터 데이터 조회 시작")
        
        # MongoDB에서 cell_master 컬렉션 조회
        cursor = db.cell_master.find({}, {"_id": 0})  # _id 필드 제외
        cells = await cursor.to_list(length=None)
        
        logger.info(f"Cell 마스터 데이터 조회 완료: {len(cells)}개")
        
        # 데이터가 없으면 기본 하드코딩된 Cell 목록 반환
        if not cells:
            logger.warning("DB에 Cell 마스터 데이터가 없어 기본 목록 반환")
            cells = [
                {"ne": "eNB_001", "cell_id": "Cell_001", "description": "1호기 Cell 1"},
                {"ne": "eNB_001", "cell_id": "Cell_002", "description": "1호기 Cell 2"},
                {"ne": "eNB_002", "cell_id": "Cell_001", "description": "2호기 Cell 1"},
                {"ne": "eNB_002", "cell_id": "Cell_002", "description": "2호기 Cell 2"}
            ]
        
        return cells
        
    except Exception as e:
        logger.error(f"Cell 마스터 데이터 조회 오류: {str(e)}")
        raise DatabaseConnectionException(f"Cell 마스터 데이터 조회 실패: {str(e)}")


@router.get("/info")
async def get_master_info():
    """
    마스터 데이터 API 정보를 반환합니다.
    """
    return {
        "service": "Master Data API",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/api/master/pegs", "description": "PEG 마스터 데이터 조회"},
            {"path": "/api/master/cells", "description": "Cell 마스터 데이터 조회"}
        ]
    }


class PostgresConnectionConfig(BaseModel):
    """PostgreSQL 연결 테스트 요청 모델"""
    host: str = Field(..., description="DB Host")
    port: int = Field(default=5432, description="DB Port")
    user: str = Field(default="postgres", description="DB User")
    password: str = Field(default="", description="DB Password")
    dbname: str = Field(default="postgres", description="Database Name")
    table: Optional[str] = Field(default=None, description="존재 여부를 확인할 테이블명(선택)")


class DbPegRequest(BaseModel):
    """PostgreSQL에서 PEG 목록을 조회하기 위한 요청 모델"""
    host: str
    port: int = 5432
    user: str = "postgres"
    password: str = ""
    dbname: str = "postgres"
    table: str = "summary"
    limit: int = 500


class DbNeCellsRequest(BaseModel):
    """PostgreSQL에서 NE/Cell 목록을 조회하기 위한 요청 모델"""
    host: str
    port: int = 5432
    user: str = "postgres"
    password: str = ""
    dbname: str = "postgres"
    table: str = "summary"


class DbSuggestRequest(BaseModel):
    """NE/CellID 자동완성 및 목록 조회 공통 요청 모델"""
    host: str
    port: int = 5432
    user: str = "postgres"
    password: str = ""
    dbname: str = "postgres"
    table: str = "summary"
    # 컬럼 맵핑
    columns: Dict[str, str] | None = None
    # 기간/검색어/제한
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    q: Optional[str] = None
    limit: int = 100


class DbHierarchyBase(BaseModel):
    """PostgreSQL 연결/테이블 공통 모델"""
    host: str
    port: int = 5432
    user: str = "postgres"
    password: str = ""
    dbname: str = "postgres"
    table: str = "summary"


class HostsRequest(DbHierarchyBase):
    """Host 목록 조회 요청"""
    pass


class NesByHostRequest(DbHierarchyBase):
    """Host 기준으로 NE 목록 조회 요청"""
    hosts: List[str] = Field(default_factory=list, description="선택된 HOST 목록")


class CellsByHostNeRequest(DbHierarchyBase):
    """Host/NE 기준으로 CellID 목록 조회 요청"""
    hosts: List[str] = Field(default_factory=list)
    nes: List[str] = Field(default_factory=list)


@router.post("/test-connection")
async def test_postgres_connection(config: PostgresConnectionConfig) -> Dict[str, Any]:
    """
    PostgreSQL 연결을 테스트합니다.

    요청 본문에 포함된 접속 정보로 DB에 접속해 간단한 쿼리를 수행하고,
    선택적으로 특정 테이블 존재 여부를 확인합니다.
    """
    try:
        logger.info(
            "PostgreSQL 연결 테스트 시작: host=%s port=%s db=%s user=%s",
            config.host, config.port, config.dbname, config.user,
        )

        # 공백 제거 및 기본값 보정
        host = (config.host or "").strip()
        user = (config.user or "").strip()
        password = (config.password or "").strip()
        dbname = (config.dbname or "").strip()
        port = int(config.port)

        if not host:
            raise ValueError("DB Host가 비어있습니다. Preference > Database에서 올바른 host를 입력하세요 (컨테이너 환경: 'postgres').")

        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=dbname,
            connect_timeout=5,
        )
        cur = conn.cursor()
        cur.execute("SELECT 1")
        _ = cur.fetchone()

        table_exists: Optional[bool] = None
        if config.table:
            # PostgreSQL에서 테이블 존재 여부 확인
            cur.execute("SELECT to_regclass(%s)", (config.table,))
            table_exists = cur.fetchone()[0] is not None

        cur.close()
        conn.close()

        return {
            "success": True,
            "message": "Connection successful",
            "table_exists": table_exists,
        }
    except Exception as e:
        logger.error("PostgreSQL 연결 실패: %s", e)
        raise HTTPException(status_code=400, detail=f"Connection failed: {e}")


@router.post("/pegs")
async def get_db_pegs(req: DbPegRequest) -> Dict[str, Any]:
    """
    PostgreSQL에서 PEG(peg_name) 목록을 조회합니다. 중복 제거 후 최대 limit개 반환.
    """
    try:
        conn = psycopg2.connect(
            host=str(req.host).strip(),
            port=int(req.port),
            user=str(req.user).strip(),
            password=str(req.password).strip(),
            dbname=str(req.dbname).strip(),
            connect_timeout=5,
        )
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT DISTINCT peg_name
            FROM {req.table}
            WHERE peg_name IS NOT NULL
            ORDER BY peg_name ASC
            LIMIT %s
            """,
            (int(req.limit),)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()

        pegs = [{"id": r[0], "name": r[0]} for r in rows]
        # 호환성: 단순 배열만 기대하는 클라이언트도 지원
        return {"success": True, "pegs": pegs, "items": [r[0] for r in rows]}
    except Exception as e:
        logger.error(f"DB PEG 조회 실패: {e}")
        raise HTTPException(status_code=400, detail=f"PEG 조회 실패: {e}")


@router.post("/ne-cells")
async def get_db_ne_cells(req: DbNeCellsRequest) -> Dict[str, Any]:
    """
    PostgreSQL에서 NE/Cell 목록을 조회합니다. NE 우선 조회, 선택된 NE에 해당하는 Cell 목록 반환 지원.
    - 전체 NE 목록: SELECT DISTINCT ne
    - 특정 NE의 Cell 목록: SELECT DISTINCT cellid WHERE ne = %s
    """
    try:
        conn = psycopg2.connect(
            host=str(req.host).strip(),
            port=int(req.port),
            user=str(req.user).strip(),
            password=str(req.password).strip(),
            dbname=str(req.dbname).strip(),
            connect_timeout=5,
        )
        cur = conn.cursor()

        # 모든 NE 목록 조회
        cur.execute(f"SELECT DISTINCT ne FROM {req.table} WHERE ne IS NOT NULL ORDER BY ne ASC LIMIT 2000")
        nes = [r[0] for r in cur.fetchall()]

        # 각 NE에 대한 대표 cellid 0~N 중 일부만 선조회(선택 시 상세 조회 권장)
        sample_cells = {}
        cur.execute(f"SELECT ne, cellid FROM {req.table} WHERE ne IS NOT NULL AND cellid IS NOT NULL GROUP BY ne, cellid LIMIT 5000")
        for ne, cid in cur.fetchall():
            sample_cells.setdefault(ne, []).append(cid)

        cur.close()
        conn.close()

        return {"success": True, "nes": nes, "sampleCells": sample_cells}
    except Exception as e:
        logger.error(f"DB NE/Cell 조회 실패: {e}")
        raise HTTPException(status_code=400, detail=f"NE/Cell 조회 실패: {e}")


@router.post("/ne-list")
async def get_ne_list(req: DbSuggestRequest) -> Dict[str, Any]:
    """NE 목록/자동완성: DISTINCT ne with optional LIKE and date range."""
    try:
        ne_col = (req.columns or {}).get("ne", "ne")
        time_col = (req.columns or {}).get("time", "datetime")
        conn = psycopg2.connect(
            host=str(req.host).strip(), port=int(req.port), user=str(req.user).strip(),
            password=str(req.password).strip(), dbname=str(req.dbname).strip(), connect_timeout=5,
        )
        cur = conn.cursor()
        conds = [f"{ne_col} IS NOT NULL"]
        params = []
        if req.q:
            conds.append(f"{ne_col} ILIKE %s")
            params.append(f"%{req.q}%")
        if req.start_date and req.end_date:
            conds.append(f"{time_col} BETWEEN %s AND %s")
            params.extend([req.start_date, req.end_date])
        where = " AND ".join(conds)
        sql = f"SELECT DISTINCT {ne_col} FROM {req.table} WHERE {where} ORDER BY {ne_col} ASC LIMIT %s"
        params.append(int(req.limit))
        cur.execute(sql, tuple(params))
        items = [r[0] for r in cur.fetchall()]
        cur.close(); conn.close()
        return {"success": True, "items": items}
    except Exception as e:
        logger.error(f"NE 목록 조회 실패: {e}")
        raise HTTPException(status_code=400, detail=f"NE 목록 조회 실패: {e}")


@router.post("/cellid-list")
async def get_cellid_list(req: DbSuggestRequest) -> Dict[str, Any]:
    """CellID 목록/자동완성: DISTINCT cellid with optional LIKE and date range, optionally scoped by NE via q."""
    try:
        cell_col = (req.columns or {}).get("cellid", "cellid")
        ne_col = (req.columns or {}).get("ne", "ne")
        time_col = (req.columns or {}).get("time", "datetime")
        conn = psycopg2.connect(
            host=str(req.host).strip(), port=int(req.port), user=str(req.user).strip(),
            password=str(req.password).strip(), dbname=str(req.dbname).strip(), connect_timeout=5,
        )
        cur = conn.cursor()
        conds = [f"{cell_col} IS NOT NULL"]
        params = []
        # q가 제공되면 cellid 또는 ne 둘 다에 사용 가능하도록 간단 처리
        if req.q:
            conds.append(f"({cell_col}::text ILIKE %s OR {ne_col} ILIKE %s)")
            like = f"%{req.q}%"
            params.extend([like, like])
        if req.start_date and req.end_date:
            conds.append(f"{time_col} BETWEEN %s AND %s")
            params.extend([req.start_date, req.end_date])
        where = " AND ".join(conds)
        sql = f"SELECT DISTINCT {cell_col} FROM {req.table} WHERE {where} ORDER BY {cell_col} ASC LIMIT %s"
        params.append(int(req.limit))
        cur.execute(sql, tuple(params))
        items = [r[0] for r in cur.fetchall()]
        cur.close(); conn.close()
        return {"success": True, "items": items}
    except Exception as e:
        logger.error(f"CellID 목록 조회 실패: {e}")
        raise HTTPException(status_code=400, detail=f"CellID 목록 조회 실패: {e}")


@router.post("/hosts")
async def get_hosts(req: HostsRequest) -> Dict[str, Any]:
    """
    PostgreSQL에서 DISTINCT host 목록을 조회합니다.
    불필요한 날짜/검색 인자 없이 순수한 마스터 조회만 수행합니다.
    """
    try:
        conn = psycopg2.connect(
            host=str(req.host).strip(),
            port=int(req.port),
            user=str(req.user).strip(),
            password=str(req.password).strip(),
            dbname=str(req.dbname).strip(),
            connect_timeout=5,
        )
        cur = conn.cursor()
        cur.execute(f"SELECT DISTINCT host FROM {req.table} WHERE host IS NOT NULL ORDER BY host ASC LIMIT 5000")
        hosts = [r[0] for r in cur.fetchall()]
        cur.close(); conn.close()
        logger.info("HOST 목록 조회 완료: %d개", len(hosts))
        return {"success": True, "hosts": hosts}
    except Exception as e:
        logger.error("HOST 목록 조회 실패: %s", e)
        raise HTTPException(status_code=400, detail=f"HOST 조회 실패: {e}")


@router.post("/nes-by-host")
async def get_nes_by_host(req: NesByHostRequest) -> Dict[str, Any]:
    """
    선택된 HOST 범위 내에서 DISTINCT ne 목록을 조회합니다.
    """
    try:
        conn = psycopg2.connect(
            host=str(req.host).strip(),
            port=int(req.port),
            user=str(req.user).strip(),
            password=str(req.password).strip(),
            dbname=str(req.dbname).strip(),
            connect_timeout=5,
        )
        cur = conn.cursor()
        params: List[Any] = []
        where = ["ne IS NOT NULL"]
        if req.hosts:
            where.append("host = ANY(%s)")
            params.append(req.hosts)
        sql = f"SELECT DISTINCT ne FROM {req.table} WHERE {' AND '.join(where)} ORDER BY ne ASC LIMIT 5000"
        cur.execute(sql, tuple(params))
        nes = [r[0] for r in cur.fetchall()]
        cur.close(); conn.close()
        logger.info("NE 목록 조회 완료(host 필터 적용): %d개", len(nes))
        return {"success": True, "nes": nes}
    except Exception as e:
        logger.error("NE 목록 조회 실패: %s", e)
        raise HTTPException(status_code=400, detail=f"NE 조회 실패: {e}")


@router.post("/cells-by-host-ne")
async def get_cells_by_host_ne(req: CellsByHostNeRequest) -> Dict[str, Any]:
    """
    선택된 HOST/NE 범위 내에서 DISTINCT cellid 목록을 조회합니다.
    """
    try:
        conn = psycopg2.connect(
            host=str(req.host).strip(),
            port=int(req.port),
            user=str(req.user).strip(),
            password=str(req.password).strip(),
            dbname=str(req.dbname).strip(),
            connect_timeout=5,
        )
        cur = conn.cursor()
        params: List[Any] = []
        where = ["cellid IS NOT NULL"]
        if req.hosts:
            where.append("host = ANY(%s)")
            params.append(req.hosts)
        if req.nes:
            where.append("ne = ANY(%s)")
            params.append(req.nes)
        sql = f"SELECT DISTINCT cellid FROM {req.table} WHERE {' AND '.join(where)} ORDER BY cellid ASC LIMIT 10000"
        cur.execute(sql, tuple(params))
        cells = [r[0] for r in cur.fetchall()]
        cur.close(); conn.close()
        logger.info("CellID 목록 조회 완료(host/ne 필터 적용): %d개", len(cells))
        return {"success": True, "cellids": cells}
    except Exception as e:
        logger.error("CellID 목록 조회 실패: %s", e)
        raise HTTPException(status_code=400, detail=f"CellID 조회 실패: {e}")
