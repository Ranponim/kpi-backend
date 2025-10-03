import os
import logging
from datetime import datetime
from typing import List, Dict, Any
import psycopg2
import psycopg2.extras
from contextlib import contextmanager

logger = logging.getLogger(__name__)

def get_db_connection_details():
    """
    환경 변수 또는 기본값에서 DB 연결 정보를 가져옵니다.
    """
    return {
        "host": os.getenv("DB_HOST", "postgres"),
        "port": os.getenv("DB_PORT", 5432),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "pass"),
        "dbname": os.getenv("DB_NAME", "netperf"),
    }

@contextmanager
def get_db_connection():
    """
    PostgreSQL 데이터베이스 연결을 위한 컨텍스트 관리자를 제공합니다.
    """
    conn = None
    try:
        db_details = get_db_connection_details()
        conn = psycopg2.connect(**db_details)
        logger.info(f"PostgreSQL DB 연결 성공: host={db_details['host']}, dbname={db_details['dbname']}")
        yield conn
    except psycopg2.OperationalError as e:
        logger.error(f"PostgreSQL DB 연결 실패: {e}")
        raise
    finally:
        if conn:
            conn.close()
            logger.info("PostgreSQL DB 연결 종료")

def query_kpi_data(
    start_date: str,
    end_date: str,
    kpi_types: List[str],
    ne_filters: List[str] = None,
    cellid_filters: List[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    PostgreSQL에서 여러 KPI에 대한 시계열 데이터를 조회합니다.
    """
    logger.info(f"PostgreSQL KPI 데이터 조회 시작: {len(kpi_types)}개 KPI")

    # 새 스키마 지원: summary(datetime, family_id, family_name, ne_key, rel_ver, name, values(jsonb), version)
    # values 형태 2가지 지원
    # 1) 평면: { "peg": "value", ..., "index_name":"..." }
    # 2) 중첩: { "<cellid>": { "peg": "value" }, ..., "index_name":"CellIdentity" }

    # 평면/중첩을 모두 행으로 펼친 뒤 공통 WHERE 절을 적용
    query = """
        WITH flat AS (
            SELECT
                s.datetime AS timestamp,
                s.ne_key AS ne,
                NULL::int AS cell_id,
                s.ne_key AS entity_id,
                jt.key AS peg_name,
                NULLIF(jt.value, '')::numeric AS value,
                to_char(s.datetime, 'YYYY-MM-DD') AS date,
                extract(hour from s.datetime) AS hour
            FROM summary s
            CROSS JOIN LATERAL jsonb_each_text(s.values) AS jt(key, value)
            WHERE s.datetime BETWEEN %s AND %s
              AND jt.key <> 'index_name'
        ),
        nested AS (
            SELECT
                s.datetime AS timestamp,
                s.ne_key AS ne,
                NULLIF(l1.key, '')::int AS cell_id,
                s.ne_key || '#' || l1.key AS entity_id,
                jt.key AS peg_name,
                NULLIF(jt.value, '')::numeric AS value,
                to_char(s.datetime, 'YYYY-MM-DD') AS date,
                extract(hour from s.datetime) AS hour
            FROM summary s
            CROSS JOIN LATERAL jsonb_each(s.values) AS l1(key, value)
            JOIN LATERAL jsonb_each_text(l1.value) AS jt(key, value)
                 ON jsonb_typeof(l1.value) = 'object'
            WHERE s.datetime BETWEEN %s AND %s
        )
        SELECT *
        FROM (
            SELECT * FROM flat
            UNION ALL
            SELECT * FROM nested
        ) x
        WHERE x.peg_name = ANY(%s)
    """

    params = [start_date, end_date, start_date, end_date, kpi_types]

    if ne_filters:
        query += " AND x.ne = ANY(%s)"
        params.append(ne_filters)

    if cellid_filters:
        # cellid_filters를 정수 배열로 변환 (중첩 구조에서만 의미 있음)
        cellid_ints = [int(cellid) for cellid in cellid_filters if str(cellid).isdigit()]
        if cellid_ints:
            query += " AND x.cell_id = ANY(%s)"
            params.append(cellid_ints)

    query += " ORDER BY x.timestamp ASC;"

    logger.info(f"PostgreSQL 쿼리: {query}")
    logger.info(f"PostgreSQL 파라미터: {params}")

    data_by_kpi = {kpi: [] for kpi in kpi_types}

    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(query, tuple(params))
                rows = cur.fetchall()

                logger.info(f"PostgreSQL 조회 결과: {len(rows)}개 행")

                for row in rows:
                    row_dict = dict(row)
                    # datetime 객체를 ISO 8601 형식의 문자열로 변환
                    if isinstance(row_dict.get('timestamp'), datetime):
                        row_dict['timestamp'] = row_dict['timestamp'].isoformat()

                    kpi_type = row_dict['kpi_type']
                    if kpi_type in data_by_kpi:
                        data_by_kpi[kpi_type].append(row_dict)

        logger.info(f"PostgreSQL 조회 완료: {sum(len(v) for v in data_by_kpi.values())}개 레코드")
        return data_by_kpi

    except Exception as e:
        logger.error(f"PostgreSQL KPI 데이터 조회 중 오류 발생: {e}", exc_info=True)
        # 오류 발생 시 빈 데이터를 반환하여 API가 중단되지 않도록 함
        return data_by_kpi


def query_kpi_time_series(
    start_date: str,
    end_date: str,
    kpi_types: List[str],
    ne_filters: List[str] | None = None,
    cellid_filters: List[str] | None = None,
    aggregate_cells_if_missing: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    시간 범위 내 KPI 시계열 조회.
    - cellid_filters가 비어있고 aggregate_cells_if_missing=True인 경우, 동일 NE 내 Cell들을 평균 집계하여 반환합니다.
    - 그렇지 않으면 개별 Cell 시계열을 반환합니다.
    """
    logger.info(
        "KPI TimeSeries 조회 시작: %s ~ %s, kpis=%d, ne=%d, cell=%d, aggregate=%s",
        start_date,
        end_date,
        len(kpi_types or []),
        len(ne_filters or []),
        len(cellid_filters or []),
        aggregate_cells_if_missing,
    )

    data_by_kpi: Dict[str, List[Dict[str, Any]]] = {k: [] for k in (kpi_types or [])}

    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                aggregate_mode = aggregate_cells_if_missing and not cellid_filters

                base_params: List[Any] = [start_date, end_date, start_date, end_date, kpi_types]

                ne_clause = ""
                ne_params: List[Any] = []
                if ne_filters:
                    ne_clause = " AND x.ne = ANY(%s)"
                    ne_params.append(ne_filters)

                cell_clause = ""
                cell_params: List[Any] = []
                if not aggregate_mode and cellid_filters:
                    cellid_ints = [int(cellid) for cellid in cellid_filters if str(cellid).isdigit()]
                    if cellid_ints:
                        cell_clause = " AND x.cell_id = ANY(%s)"
                        cell_params.append(cellid_ints)

                if aggregate_mode:
                    # 평면/중첩을 펼친 뒤 NE 기준 평균 집계
                    sql = """
                        WITH flat AS (
                            SELECT
                                s.datetime AS timestamp,
                                s.ne_key AS ne,
                                NULL::int AS cell_id,
                                s.ne_key AS entity_id,
                                jt.key AS peg_name,
                                NULLIF(jt.value, '')::numeric AS value
                            FROM summary s
                            CROSS JOIN LATERAL jsonb_each_text(s.values) AS jt(key, value)
                            WHERE s.datetime BETWEEN %s AND %s
                              AND jt.key <> 'index_name'
                        ),
                        nested AS (
                            SELECT
                                s.datetime AS timestamp,
                                s.ne_key AS ne,
                                NULLIF(l1.key, '')::int AS cell_id,
                                s.ne_key || '#' || l1.key AS entity_id,
                                jt.key AS peg_name,
                                NULLIF(jt.value, '')::numeric AS value
                            FROM summary s
                            CROSS JOIN LATERAL jsonb_each(s.values) AS l1(key, value)
                            JOIN LATERAL jsonb_each_text(l1.value) AS jt(key, value)
                                 ON jsonb_typeof(l1.value) = 'object'
                            WHERE s.datetime BETWEEN %s AND %s
                        )
                        SELECT
                            x.timestamp,
                            x.peg_name,
                            AVG(x.value) AS value,
                            x.ne,
                            NULL::int AS cell_id,
                            x.ne AS entity_id
                        FROM (
                            SELECT * FROM flat
                            UNION ALL
                            SELECT * FROM nested
                        ) x
                        WHERE x.peg_name = ANY(%s)
                    """ + ne_clause + "\n                        GROUP BY x.timestamp, x.peg_name, x.ne\n                        ORDER BY x.timestamp ASC\n                    "

                    params = base_params + ne_params
                else:
                    # 개별 Cell 시계열 (중첩 구조 우선, 평면 구조는 cell_id NULL)
                    sql = """
                        WITH flat AS (
                            SELECT
                                s.datetime AS timestamp,
                                s.ne_key AS ne,
                                NULL::int AS cell_id,
                                s.ne_key AS entity_id,
                                jt.key AS peg_name,
                                NULLIF(jt.value, '')::numeric AS value
                            FROM summary s
                            CROSS JOIN LATERAL jsonb_each_text(s.values) AS jt(key, value)
                            WHERE s.datetime BETWEEN %s AND %s
                              AND jt.key <> 'index_name'
                        ),
                        nested AS (
                            SELECT
                                s.datetime AS timestamp,
                                s.ne_key AS ne,
                                NULLIF(l1.key, '')::int AS cell_id,
                                s.ne_key || '#' || l1.key AS entity_id,
                                jt.key AS peg_name,
                                NULLIF(jt.value, '')::numeric AS value
                            FROM summary s
                            CROSS JOIN LATERAL jsonb_each(s.values) AS l1(key, value)
                            JOIN LATERAL jsonb_each_text(l1.value) AS jt(key, value)
                                 ON jsonb_typeof(l1.value) = 'object'
                            WHERE s.datetime BETWEEN %s AND %s
                        )
                        SELECT
                            x.timestamp,
                            x.peg_name,
                            x.value,
                            x.ne,
                            x.cell_id,
                            CASE WHEN x.cell_id IS NULL THEN x.ne ELSE x.ne || '#' || x.cell_id::text END AS entity_id
                        FROM (
                            SELECT * FROM flat
                            UNION ALL
                            SELECT * FROM nested
                        ) x
                        WHERE x.peg_name = ANY(%s)
                    """ + ne_clause + cell_clause + "\n                        ORDER BY x.timestamp ASC\n                    "

                    params = base_params + ne_params + cell_params

                cur.execute(sql, tuple(params))
                rows = cur.fetchall()

                for row in rows:
                    row_dict = dict(row)
                    if isinstance(row_dict.get('timestamp'), datetime):
                        row_dict['timestamp'] = row_dict['timestamp'].isoformat()
                    k = row_dict.get('peg_name')
                    if k in data_by_kpi:
                        data_by_kpi[k].append(row_dict)

        logger.info(
            "KPI TimeSeries 조회 완료: %d rows",
            sum(len(v) for v in data_by_kpi.values()),
        )
        return data_by_kpi

    except Exception as e:
        logger.error("KPI TimeSeries 조회 오류: %s", e, exc_info=True)
        return data_by_kpi