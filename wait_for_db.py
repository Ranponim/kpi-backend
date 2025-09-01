import os
import time
import logging
from pymongo import MongoClient
import psycopg2


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_mongo_url():
    """환경변수에서 Mongo URL을 가져옵니다."""
    return os.getenv("MONGO_URL", "mongodb://mongo:27017")


def wait_for_db(max_attempts: int = 30, sleep_seconds: float = 1.0) -> None:
    """
    MongoDB와 PostgreSQL 모두 접속 가능할 때까지 대기합니다.

    - MongoDB: 환경변수 MONGO_URL 기준
    - PostgreSQL: 환경변수(DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME) 기준
    """
    mongo_url = get_mongo_url()

    # 1) MongoDB 대기
    for attempt in range(1, max_attempts + 1):
        try:
            logging.info("Mongo 연결 확인 %d/%d: %s", attempt, max_attempts, mongo_url)
            client = MongoClient(mongo_url, serverSelectionTimeoutMS=3000)
            client.admin.command("ping")
            logging.info("Mongo 연결 성공")
            break
        except Exception as e:
            logging.warning("Mongo 연결 대기 중: %s", e)
            time.sleep(sleep_seconds)
    else:
        raise RuntimeError("Mongo 준비 대기 시간 초과")

    # 2) PostgreSQL 대기
    db_host = os.getenv("DB_HOST", "postgres")
    db_port = int(os.getenv("DB_PORT", "5432"))
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "pass")
    db_name = os.getenv("DB_NAME", "netperf")

    for attempt in range(1, max_attempts + 1):
        try:
            logging.info(
                "PostgreSQL 연결 확인 %d/%d: host=%s port=%s db=%s",
                attempt, max_attempts, db_host, db_port, db_name,
            )
            conn = psycopg2.connect(
                host=db_host,
                port=db_port,
                user=db_user,
                password=db_password,
                dbname=db_name,
                connect_timeout=3,
            )
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                _ = cur.fetchone()
            conn.close()
            logging.info("PostgreSQL 연결 성공")
            break
        except Exception as e:
            logging.warning("PostgreSQL 연결 대기 중: %s", e)
            time.sleep(sleep_seconds)
    else:
        raise RuntimeError("PostgreSQL 준비 대기 시간 초과")


if __name__ == "__main__":
    wait_for_db()


