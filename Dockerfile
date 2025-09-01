# === 개발 및 기본 빌드 ===
FROM python:3.11-slim AS base

# 기본 환경 변수 설정: 버퍼링 비활성화, 바이트코드 생성 비활성화
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 시스템 의존성 설치 (curl for health checks)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 필요한 파일만 먼저 복사하여 캐시 레이어 극대화
COPY requirements.txt /app/requirements.txt

# 의존성 설치 (깨끗한 격리 환경)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# 애플리케이션 소스 복사
COPY . /app

# 로그 디렉토리 생성
RUN mkdir -p /app/logs /app/data

# 개발용 설정
EXPOSE 8000
CMD ["sh", "-c", "python wait_for_db.py && uvicorn main:app --host 0.0.0.0 --port 8000 --reload"]

# === 프로덕션 빌드 ===
FROM base AS production

# 프로덕션 환경 변수
ENV ENVIRONMENT=production \
    PYTHONPATH=/app \
    LOG_LEVEL=INFO

# 비루트 사용자 생성 (보안)
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 디렉토리 권한 설정
RUN chown -R appuser:appuser /app
USER appuser

# 프로덕션용 실행 명령
EXPOSE 8000
CMD ["sh", "-c", "python wait_for_db.py && uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4 --access-log"]

