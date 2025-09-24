# Docker 로컬 배포 가이드

## 📋 개요

3GPP KPI 대시보드 시스템을 Docker를 사용하여 로컬 환경에 배포하는 방법을 설명합니다.

## 🏗️ 시스템 구성

### 컨테이너 구성

- **Frontend**: React (Vite) - http://localhost:5173
- **Backend**: FastAPI - http://localhost:8000
- **PostgreSQL**: Raw KPI 데이터 - localhost:5432 (netperf DB)
- **MongoDB**: Backend 저장소 - localhost:27017 (kpi DB)

### 네트워크 구성

```
Frontend (5173) ↔ Backend (8000) ↔ PostgreSQL (5432)
                              ↔ MongoDB (27017)
```

## 🚀 배포 방법

### 1) 전체 시스템 배포

```bash
# 1. 프로젝트 루트에서 실행
docker compose up -d

# 2. 컨테이너 상태 확인
docker compose ps

# 3. 로그 확인
docker compose logs -f
```

### 2) 개별 서비스 배포

#### Backend만 배포

```bash
cd backend
docker build -t kpi-backend .
docker run -p 8000:8000 \
  -e DB_HOST=localhost \
  -e DB_PORT=5432 \
  -e DB_USER=postgres \
  -e DB_PASSWORD=password \
  -e DB_NAME=netperf \
  -e MONGO_URL=mongodb://localhost:27017 \
  -e MONGO_DB_NAME=kpi \
  kpi-backend
```

#### Frontend만 배포

```bash
cd frontend
docker build -t kpi-frontend .
docker run -p 5173:5173 kpi-frontend
```

## ⚙️ 환경 설정

### 환경 변수

#### Backend 환경 변수

```bash
# PostgreSQL (Raw KPI Data)
DB_HOST=postgres
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=pass
DB_NAME=netperf

# MongoDB (Backend Storage)
MONGO_URL=mongodb://mongo:27017
MONGO_DB_NAME=kpi

# MCP (옵션)
MCP_ANALYZER_URL=http://mcp-host:8001/analyze
MCP_API_KEY=xxx
```

#### Frontend 환경 변수

```bash
# API 서버 URL
VITE_API_BASE_URL=http://localhost:8000
```

### Docker Compose 설정

```yaml
version: "3.8"

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: netperf
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: pass
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  mongo:
    image: mongo:7
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      DB_HOST: postgres
      DB_PORT: 5432
      DB_USER: postgres
      DB_PASSWORD: pass
      DB_NAME: netperf
      MONGO_URL: mongodb://mongo:27017
      MONGO_DB_NAME: kpi
    depends_on:
      - postgres
      - mongo

  frontend:
    build: ./frontend
    ports:
      - "5173:5173"
    environment:
      VITE_API_BASE_URL: http://localhost:8000
    depends_on:
      - backend

volumes:
  postgres_data:
  mongo_data:
```

## 🧪 테스트 및 검증

### 1) 서비스 상태 확인

```bash
# 컨테이너 상태 확인
docker compose ps

# 서비스 로그 확인
docker compose logs backend
docker compose logs frontend
docker compose logs postgres
docker compose logs mongo
```

### 2) API 테스트

```bash
# Backend API 테스트
curl http://localhost:8000/health

# KPI API 테스트
curl -X POST http://localhost:8000/api/kpi/query \
  -H "Content-Type: application/json" \
  -d '{"start_date": "2025-01-01", "end_date": "2025-01-02", "kpi_types": ["availability"]}'
```

### 3) 데이터베이스 연결 확인

```bash
# PostgreSQL 연결 확인
docker exec -it kpi-postgres psql -U postgres -d netperf -c "SELECT 1;"

# MongoDB 연결 확인
docker exec -it kpi-mongo mongosh --eval "db.runCommand({ping: 1})"
```

## 🔧 개발 환경 설정

### 1) 개발 모드 실행

```bash
# Backend 개발 모드
cd backend
docker compose -f docker-compose.dev.yml up -d

# Frontend 개발 모드
cd frontend
npm run dev
```

### 2) 디버깅

```bash
# 컨테이너 내부 접속
docker exec -it kpi-backend bash
docker exec -it kpi-frontend sh

# 로그 실시간 모니터링
docker compose logs -f backend
```

## 📊 모니터링

### 1) 리소스 사용량 확인

```bash
# 컨테이너 리소스 사용량
docker stats

# 특정 컨테이너 리소스 사용량
docker stats kpi-backend kpi-frontend
```

### 2) 로그 모니터링

```bash
# 모든 서비스 로그
docker compose logs -f

# 특정 서비스 로그
docker compose logs -f backend
docker compose logs -f frontend
```

## 🚨 문제 해결

### 1) 일반적인 문제

#### 포트 충돌

```bash
# 포트 사용 중인 프로세스 확인
netstat -tulpn | grep :8000
netstat -tulpn | grep :5173

# 프로세스 종료
kill -9 <PID>
```

#### 컨테이너 재시작

```bash
# 특정 서비스 재시작
docker compose restart backend

# 전체 시스템 재시작
docker compose down
docker compose up -d
```

### 2) 데이터베이스 문제

#### PostgreSQL 연결 실패

```bash
# PostgreSQL 컨테이너 상태 확인
docker compose ps postgres

# PostgreSQL 로그 확인
docker compose logs postgres

# 데이터베이스 재생성
docker compose down -v
docker compose up -d
```

#### MongoDB 연결 실패

```bash
# MongoDB 컨테이너 상태 확인
docker compose ps mongo

# MongoDB 로그 확인
docker compose logs mongo

# 데이터베이스 재생성
docker compose down -v
docker compose up -d
```

## 🔄 업데이트 및 배포

### 1) 코드 업데이트

```bash
# 코드 변경 후 재빌드
docker compose build

# 서비스 재시작
docker compose up -d
```

### 2) 데이터베이스 마이그레이션

```bash
# 백업 생성
docker exec kpi-postgres pg_dump -U postgres netperf > backup.sql

# 마이그레이션 실행
docker exec kpi-backend python -m alembic upgrade head
```

## 📝 주의사항

1. **데이터 보존**: `docker compose down -v` 명령은 볼륨을 삭제하므로 주의
2. **포트 충돌**: 기존에 실행 중인 서비스와 포트가 충돌하지 않도록 확인
3. **환경 변수**: 프로덕션 환경에서는 보안을 위해 환경 변수를 안전하게 관리
4. **리소스**: Docker 컨테이너가 충분한 메모리와 CPU를 할당받을 수 있도록 확인

_문서 업데이트: 2025-01-14 (Docker Compose 및 환경 설정 반영)_
