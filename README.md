# 3GPP KPI Dashboard Backend

## 📋 개요

3GPP KPI 대시보드 시스템의 백엔드 서비스입니다. FastAPI 기반으로 구축되어 있으며, PostgreSQL과 MongoDB를 연동하여 KPI 데이터를 처리하고 분석 결과를 제공합니다.

## 🏗️ 아키텍처

### 시스템 구성

```
Frontend (React) ↔ Backend (FastAPI) ↔ PostgreSQL (Raw KPI Data)
                                    ↔ MongoDB (Analysis Results)
                                    ↔ MCP (LLM Analysis)
```

### 주요 구성요소

1. **FastAPI**: RESTful API 서버
2. **PostgreSQL**: Raw KPI/PEG 데이터 저장소
3. **MongoDB**: 분석 결과, 사용자 설정, 통계 결과 저장소
4. **MCP**: LLM 분석 서비스 (별도 환경)

## 🔧 구현된 기능

### 1) API 엔드포인트

#### KPI 조회 API

- **POST `/api/kpi/query`**: KPI 데이터 조회
- **POST `/api/kpi/statistics/batch`**: 여러 KPI 동시 조회

#### 분석 결과 API

- **POST `/api/analysis/trigger-llm-analysis`**: LLM 분석 트리거
- **GET `/api/analysis/llm-analysis/{id}`**: 분석 결과 조회
- **POST `/api/analysis/results`**: 분석 결과 생성
- **GET `/api/analysis/results`**: 분석 결과 목록 조회
- **GET `/api/analysis/results/{id}`**: 단일 분석 결과 상세 조회
- **PUT `/api/analysis/results/{id}`**: 분석 결과 업데이트
- **DELETE `/api/analysis/results/{id}`**: 분석 결과 삭제

#### 비동기 분석 API

- **POST `/api/async-analysis/start`**: 비동기 분석 시작
- **GET `/api/async-analysis/status/{id}`**: 분석 상태 조회
- **GET `/api/async-analysis/result/{id}`**: 분석 결과 조회
- **POST `/api/async-analysis/cancel/{id}`**: 분석 취소
- **GET `/api/async-analysis/list`**: 실행 중인 작업 목록
- **GET `/api/async-analysis/health`**: 서비스 상태 확인

#### 기타 API

- **GET `/api/master/pegs`**: PEG 마스터 데이터 조회
- **GET `/api/preference`**: 사용자 설정 조회
- **POST `/api/preference`**: 사용자 설정 저장

### 2) 데이터 모델

#### AnalysisResultModel

```python
class AnalysisResultModel(BaseModel):
    ne_id: str
    cell_id: str
    analysis_date: datetime
    status: str
    time_ranges: Dict[str, Any]
    peg_metrics: PegMetricsPayload
    llm_analysis: LLMAnalysisSummary
    metadata: AnalysisMetadataPayload
    legacy_payload: Optional[Dict[str, Any]]
```

#### KPI Query Request

```python
class KPIQueryRequest(BaseModel):
    start_date: str
    end_date: str
    kpi_types: List[str]
    ne: Optional[str] = None
    cellid: Optional[str] = None
```

### 3) 서비스 레이어

#### AsyncAnalysisService

- 비동기 분석 작업 관리
- 백그라운드 작업 실행
- 상태 추적 및 결과 저장

#### MCPClientService

- MCP 서비스와의 통신
- 분석 요청 및 결과 처리

## ⚙️ 설정 및 배포

### 필수 의존성

```txt
# FastAPI 및 웹 프레임워크
fastapi
uvicorn
pydantic

# 데이터베이스
psycopg2-binary
pymongo
motor

# 데이터 처리
pandas
numpy
scipy

# HTTP 통신
requests
httpx
```

### 환경 변수

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

### Docker 배포

```bash
# Docker Compose로 전체 시스템 실행
docker compose up -d

# 백엔드만 실행
cd backend
docker build -t kpi-backend .
docker run -p 8000:8000 kpi-backend
```

## 🧪 테스트 방법

### 1) API 테스트 (PowerShell)

```powershell
# LLM 분석 요청
$body = '{"user_id":"default", "n_minus_1":"2024-01-01_00:00~2024-01-01_23:59", "n":"2024-01-02_00:00~2024-01-02_23:59", "enable_mock": false}'
$response = Invoke-RestMethod -Uri "http://localhost:8000/api/analysis/trigger-llm-analysis" -Method POST -Body $body -ContentType "application/json"

# 결과 조회
$result = Invoke-RestMethod -Uri "http://localhost:8000/api/analysis/llm-analysis/$($response.analysis_id)" -Method GET
```

### 2) 비동기 분석 테스트

```python
# 비동기 분석 시작
import requests

response = requests.post("http://localhost:8000/api/async-analysis/start",
                        json=request_data)
analysis_id = response.json()["analysis_id"]

# 상태 확인
status = requests.get(f"http://localhost:8000/api/async-analysis/status/{analysis_id}")
print(status.json())
```

## 🔍 데이터 흐름

1. **Frontend → Backend**: API 요청
2. **Backend → PostgreSQL**: Raw KPI 데이터 쿼리
3. **Backend → MCP**: LLM 분석 요청
4. **Backend → MongoDB**: 분석 결과 저장
5. **Backend → Frontend**: 결과 반환

## 📊 성능 최적화

### 1) 데이터베이스 최적화

- PostgreSQL 인덱스 최적화
- MongoDB 쿼리 최적화
- 연결 풀링 설정

### 2) API 성능

- 응답 캐싱
- 비동기 처리
- 배치 처리

### 3) 리소스 관리

- 메모리 사용량 모니터링
- CPU 사용률 최적화
- 디스크 I/O 최적화

## ⚠️ 현재 제한사항

- MCP 미설정/오류 시 Mock 폴백(자동)
- 실시간 상태는 폴링 기반(추후 SSE/WebSocket 가능)

## 🔄 향후 개선

- 실시간 스트리밍 업데이트(SSE/WebSocket)
- 권장사항/원인분석 자동 생성 강화
- 대량 KPI 성능 튜닝(서버/클라이언트)
- 마이크로서비스 아키텍처 전환

_문서 업데이트: 2025-01-14 (DTO 구조 및 비동기 처리 반영)_
