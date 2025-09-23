# 백엔드 PEG 비교분석 API 구현 가이드

## 개요

이 문서는 PEG(Performance Engineering Guidelines) 비교분석 기능을 위한 백엔드 API 구현 가이드입니다. 백엔드는 HTTP API 엔드포인트 제공, 사용자 인증, MCP 서버 통신, 캐싱, 에러 처리 등의 인프라 역할을 담당합니다.

**⚠️ 아키텍처 분리**: 이 문서는 백엔드 API 구현에 집중하며, MCP 서버의 PEG 비교분석 알고리즘 구현은 [PEG_COMPARISON_ANALYSIS.md](./PEG_COMPARISON_ANALYSIS.md)를 참조하세요.

## 시스템 아키텍처

```mermaid
graph TB
    FE[프론트엔드] --> BE[백엔드 API]
    BE --> MCP[MCP 서버]
    BE --> DB[(데이터베이스)]
    BE --> CACHE[(Redis 캐시)]
    BE --> AUTH[인증 서비스]

    subgraph "백엔드 API"
        BE1[/api/analysis/results/{id}/peg-comparison]
        BE2[MCP 호출 로직]
        BE3[결과 캐싱]
        BE4[에러 처리]
        BE5[인증 및 권한]
        BE6[Rate Limiting]
    end
```

## API 엔드포인트 명세

### 1. PEG 비교분석 결과 조회

```http
GET /api/analysis/results/{id}/peg-comparison
Authorization: Bearer {user_token}
```

**쿼리 파라미터:**

- `include_metadata`: boolean (기본값: true)
- `cache_ttl`: integer (기본값: 3600)
- `async`: boolean (기본값: false)

**응답:**

```json
{
  "success": true,
  "data": {
    "analysis_id": "result_123",
    "peg_comparison_results": [...],
    "summary": {...},
    "analysis_metadata": {...}
  },
  "processing_time": 1.23,
  "cached": false,
  "mcp_version": "v2.1.0"
}
```

### 2. 비동기 작업 상태 조회

```http
GET /api/analysis/results/{id}/peg-comparison/status
Authorization: Bearer {user_token}
```

**응답:**

```json
{
  "success": true,
  "task_id": "task_456",
  "status": "COMPLETED",
  "progress": 100,
  "estimated_completion": "2024-01-15T10:35:00Z"
}
```

## 백엔드-MCP 통신 프로토콜

### 1. MCP 서버 호출 구조

```javascript
// 백엔드에서 MCP로 전달하는 데이터
const mcpRequest = {
  analysis_id: "result_123",
  raw_data: {
    stats: result.stats, // 원시 KPI 데이터
    peg_definitions: result.request_params.peg_definitions,
    period_info: {
      n1_start: "2024-01-01T00:00:00Z",
      n1_end: "2024-01-07T23:59:59Z",
      n_start: "2024-01-08T00:00:00Z",
      n_end: "2024-01-14T23:59:59Z",
    },
  },
  options: {
    include_metadata: true,
    cache_ttl: 3600,
    algorithm_version: "v2.1.0",
  },
};
```

### 2. 에러 처리 및 재시도 로직

```javascript
// 백엔드에서 MCP 호출 시 에러 처리
const callMCPWithRetry = async (request, maxRetries = 3) => {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const response = await mcpClient.post(
        "/mcp/peg-comparison/analysis",
        request
      );
      return response.data;
    } catch (error) {
      if (attempt === maxRetries) {
        throw new Error(`MCP 호출 실패: ${error.message}`);
      }

      // 지수 백오프
      const delay = Math.pow(2, attempt) * 1000;
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
};
```

## 데이터베이스 설계

### 1. PEG 비교분석 결과 저장 테이블

```sql
-- PEG 비교분석 결과 저장 테이블
CREATE TABLE peg_comparison_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id VARCHAR(255) NOT NULL,
    peg_name VARCHAR(255) NOT NULL,
    weight INTEGER NOT NULL,
    n1_period_data JSONB NOT NULL,
    n_period_data JSONB NOT NULL,
    comparison_data JSONB NOT NULL,
    metadata JSONB NOT NULL,
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 인덱스 설정
CREATE INDEX idx_peg_comparison_analysis_id ON peg_comparison_results(analysis_id);
CREATE INDEX idx_peg_comparison_peg_name ON peg_comparison_results(peg_name);
CREATE INDEX idx_peg_comparison_calculated_at ON peg_comparison_results(calculated_at);
```

### 2. 요약 통계 저장 테이블

```sql
-- 요약 통계 저장 테이블
CREATE TABLE peg_comparison_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id VARCHAR(255) UNIQUE NOT NULL,
    summary_data JSONB NOT NULL,
    analysis_metadata JSONB NOT NULL,
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 인덱스 설정
CREATE INDEX idx_peg_summary_analysis_id ON peg_comparison_summaries(analysis_id);
CREATE INDEX idx_peg_summary_calculated_at ON peg_comparison_summaries(calculated_at);
```

### 3. 사용자 권한 관리 테이블

```sql
-- 사용자 권한 관리 테이블
CREATE TABLE user_analysis_permissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    analysis_id VARCHAR(255) NOT NULL,
    permission_type VARCHAR(50) NOT NULL, -- 'read', 'write', 'admin'
    granted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    granted_by VARCHAR(255) NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 인덱스 설정
CREATE INDEX idx_user_permissions_user_id ON user_analysis_permissions(user_id);
CREATE INDEX idx_user_permissions_analysis_id ON user_analysis_permissions(analysis_id);
CREATE UNIQUE INDEX idx_user_permissions_unique ON user_analysis_permissions(user_id, analysis_id, permission_type);
```

## 캐싱 전략

### 1. Redis 기반 캐싱 구현

```python
import redis
import json
from datetime import datetime, timedelta

class PEGComparisonCache:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.cache_ttl = {
            'peg_comparison': 3600,  # 1시간
            'summary_stats': 1800,   # 30분
            'metadata': 7200         # 2시간
        }

    def get_cache_key(self, analysis_id, cache_type):
        """캐시 키 생성"""
        return f"peg_comparison:{cache_type}:{analysis_id}"

    def cache_result(self, analysis_id, result, cache_type):
        """결과 캐싱"""
        cache_key = self.get_cache_key(analysis_id, cache_type)
        ttl = self.cache_ttl.get(cache_type, 3600)

        cache_data = {
            'data': result,
            'cached_at': datetime.utcnow().isoformat(),
            'ttl': ttl
        }

        self.redis.setex(cache_key, ttl, json.dumps(cache_data))

    def get_cached_result(self, analysis_id, cache_type):
        """캐시된 결과 조회"""
        cache_key = self.get_cache_key(analysis_id, cache_type)
        cached_data = self.redis.get(cache_key)

        if cached_data:
            return json.loads(cached_data)
        return None

    def invalidate_cache(self, analysis_id):
        """캐시 무효화"""
        patterns = [
            f"peg_comparison:*:{analysis_id}",
            f"peg_comparison:summary:{analysis_id}",
            f"peg_comparison:metadata:{analysis_id}"
        ]

        for pattern in patterns:
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
```

### 2. 캐시 전략 설정

```python
class CacheStrategy:
    def __init__(self):
        self.cache_policies = {
            'hot_data': {
                'ttl': 1800,  # 30분
                'max_size': 1000,
                'eviction_policy': 'lru'
            },
            'warm_data': {
                'ttl': 3600,  # 1시간
                'max_size': 500,
                'eviction_policy': 'lru'
            },
            'cold_data': {
                'ttl': 7200,  # 2시간
                'max_size': 100,
                'eviction_policy': 'lru'
            }
        }

    def get_cache_policy(self, data_type, access_frequency):
        """데이터 타입과 접근 빈도에 따른 캐시 정책 결정"""
        if access_frequency > 100:  # 높은 접근 빈도
            return self.cache_policies['hot_data']
        elif access_frequency > 10:  # 중간 접근 빈도
            return self.cache_policies['warm_data']
        else:  # 낮은 접근 빈도
            return self.cache_policies['cold_data']
```

## 보안 구현

### 1. JWT 기반 인증

```python
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity
from functools import wraps

def require_permission(permission_type):
    """권한 검증 데코레이터"""
    def decorator(f):
        @wraps(f)
        @jwt_required()
        def decorated_function(*args, **kwargs):
            user_id = get_jwt_identity()
            analysis_id = kwargs.get('analysis_id')

            if not has_permission(user_id, analysis_id, permission_type):
                return jsonify({'error': '권한이 없습니다'}), 403

            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route('/api/analysis/results/<analysis_id>/peg-comparison', methods=['GET'])
@require_permission('read')
def get_peg_comparison(analysis_id):
    """PEG 비교분석 결과 조회 (인증 필요)"""
    user_id = get_jwt_identity()

    # Rate limiting 확인
    if not check_rate_limit(user_id):
        return jsonify({'error': '요청 한도 초과'}), 429

    # 분석 결과 조회
    result = get_peg_comparison_result(analysis_id)
    return jsonify(result)
```

### 2. Rate Limiting

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["1000 per hour"]
)

@app.route('/api/analysis/results/<analysis_id>/peg-comparison', methods=['GET'])
@limiter.limit("10 per minute")
@jwt_required()
def get_peg_comparison(analysis_id):
    """Rate limiting이 적용된 PEG 비교분석 조회"""
    pass
```

### 3. 데이터 암호화

```python
from cryptography.fernet import Fernet
import base64

class DataEncryption:
    def __init__(self, key):
        self.cipher = Fernet(key)

    def encrypt_sensitive_data(self, data):
        """민감한 데이터 암호화"""
        if isinstance(data, dict):
            encrypted_data = {}
            for key, value in data.items():
                if key in ['cell_id', 'user_id']:  # 민감한 필드
                    encrypted_data[key] = self.cipher.encrypt(
                        str(value).encode()
                    ).decode()
                else:
                    encrypted_data[key] = value
            return encrypted_data
        return data

    def decrypt_sensitive_data(self, encrypted_data):
        """암호화된 데이터 복호화"""
        if isinstance(encrypted_data, dict):
            decrypted_data = {}
            for key, value in encrypted_data.items():
                if key in ['cell_id', 'user_id']:  # 민감한 필드
                    decrypted_data[key] = self.cipher.decrypt(
                        value.encode()
                    ).decode()
                else:
                    decrypted_data[key] = value
            return decrypted_data
        return encrypted_data
```

## 에러 처리 및 로깅

### 1. 에러 처리 전략

```python
import logging
from flask import jsonify
from werkzeug.exceptions import HTTPException

class PEGComparisonError(Exception):
    """PEG 비교분석 전용 에러 클래스"""
    def __init__(self, message, error_code, status_code=500):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        super().__init__(self.message)

class MCPConnectionError(PEGComparisonError):
    """MCP 서버 연결 에러"""
    def __init__(self, message="MCP 서버 연결 실패"):
        super().__init__(message, "MCP_CONNECTION_FAILED", 503)

class DataValidationError(PEGComparisonError):
    """데이터 검증 에러"""
    def __init__(self, message="데이터 검증 실패"):
        super().__init__(message, "DATA_VALIDATION_ERROR", 400)

@app.errorhandler(PEGComparisonError)
def handle_peg_comparison_error(error):
    """PEG 비교분석 에러 처리"""
    response = {
        'success': False,
        'error': {
            'code': error.error_code,
            'message': error.message,
            'timestamp': datetime.utcnow().isoformat()
        }
    }
    return jsonify(response), error.status_code
```

### 2. 로깅 시스템

```python
import logging
import json
from datetime import datetime

class PEGComparisonLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # 파일 핸들러 설정
        file_handler = logging.FileHandler('peg_comparison.log')
        file_handler.setLevel(logging.INFO)

        # 포맷터 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

    def log_request(self, user_id, analysis_id, request_data):
        """요청 로깅"""
        log_data = {
            'event': 'peg_comparison_request',
            'user_id': user_id,
            'analysis_id': analysis_id,
            'timestamp': datetime.utcnow().isoformat(),
            'request_size': len(json.dumps(request_data))
        }
        self.logger.info(json.dumps(log_data))

    def log_response(self, user_id, analysis_id, response_time, cached):
        """응답 로깅"""
        log_data = {
            'event': 'peg_comparison_response',
            'user_id': user_id,
            'analysis_id': analysis_id,
            'response_time': response_time,
            'cached': cached,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.logger.info(json.dumps(log_data))

    def log_error(self, user_id, analysis_id, error_type, error_message):
        """에러 로깅"""
        log_data = {
            'event': 'peg_comparison_error',
            'user_id': user_id,
            'analysis_id': analysis_id,
            'error_type': error_type,
            'error_message': error_message,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.logger.error(json.dumps(log_data))
```

## 성능 최적화

### 1. 비동기 처리

```python
import asyncio
import aiohttp
from celery import Celery

app = Celery('peg_comparison_backend')

@app.task(bind=True)
def process_peg_comparison_async(self, analysis_id, user_id, raw_data):
    """비동기 PEG 비교분석 처리"""
    try:
        # 진행률 업데이트
        self.update_state(state='PROGRESS', meta={'progress': 0})

        # MCP 서버 호출
        self.update_state(state='PROGRESS', meta={'progress': 50})
        result = call_mcp_server(analysis_id, raw_data)

        # 결과 저장
        self.update_state(state='PROGRESS', meta={'progress': 80})
        save_peg_comparison_result(analysis_id, result)

        # 캐싱
        self.update_state(state='PROGRESS', meta={'progress': 100})
        cache_peg_comparison_result(analysis_id, result)

        return {'status': 'SUCCESS', 'analysis_id': analysis_id}
    except Exception as exc:
        self.update_state(state='FAILURE', meta={'error': str(exc)})
        raise
```

### 2. 연결 풀링

```python
import aiohttp
import asyncio

class MCPClient:
    def __init__(self, base_url, max_connections=100):
        self.base_url = base_url
        self.connector = aiohttp.TCPConnector(
            limit=max_connections,
            limit_per_host=30,
            keepalive_timeout=30
        )
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(connector=self.connector)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def call_peg_comparison(self, analysis_id, raw_data):
        """MCP 서버 호출"""
        url = f"{self.base_url}/mcp/peg-comparison/analysis"
        payload = {
            'analysis_id': analysis_id,
            'raw_data': raw_data
        }

        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise MCPConnectionError(f"MCP 서버 응답 오류: {response.status}")
```

## 테스트 전략

### 1. 단위 테스트

```python
import unittest
from unittest.mock import Mock, patch
from your_app import PEGComparisonService

class TestPEGComparisonService(unittest.TestCase):
    def setUp(self):
        self.service = PEGComparisonService()
        self.mock_mcp_client = Mock()
        self.mock_cache = Mock()

    def test_get_peg_comparison_success(self):
        """성공적인 PEG 비교분석 조회 테스트"""
        # Given
        analysis_id = "test_123"
        expected_result = {"success": True, "data": {}}

        self.mock_cache.get_cached_result.return_value = None
        self.mock_mcp_client.call_peg_comparison.return_value = expected_result

        # When
        result = self.service.get_peg_comparison(analysis_id)

        # Then
        self.assertEqual(result, expected_result)
        self.mock_mcp_client.call_peg_comparison.assert_called_once()

    def test_get_peg_comparison_cached(self):
        """캐시된 결과 조회 테스트"""
        # Given
        analysis_id = "test_123"
        cached_result = {"success": True, "cached": True}

        self.mock_cache.get_cached_result.return_value = cached_result

        # When
        result = self.service.get_peg_comparison(analysis_id)

        # Then
        self.assertEqual(result, cached_result)
        self.mock_mcp_client.call_peg_comparison.assert_not_called()
```

### 2. 통합 테스트

```python
import pytest
from your_app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_peg_comparison_api_integration(client):
    """PEG 비교분석 API 통합 테스트"""
    # Given
    analysis_id = "test_123"
    headers = {'Authorization': 'Bearer test_token'}

    # When
    response = client.get(
        f'/api/analysis/results/{analysis_id}/peg-comparison',
        headers=headers
    )

    # Then
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True
    assert 'data' in data
```

## 배포 및 모니터링

### 1. Docker 설정

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

### 2. 모니터링 설정

```python
from prometheus_client import Counter, Histogram, generate_latest
import time

# 메트릭 정의
REQUEST_COUNT = Counter('peg_comparison_requests_total', 'Total PEG comparison requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('peg_comparison_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
MCP_CALL_DURATION = Histogram('mcp_call_duration_seconds', 'MCP call duration')

@app.before_request
def before_request():
    request.start_time = time.time()

@app.after_request
def after_request(response):
    # 요청 지속 시간 측정
    duration = time.time() - request.start_time
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.endpoint
    ).observe(duration)

    # 요청 수 카운트
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.endpoint,
        status=response.status_code
    ).inc()

    return response

@app.route('/metrics')
def metrics():
    return generate_latest()
```

## 마이그레이션 체크리스트

### 백엔드 API 구현

#### 핵심 기능

- [ ] PEG 비교분석 API 엔드포인트 구현
- [ ] MCP 서버 통신 로직 구현
- [ ] 사용자 인증 및 권한 검증 구현
- [ ] Rate limiting 구현
- [ ] 에러 처리 및 로깅 시스템 구현

#### 데이터 관리

- [ ] 데이터베이스 스키마 설계 및 구현
- [ ] Redis 캐싱 시스템 구현
- [ ] 데이터 암호화 구현
- [ ] 백업 및 복구 시스템 구현

#### 성능 최적화

- [ ] 비동기 처리 시스템 구현
- [ ] 연결 풀링 구현
- [ ] 메모리 사용량 최적화
- [ ] 응답 시간 최적화

#### 보안 및 모니터링

- [ ] JWT 인증 시스템 구현
- [ ] API 보안 설정
- [ ] 모니터링 및 알림 시스템 구현
- [ ] 로그 수집 및 분석 시스템 구현

#### 테스트 및 배포

- [ ] 단위 테스트 작성
- [ ] 통합 테스트 작성
- [ ] 성능 테스트 수행
- [ ] 보안 테스트 수행
- [ ] CI/CD 파이프라인 구축
- [ ] 배포 환경 구성

## 결론

백엔드 API는 PEG 비교분석 기능의 인프라 역할을 담당하며, 다음과 같은 핵심 책임을 가집니다:

1. **API 제공**: HTTP 엔드포인트를 통한 클라이언트 서비스
2. **인증 및 보안**: 사용자 인증, 권한 검증, 데이터 보호
3. **MCP 통신**: MCP 서버와의 안정적인 통신 관리
4. **캐싱**: 성능 향상을 위한 결과 캐싱
5. **에러 처리**: 견고한 에러 처리 및 복구 메커니즘
6. **모니터링**: 시스템 상태 및 성능 모니터링

MCP 서버의 PEG 비교분석 알고리즘 구현은 [PEG_COMPARISON_ANALYSIS.md](./PEG_COMPARISON_ANALYSIS.md)를 참조하세요.

---

_이 문서는 KPI Dashboard Backend API 프로젝트의 PEG 비교분석 기능 구현을 위한 완전한 기술 가이드입니다._
