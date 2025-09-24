# KPI Query API 명세

프론트엔드 `Dashboard`/`Dashboard.optimized` 컴포넌트가 사용하는 KPI 조회 API에 대한 명세입니다.

## 엔드포인트

- POST `/api/kpi/query`
- POST `/api/kpi/statistics/batch` (여러 KPI 동시 조회)

## 요청 (Request)

```json
{
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "kpi_types": ["availability", "rrc", "erab"],
  "ne": "optional-ne-filter-or-csv",
  "cellid": "optional-cellid-filter-or-csv"
}
```

- `kpi_types`: 여러 KPI를 동시에 조회하기 위한 배열. 백엔드는 하위 호환을 위해 단수 `kpi_type`(문자열)도 지원합니다.
- `ne`, `cellid`: 단일 문자열, 콤마 구분 문자열, 배열 모두 허용됩니다.

## 응답 (Response)

```json
{
  "success": true,
  "data": {
    "availability": [
      {
        "timestamp": "2025-08-14 10:00:00",
        "entity_id": "LHK078ML1",
        "value": 99.2,
        "kpi_type": "availability",
        "peg_name": "availability_rate",
        "ne": "ne_LHK078",
        "cell_id": "cell_1",
        "date": "2025-08-14",
        "hour": 10
      }
    ],
    "rrc": [{ "timestamp": "...", "entity_id": "...", "value": 98.7 }]
  },
  "metadata": {
    "total_records": 1234,
    "kpi_types": ["availability", "rrc"],
    "date_range": "2025-08-07 ~ 2025-08-14",
    "entity_count": 2,
    "ne_filters": ["..."],
    "cellid_filters": ["..."],
    "generated_at": "2025-08-14 11:00:00",
    "data_source": "PostgreSQL"
  }
}
```

- `data`는 "KPI 타입 → 행 배열"의 매핑 객체입니다. 각 배열 원소는 최소한 `timestamp`, `entity_id`, `value` 필드를 포함합니다.

## 프론트엔드 사용 시 주의

- 프론트는 단일 요청으로 여러 KPI를 받아 `data` 객체를 그대로 상태에 저장합니다.
- 데이터 전처리(차트 포맷)는 방어적으로 동작하도록 구현되어 있습니다.
  - 객체 혹은 배열 입력 모두 허용: `const rows = Array.isArray(data) ? data : Object.values(data || {}).flat()`
  - 시간 단위 그룹화 후 `[{ time, <entityId1>: value, <entityId2>: value, ... }]` 형태로 변환

### 배치 API 빠른 예시

```json
{
  "start_date": "2025-08-06",
  "end_date": "2025-08-07",
  "kpi_types": ["availability", "rrc", "erab"],
  "entity_ids": "LHK078ML1,LHK078MR1",
  "interval_minutes": 60
}
```

## 예시 코드 (프론트 요청)

```ts
await apiClient.post("/api/kpi/query", {
  start_date: "2025-08-07",
  end_date: "2025-08-14",
  kpi_types: ["availability", "rrc"],
  ne: "nvgnb#10000",
  cellid: "2010",
});
```

## 변경 이력

- 2025-08-14: 단일 요청(`kpi_types`) 사용 및 응답 객체 스펙 문서화. 프론트 방어 로직 추가.
