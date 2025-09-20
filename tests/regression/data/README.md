# Choi 알고리즘 회귀 테스트 데이터

이 디렉토리는 Choi 알고리즘의 회귀 테스트를 위한 고정된 입력/출력 데이터셋을 포함합니다.
이 테스트들은 알고리즘 로직의 의도하지 않은 변경을 감지하는 '스냅샷' 역할을 합니다.

## 디렉토리 구조

### `normal_cases/`

정상적인 시나리오들로, 일반적인 운영 환경에서 발생하는 케이스들입니다.

- **`basic_similar_judgement.json`**: 기본적인 Similar 판정 시나리오
- **`high_traffic_ok_judgement.json`**: 고트래픽에서 OK 판정
- **`low_traffic_similar.json`**: 저트래픽에서 Similar 판정
- **`multiple_cells_normal.json`**: 다중 셀 정상 처리

### `edge_cases/`

경계 조건이나 특수한 상황들을 테스트하는 케이스들입니다.

- **`exactly_50_percent_filter.json`**: 정확히 50% 필터링 비율
- **`boundary_beta_values.json`**: β 임계값 경계선 케이스
- **`zero_samples_handling.json`**: 0 값 샘플 처리
- **`single_sample_series.json`**: 단일 샘플 시리즈

### `abnormal_triggers/`

각종 이상 탐지 및 특수 판정을 트리거하는 케이스들입니다.

- **`nd_cant_judge.json`**: ND로 인한 Can't Judge
- **`high_variation_nok.json`**: High Variation NOK 판정
- **`improve_degrade_detection.json`**: Improve/Degrade 탐지
- **`high_delta_anomaly.json`**: High Delta 이상 탐지
- **`zero_anomaly_detection.json`**: Zero 이상 탐지

### `boundary_conditions/`

수학적 경계 조건들을 테스트하는 케이스들입니다.

- **`min_max_threshold_exact.json`**: 정확한 Min/Max 임계값
- **`beta_threshold_boundaries.json`**: β 임계값들의 경계선
- **`extreme_cv_values.json`**: 극한 CV 값들
- **`distribution_separation.json`**: 분포 완전 분리 케이스

## 파일 명명 규칙

각 테스트 케이스는 다음 구조를 따릅니다:

- **입력 파일**: `{scenario_name}_input.json`
- **예상 출력**: `{scenario_name}_expected.json`
- **설명 파일**: `{scenario_name}_description.md` (선택사항)

## 데이터 구조

### 입력 파일 구조 (`*_input.json`)

```json
{
  "scenario_name": "basic_similar_judgement",
  "description": "기본적인 Similar 판정 시나리오",
  "test_purpose": "정상적인 KPI 분석 워크플로우 검증",
  "input_data": {
    "ems_ip": "192.168.1.100",
    "ne_list": ["NE001"]
  },
  "cell_ids": ["cell_001"],
  "time_range": {
    "pre_start": "2025-09-20T10:00:00",
    "pre_end": "2025-09-20T11:00:00",
    "post_start": "2025-09-20T14:00:00",
    "post_end": "2025-09-20T15:00:00"
  },
  "peg_data": {
    "cell_001": [...]
  },
  "expected_behavior": {
    "filtering_should_pass": true,
    "expected_anomalies": [],
    "expected_kpi_judgements": ["Similar"]
  }
}
```

### 예상 출력 구조 (`*_expected.json`)

Choi 알고리즘의 실제 응답 JSON과 동일한 구조를 가집니다.

## 테스트 케이스 설계 원칙

1. **완전성**: 모든 알고리즘 분기를 커버
2. **독립성**: 각 테스트는 다른 테스트에 의존하지 않음
3. **재현성**: 동일한 입력에 대해 항상 동일한 출력
4. **명확성**: 각 테스트의 목적과 예상 결과가 명확
5. **현실성**: 실제 운영 환경에서 발생 가능한 데이터

## 회귀 테스트 실행

```bash
# 전체 회귀 테스트 실행
python -m pytest tests/regression/ -v

# 특정 카테고리만 실행
python -m pytest tests/regression/ -k "normal_cases" -v

# 성능 회귀 테스트 포함
python -m pytest tests/regression/ -m "regression" -v
```

## 새로운 테스트 케이스 추가

1. 적절한 카테고리 디렉토리에 `{scenario_name}_input.json` 생성
2. `calculate_regression_outputs.py` 스크립트로 예상 출력 생성
3. 테스트 실행하여 검증

## 주의사항

- **절대 수정 금지**: 기존 회귀 테스트 데이터는 수정하지 마세요
- **버전 관리**: 모든 회귀 테스트 데이터는 Git으로 관리됩니다
- **알고리즘 변경 시**: 의도된 변경인 경우에만 예상 출력을 업데이트하세요
- **성능 기준**: 각 테스트는 5초 이내에 완료되어야 합니다
