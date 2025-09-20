# 알고리즘 구현 및 분석 계획
_Exported on 9/20/2025 at 17:23:53 GMT+9 from Cursor (1.6.35)_

---

**User**

3~6장에 대해서 분석해줘. 이 알고리즘은 peg의 '판정' 알고리즘이며, 우리는 choi 알고리즘이라고 부르고있어.  3GPP KPI pegs '판정' 알고리즘 관점으로 상세하게 분석해줘. 우리는 이 상태에서 더 향상시키지 않고, 이 문서의 알고리즘을 그대로 구현하고자 하는 것이 목적이야. 이 알고리즘을 파이선 코드로 구현할 것이고 우리의 MCP / backend / frontend에 결합하는 것이 목적이야. 어느 부분에서 구현하는것이  합리적이고 나중에 '판정'알고리즘을 교체 하거나 'enhancement를 위한 리펙토링'을 대비하는 것까지 고려해야해.

깊이 생각해서 이 알고리즘에 대해 분석 해줘. 단계적으로 어느 위치에 구현해야 할지 탐구해서 종합 리포트로 구성해서 알려줘. 보고서 결정하도록 할게



---

**Cursor**

지금 `TES.web_Choi.md` 3~6장을 확인하고, 레포의 `peg` 관련 서비스/테스트를 함께 열어 실제 적용 지점을 파악하겠습니다. 곧바로 관련 파일들을 동시에 읽어 전체 구조와 통합 포인트를 잡겠습니다.

방금 `TES.web_Choi.md` 3~6장을 정독하고, 레포의 `PEGCalculator`/`PEGProcessingService`/도메인 모델을 훑어 통합 지점을 확인했습니다. 이제 ‘Choi 판정 알고리즘’을 3GPP KPI pegs 관점에서 그대로 구현·결합하는 종합 보고서를 드립니다.

### 목적과 범위
- 목적: 문서의 3~6장을 “있는 그대로(as-is)” 파이썬으로 구현하여 MCP / backend / frontend에 결합.
- 범위:
  - 4장: Abnormal Stats Detecting (PEG 전수)
  - 5장: Stats Analyzing (Main KPI + Sub KPI 묶음 판정)
  - 6장: Filtering (유효 시간대 선정)
  - 3장은 결과 UI/요약 형태에 해당하며, 백엔드 출력 구조와 프론트 렌더 기준으로 반영

### 핵심 개념 요약 (알고리즘 규칙 그대로)
- **Filtering(6장)**:
  - 샘플 전처리(이웃 시각 0 처리, DL/UL 합 0 제외) → Peg별 중앙값 산출 → 샘플/중앙값으로 정규화 → 각 Peg에서 Min_Th ≤ 3 ≤ Max_Th 인 시각만 채택 → 모든 Peg에 공통으로 남는 시각만 최종 유효 시각으로 선택.
  - 필터링 결과가 입력 대비 50% 이하이면 “TES can't filter...” 메시지를 남기고 전체 시각 채택.
  - 변수: Min_Threshold=0.87, Max_Threshold=1.13, Filter_Ratio=50%.
- **Abnormal Detect(4장)**: 모든 통계(peg) 대상. 셀 단위 판정 후, ‘해당 셀 수 ≥ α0(=2)’이면 결과 표시. 전체 셀이 α0 미만이면 모든 셀이 동일 판정이어야 표시.
  - Range(Analyze/Compare 공통): DIMS의 [min,max] 밖
  - New(Compare): 신규 통계
  - ND(Compare): pre/post 한쪽만 ND
  - Zero(Compare): pre/post 한쪽만 0
  - High Delta(Compare): δ > β3, δ = (post-pre)/pre*100
  - 변수: α0=2, α1=10(표 내 정의), β3=500
- **Stats Analyze(5장, Main/Sub KPI 묶음)**:
  - Compare 모드에서 pre/post 분포를 비교해 결과(OK/POK/NOK/Can’t judge/Low/Medium/High Delta/Improve/Degrade/High Variation) 산출
  - Can’t judge: pre/post 둘 중 ND 포함
  - High Variation: CV(pre) 또는 CV(post) > β4(=10) 등, ND/0 특례 시 δ=±100% 처리 포함
  - Degrade/Improve: KPI ‘positive/negative’ 여부에 따라 min/max 범위 비교
  - Similar/Low/Medium/High Delta: β0(=1000) 경계로 β1/β2 적용 방식 분기 + ||δ||<β5(=3) 시 Similar
  - 변수(5.3): β0=1000, β1=5, β2=10, β3=500, β4=10, β5=3
  - 결과 요약(5.4): Main과 Sub 결과 조합으로 Final(OK/POK/NOK/Can’t judge) 및 요약 문구 생성

### 현 레포 구조와 이상적 구현 위치
- 필터링/판정은 “집계와 파생계산”과 별도 층으로 유지하는 게 교체성에 유리.
- 권장 계층:
  - `services/peg_filtering_service.py` (신규) ← 6장 구현
  - `services/peg_judgement_service.py` (신규) ← 4장, 5장 구현
  - `services/peg_processing_service.py` ← 오케스트레이션만 담당, 단계 추가
  - `models/judgement.py` (신규) ← Enum/DTO(판정결과/요약/근거) 정의
  - `config/choi_algorithm.yml` or `.py` (신규) ← 상수/메타(β, α, Main↔Sub 구성, positive/negative 지표표)
- 통합 포인트(인용):
```98:106:analysis_llm/services/peg_processing_service.py
        # 처리 단계 정의
        self.processing_steps = [
            "data_retrieval",
            "data_validation",
            "aggregation",
            "derived_calculation",
            "result_formatting",
        ]
```
- 여기에 `"filtering"`, `"judgement"` 단계를 추가해 오케스트레이션 확장.

### 데이터 요건과 파이프라인
- 알고리즘은 “샘플 레벨 분포”가 필요(평균만으로 불가). 즉:
  - 각 Peg, 각 Cell, 각 기간(pre/post)의 샘플 배열(5/15분 단위)이 필요
  - 현 `PEGProcessingService`는 평균 중심. 아래 구조로 확장 필요:
    - DB 조회: pre/post 기간의 시계열(raw) + 셀 기준 그룹핑
    - FilteringService: 6장 알고리즘으로 유효 시각 인덱스 집합 산출(셀/참조 Peg 교집합)
    - 통계치: 유효 시각만 대상으로 평균/최솟값/최댓값/표준편차/CV 계산
    - JudgementService: 4장, 5장 판정 실행 → 결과 DTO
- 기존 집계/파생은 그대로 유지(`PEGCalculator`). 판정은 별도.

### 설계 제안 (교체 가능성 고려: Strategy 패턴)
- **인터페이스**
  - `JudgementStrategy`:
    - `judge_abnormal(pegs_timeseries_by_cell, dims_ranges) -> AbnormalSummary`
    - `judge_kpis(main_kpis_by_topic, sub_kpis_by_topic, positivity_map) -> KPIJudgementSummary`
  - `FilteringStrategy`:
    - `select_valid_times(reference_pegs_timeseries_by_cell) -> Dict[cell, Set[timeslot]]`
- **구현체**
  - `ChoiFiltering(FilteringStrategy)` ← 6장
  - `ChoiJudgement(JudgementStrategy)` ← 4·5장
- **오케스트레이터**
  - `PEGProcessingService.process_peg_data(...)`에 필터링/판정 단계 삽입
  - 교체 필요 시 전략 인스턴스만 DI로 바꿔 끼움

### 도메인 모델(요약)
- **Enums**
  - `JudgementType`: OK, POK, NOK, CANT_JUDGE
  - `CompareDetail`: SIMILAR, LOW_DELTA, MEDIUM_DELTA, HIGH_DELTA, IMPROVE, DEGRADE, HIGH_VARIATION, NEW, ND, ZERO, RANGE
- **DTO**
  - `PegSampleSeries`: peg_name, cell_id, pre_samples[], post_samples[]
  - `PegPeriodStats`: mean, min, max, std, cv, nd_ratio, zero_ratio
  - `PegCompareMetrics`: delta_pct(δ), flags(ND/Zero/…) 
  - `PegCompareDecision`: detail, reason, thresholds_used
  - `MainKPIJudgement`: main_result(JudgementType), sub_results(list), final_result(JudgementType), summary_text
  - `AbnormalDetectSummary`: 항목별/셀수 카운트와 표시 여부(α0 규칙 반영)

### 알고리즘 구현 핵심 로직 설계
- **Filtering(6장)**:
  - 입력: 참조 Peg 집합(표 6.2), 각 Peg의 pre/post 샘플(시계열), 윈도우 단위 동일
  - 전처리: 이웃(n-1, n+1) 0인 경우 n 제외; DL/UL 합계 0 제외
  - Peg별 중앙값 M 산출 → 각 샘플 x에 대해 x/M → 0.87 ≤ x/M ≤ 1.13 인 타임슬롯 유지
  - Peg 교집합 타임슬롯만 채택; 비율이 50% 이하이면 전체 사용 + 경고 메시지
- **Abnormal(4장)**:
  - Compare 전제
  - Peg별/셀별로 δ, ND/Zero/New/Range 판정
  - 셀 카운트가 α0 이상이면 결과 표출(전체 셀<α0이면 모든 셀 동일 판정 조건)
- **Analyze(5장)**:
  - 분포 기반: pre/post 각각 min/max/mean/std/cv 산출
  - Can’t judge/High Variation 처리(β4=10), ND/0 특례 δ=±100%
  - Improve/Degrade: positive/negative 매핑 필요(설정파일)
  - Similar/Low/Medium/High Delta: β0 경계로 β1/β2 적용, β3 상한, β5 절대값 소규모 구간
  - 결과 요약(5.4) 조립: Main 결과에 Sub 결과를 결합해 최종/요약문 생성

### 구성/상수(코드 외부화)
- `config/choi_algorithm.yml` 예:
  - thresholds: α0, α1, β0~β5, min_th, max_th, filter_ratio
  - positivity_map: KPI 이름 → positive/negative
  - main_kpi_topics: topic → {main: peg, subs: [pegs...]}
  - dims_ranges: peg → {min,max} (Range 판정용; 없으면 Range 스킵)
- 운영 중 변경 가능, 리로드 지원(핫 리로드/버전 태깅 권장)

### 통합 계획 (Backend/MCP/Frontend)
- Backend
  - `PEGProcessingService`에 2단계 추가:
    - `"filtering"`: `FilteringStrategy` 호출 → 유효 타임슬롯 인덱스 전달
    - `"judgement"`: `JudgementStrategy` 호출 → Abnormal(4장), KPI(5장) 결과 생성
  - 결과 스키마: 3장의 UI와 5.4 요약을 그대로 만들 수 있게 JSON 구조 정규화
  - 로깅/예외: 함수 단위 INFO/DEBUG, 단계/입력 요약, 임계치/사유(reason) 반드시 포함
- MCP
  - 태스크로 분리: “Choi Filtering”, “Choi Judgement”, “KPI Topic 정의”, “positivity 매핑” 등
  - 교체 대비: 별도 태그에서 대안 알고리즘 개발/테스트 가능
- Frontend
  - 3장의 뷰(OK/POK/NOK/Can’t judge, 툴팁으로 기준 노출, Sub KPI 토글, POST AVG/Full Screen 등)를 재현 가능한 필드 제공:
    - 각 셀·Peg의 detail(δ, CV, ND/Zero/Range/New), 기준 문구, 임계치
    - Main KPI 행: hover 시 판정 기준/수치 표시
    - Sub KPI 드릴다운: 각 Sub의 판정/근거
    - 필터링 메시지(50% 이하 시) 표기

### 테스트 전략
- 단위 테스트
  - Filtering: 경계값(0.87/1.13), 중앙값 비대칭, 50% 임계 메시지
  - Abnormal(4장): ND/Zero/New/Range/High Delta δ=500+ 케이스
  - Analyze(5장): 각 세부 규칙(positive/negative 전환, β0 스위치, β5 절대값)
- 통합 테스트
  - 실제 유사 시계열(사전 생성)로 topic별 Main/Sub 판정 일치 검증
  - α0 조건에 따른 결과 노출 여부
- 회귀 고정
  - 문서 예시/사내 표준 샘플을 픽스쳐로 고정해 스냅샷 검증

### 로그/예외(요구사항 반영)
- 모든 공개 함수는 시작/입력 요약/중간 의사결정/최종 결과 로그 남김
- 실패 시 단계·임계치·입력분포 요약을 예외 detail로 첨부
- 샘플 수 부족/ND 과다/0 과다 등의 진단 힌트 포함

### 마이그레이션 단계(실제 레포 반영 시)
- 1) 모델/설정 추가: `models/judgement.py`, `config/choi_algorithm.yml`
- 2) 서비스 추가: `services/peg_filtering_service.py`, `services/peg_judgement_service.py`
- 3) 오케스트레이션: `PEGProcessingService`에 `"filtering"`, `"judgement"` 단계 삽입
- 4) 리포지토리 확장: 시계열 조회 API 추가(셀/peg/기간별 샘플)
- 5) API 응답 스키마 확장: 3장 UI가 그대로 그려지도록 JSON 형태 정규화
- 6) 테스트: 단위/통합 케이스 작성, 베이스라인 스냅샷 고정

### 코드 결합 포인트(인용)
- 결과 조립 이전에 판정 결과를 합치는 위치:
```361:366:analysis_llm/services/peg_processing_service.py
            # 4단계: PEGCalculator 처리
            logger.info("4단계: PEGCalculator 처리")
            processed_df = self._process_with_calculator(n1_df, n_df, peg_config or {})
            
            logger.info("PEG 데이터 처리 워크플로우 완료: %d행", len(processed_df))
            return processed_df
```
- 여기서 `processed_df` 이전/이후에:
  - “시계열 로드 → Filtering → 통계치 산출 → Judgement → `processed_df`와 병합” 순으로 편입
- 도메인 유틸(시계열→통계 변환 등)은 `utils/`에 배치해 재사용화 권장

### 리스크/주의
- DIMS Range 문서값 소스 필요(없으면 Range 판정은 옵셔널 처리)
- 샘플 간격/동기화(5/15분) 강제 필요
- positive/negative 매핑과 Main↔Sub 토픽 구성 정확성(표 5.1.x 반영)
- pre/post 샘플 수 비대칭/ND 비율 높은 환경에서 예외경로 충분히 로그

- - -

- 적용 제안:
  - **서비스 분리(Filtering/Judgement) + 전략 인터페이스 + 설정 외부화**로 “그대로 구현”하면서도 교체/개선에 대비.
  - **오케스트레이션은 기존 `PEGProcessingService`**에 단계로만 추가해 침습 최소화.
  - **프론트엔드**는 3장의 뷰 로직을 그대로 구현할 수 있도록 백엔드가 풍부한 근거 필드를 제공.

요약
- 6장 필터링(중앙값 정규화/임계/교집합/50% 규칙), 4장 이상치(α0/δ/ND/Zero/New/Range), 5장 비교 판정(β0~β5, CV/Improve/Degrade/Delta 계층, 5.4 요약)을 원형대로 구현.
- 새로운 `FilteringService`/`JudgementService`와 `config/choi_algorithm.yml`로 모듈화해 교체·개선 용이.
- `PEGProcessingService`에 “filtering”/“judgement” 단계만 추가하는 형태로 안전하게 결합.
- 시계열 데이터 조회/통계 산출이 필수 전제. 단위/통합 테스트로 규칙별 경계 검증.

```python
        # 처리 단계 정의
        self.processing_steps = [
            "data_retrieval",
            "data_validation",
            "aggregation",
            "derived_calculation",
            "result_formatting",
        ]
```

```python
            # 4단계: PEGCalculator 처리
            logger.info("4단계: PEGCalculator 처리")
            processed_df = self._process_with_calculator(n1_df, n_df, peg_config or {})
            
            logger.info("PEG 데이터 처리 워크플로우 완료: %d행", len(processed_df))
            return processed_df
```

---

**User**

4,5,6 장의 판정기준에서 논리적으로 어긋난 항목은 없어? 내가 손으로 옮기면서 오류의 가능성이 있을 수 있어. 논리적으로 전체 흐름에 어긋나지 않고 kpi 판정을 내릴 수 있는지 검토해줘

---

**Cursor**

방금 4·5·6장을 기준으로 ‘판정’ 규칙을 전체 흐름 관점에서 논리 검토했습니다. 문서 그대로 구현 가능한지, 상충/모호 지점을 중심으로 짚고 수정·보완 권고안을 함께 제시합니다.

### 핵심 근거 인용
```186:205:c:\Users\aruca\Downloads\TES.web_Choi.md
4. Abnormal Stats Detecting Algorithm
...
4.1. Decision Results
| Analyze | Compare |  |  |
| O | O | Range | Statistics are out of the [min, max] range of the DIMS document. |
| X | O | New | If the statistics are new based on the DIMS document. ... |
| X | O | ND | If only one of ‘pre’ and ‘post’is ND. |
| X | O | Zero | If only one of ‘pre’ and ‘post’  is zero. |
| X | O | High Delta | ’δ’ >’β3’ |
δ : (‘post’-’pre’)/pre*100 (%)
```

```652:679:c:\Users\aruca\Downloads\TES.web_Choi.md
5.2. Decision Results
| Result | Criteria | PLM Defect |
| Can't judge | If either or both ‘pre’ and ‘post’ are ND. | X |
| High Variation | 1.  CV(pre) > β4 or CV(post) > β4 2. ... ND/0 특례 시 δ=±100% ... | X |
| Degrade | ... | NOK |
| Improve | ... | NOK |
| Similar | 1. If ‘pre’ < β0 or ‘post’< β0 , then δ <= β2 2. If ‘pre’ >= β0 or ‘post’>=β0 , then δ <= β1 3. ||δ||<β5 | OK |
| Low Delta | (β2/β1 구간) | NOK |
| Medium Delta | (2*β2/2*β1 구간) | NOK |
| High Delta | δ > β3 | NOK |
...
β0=1000, β1=5, β2=10, β3=500, β4=10, β5=3
```

```698:706:c:\Users\aruca\Downloads\TES.web_Choi.md
6.1. Algorithm Flow
1) ... 2) median ... 3) Normalize (divide by median) 4) Select only the time with sample values of Min_Th <= "3" <= Max_Th.
5) ... if filtered ratio <= 50%, show message and select entire time interval.
```

```730:735:c:\Users\aruca\Downloads\TES.web_Choi.md
6.3. Variables
Min_Threshold=0.87, Max_Threshold=1.13, Filter_Ratio=50%
```

### 논리 일관성 점검 결과와 보완 권고

- 문제 없음/의도대로 보임
  - **4장 전체 구조**: 셀 다수 일치(α0=2) 시 결과 표출, 전체 셀 수가 α0 미만이면 “전 셀 동일” 조건으로 표출. Compare 전용 항목(New/ND/Zero/High Delta) 정의 일관.
  - **5.2 Delta 계층(β1/β2/β3) 구조**: 낮은 트래픽(β0)인 경우 완화 임계(β2), 높은 트래픽인 경우 엄격 임계(β1) → Medium/High로 확장하는 계층 구조 자체는 타당.
  - **6장 필터링의 중앙값 정규화**: Peg별 중앙값으로 정규화 후 임계 범위(0.87~1.13) 교집합 채택, 50% 이하 시 전체 구간 사용 메시지 표출 → 일반적인 안정구간 추출 방식과 합리적.

- 모순/모호/오타로 보이는 지점

  - 구문/오타 의심
    - 6.1 4) “Min_Th ≤ "3" ≤ Max_Th”는 정규화 후 기준값이 1이어야 자연스럽습니다. “3”은 **오타로 보이며 “1”**이 맞습니다. 수정 권고: Min_Th ≤ normalized ≤ Max_Th.

  - 정의·변수 사용의 모호성
    - 4.2에 **α1=10이 표에 정의만 있고 사용처가 없음**. 누락이거나 불필요 변수로 보입니다. 사용 용도 명시·삭제 중 택일 권고.
    - 5.2에서 **β0=1000과 ‘pre/post’ 비교**는 KPI 값(퍼센트/율/비율 등)에 일괄 적용 시 비논리적일 수 있습니다. 업계 관례상 β0는 “샘플 수/트래픽량(분모)”의 하한으로 쓰입니다. 현재 표기는 “KPI 값”과 비교하는 문맥이라 **실제 의도는 ‘분모(시도/패킷/바이트/PRB 등) < β0’**일 가능성이 높습니다. 정확히 “어떤 기준량을 β0와 비교하는지”를 **KPI별로 명시**해야 혼선이 없습니다.
    - 5.2 Similar의 1·2번 조건이 모두 “or”을 사용해, 한쪽은 <β0, 다른쪽은 ≥β0인 경우 두 조건이 동시에 참이 됩니다. **실무 의도는 ‘둘 중 하나라도 저트래픽이면 완화 임계(β2)를 적용’**일 가능성이 높습니다. 즉 “or” 자체는 맞지만, 충돌 시 **우선순위(β2 우선)**를 명문화 권고.
    - 5.2 Low/Medium Delta 구간은 **δ 기호에 절대값 표기가 없음**. Similar에는 ||δ||가 있고 Delta 계층에는 없음. 보통 Delta 계층은 **변화량 크기(|δ|)** 기준이므로 “|δ|”로 명시 권고(해석 일관성).
    - 5.2 High Delta는 β0분기와 무관하게 “δ>β3”로 **두 줄이 동일 조건**입니다. 중복 표기이며 하나로 합쳐도 무방(오해 방지 차원에서 정리 권고).

  - 규칙 간 충돌 가능성(우선순위 필요)
    - **ND vs High Variation vs Delta**:
      - 5.2 “Can't judge: ND 존재”와 High Variation의 2·3항(“ND인 경우 δ=±100%로 간주”)이 **동시에 성립** 가능. 이 경우 **Can't judge 우선**(ND가 있으면 다른 판정 배제)을 명문화 권고.
    - **Improve/Degrade vs Delta 계층**:
      - min/max 기반 분포 분리(Degrade/Improve)와 평균 기반 δ 크기(Deltas)가 **동시에 성립** 가능. 보통 **분포 분리(Degrade/Improve)를 Delta 계층보다 우선 적용**합니다. 우선순위 표기 권고.
    - **4장 ND/Zero(New)와 5장 Can't judge/High Variation**:
      - 4장은 “이상 통계 탐지” 결과가 **복수 구분 동시 표출 허용**(문구로 명시). 5장은 KPI 판정(최종결과) 단일화가 필요. **출력 채널 구분**(4장은 목록형 다태깅, 5장은 우선순위 기반 단일 최종 판정)을 명시 권고.

  - 외부 의존/실행 정의
    - **Range**: DIMS [min,max] 참조 필요. 누락 시 Range 판정 불가 → “데이터 없음 시 Range 스킵” 또는 “N/A 처리”를 명시 권고.
    - **6장 5) 50% 이하 시 전구간 사용**: “전구간”이 1)에서 제외한 0/무효 샘플까지 되살리는지 불명확. 일반적으로는 **무효값 제외는 유지하고 정규화 임계(0.87~1.13)만 해제**가 안전합니다. 의도 명시 권고.
    - **정규화 분모(중앙값) = 0**인 경우 처리 규칙 누락. 0 분모 방지/샘플 제외/대체값 지정 중 하나를 명시 권고.

- 표시/요약의 일관성
  - 5.4 Final Summary 조합은 합리적. 다만 **표시 순서/우선순위**를 문장으로 명시 권장(예: Can't judge > High Variation > Improve/Degrade > Delta 계층 > Similar).
  - 3.3(OK/POK/NOK/Can't judge) 툴팁·세부 표시 기준은 5.2 임계/사유를 그대로 노출하면 일관성 유지.

### 종합 판단
- **큰 틀의 논리는 타당**하며, 문서 기준으로 KPI 판정 구현이 가능합니다.
- 다만 아래 5가지는 구현 전 정정/명문화가 강력 권고됩니다.
  1) 6.1 정규화 임계 “3” → “1”로 수정  
  2) 5.2 β0 비교 대상의 명확화(“KPI 값”이 아닌 “분모/트래픽량/샘플수” 등으로 KPI별 구체화)  
  3) 5.2 Delta 계층에 |δ| 사용을 명시(유사/Delta 전부 절대값 기준 통일)  
  4) ND/Zero/High Variation/Delta/Improve/Degrade **우선순위 규칙 명시**  
  5) 6장 50% 규칙 시 “무효값 제외는 유지, 정규화 임계만 해제”로 동작 정의

원문을 그대로 최대한 존중하면서 위 항목만 보완하면, 백엔드/프론트 일관성과 운영 안정성을 함께 확보할 수 있습니다.

- 지금부터는 위 권고대로 문서의 표현만 보완한 “명확판 버전”을 간단 요약해드릴 수 있습니다. 필요하시면 정리본(표/문장) 만들어 드릴게요.

요약
- 치명적 모순은 없고 구현 가능.  
- 오타(6.1의 "3"→"1"), β0의 비교 대상(값 vs 분모) 모호, Delta의 |δ| 표기 불일치, 중복/우선순위 부재 등 실무 구현 시 혼선 유발 지점 존재.  
- ND가 있으면 Can't judge 우선, Improve/Degrade가 Delta보다 우선 등 우선순위 명시 권고.  
- 필터링 50% 규칙과 분모=0 처리 규칙을 명문화 권고.

```plaintext
4. Abnormal Stats Detecting Algorithm
...
4.1. Decision Results
| Analyze | Compare |  |  |
| O | O | Range | Statistics are out of the [min, max] range of the DIMS document. |
| X | O | New | If the statistics are new based on the DIMS document. ... |
| X | O | ND | If only one of ‘pre’ and ‘post’is ND. |
| X | O | Zero | If only one of ‘pre’ and ‘post’  is zero. |
| X | O | High Delta | ’δ’ >’β3’ |
δ : (‘post’-’pre’)/pre*100 (%)
```

```plaintext
5.2. Decision Results
| Result | Criteria | PLM Defect |
| Can't judge | If either or both ‘pre’ and ‘post’ are ND. | X |
| High Variation | 1.  CV(pre) > β4 or CV(post) > β4 2. ... ND/0 특례 시 δ=±100% ... | X |
| Degrade | ... | NOK |
| Improve | ... | NOK |
| Similar | 1. If ‘pre’ < β0 or ‘post’< β0 , then δ <= β2 2. If ‘pre’ >= β0 or ‘post’>=β0 , then δ <= β1 3. ||δ||<β5 | OK |
| Low Delta | (β2/β1 구간) | NOK |
| Medium Delta | (2*β2/2*β1 구간) | NOK |
| High Delta | δ > β3 | NOK |
...
β0=1000, β1=5, β2=10, β3=500, β4=10, β5=3
```

```plaintext
6.1. Algorithm Flow
1) ... 2) median ... 3) Normalize (divide by median) 4) Select only the time with sample values of Min_Th <= "3" <= Max_Th.
5) ... if filtered ratio <= 50%, show message and select entire time interval.
```

```plaintext
6.3. Variables
Min_Threshold=0.87, Max_Threshold=1.13, Filter_Ratio=50%
```

---

**User**

MCP, 백엔드, 프론트엔드에 나눠서 구현하는게 가장 적절한가?

---

**Cursor**

**결론: 백엔드 중심 구현이 가장 적절합니다.**

현재 프로젝트 구조와 Choi 알고리즘의 특성을 종합 분석한 결과, **MCP/백엔드/프론트엔드 3분할보다는 백엔드 중심 구현**을 강력 권장합니다.

### 권장 아키텍처: 백엔드 중심 + MCP 보조

**🎯 핵심 구현 위치**
- **백엔드 (analysis_llm)**: 6장, 4장, 5장 알고리즘 **전체 구현**
- **MCP**: 결과 **해석 및 설명** 지원  
- **프론트엔드**: **순수 UI 렌더링**

### 각 계층별 역할 분담

**📊 백엔드 (analysis_llm) - 핵심 엔진**
```python
# 신규 추가 서비스들
services/
├── choi_filtering_service.py      # 6장: 필터링 알고리즘
├── choi_judgement_service.py      # 4장, 5장: 이상탐지 + KPI 판정
└── peg_processing_service.py      # 기존: 오케스트레이션 확장

# 설정 외부화
config/
└── choi_algorithm.yml             # α, β 상수 + KPI 정의
```

**🤖 MCP - 해석 지원**
- 최종 판정 결과에 대한 **자연어 설명** 생성
- "왜 이 KPI가 NOK인지" **근거 설명**
- 사용자 질의응답: "이 변화가 정상인가요?"
- **트렌드 분석** 보조

**🎨 프론트엔드 - UI 렌더링**
- 3장 형태로 **OK/POK/NOK/Can't judge 표시**
- **툴팁**으로 판정 근거 표시 (β값, 임계치 등)
- **Sub KPI 드릴다운**, 필터링 UI
- **Full Screen** 모드, 통계 그래프

### 왜 이 구조가 최적인가?

**✅ 장점들**
1. **성능 최적화**: 계산 집약적 작업(필터링, 통계)을 백엔드에서 일괄 처리
2. **일관성 보장**: 단일 시스템 내 상태 관리, 버전 동기화 용이
3. **디버깅 편의**: 에러 추적이 주로 백엔드 내에서 완결
4. **교체 용이성**: Strategy 패턴으로 알고리즘 교체 시 백엔드 내 서비스만 변경
5. **AI 활용**: MCP가 규칙 적용이 아닌 **결과 해석**에 집중 → 더 큰 가치 창출

**❌ 3분할의 문제점들**
- **복잡성 폭증**: 계층 간 데이터 전달, 상태 동기화
- **성능 오버헤드**: 네트워크 통신, 직렬화 비용
- **일관성 리스크**: 각 계층 버전 불일치 시 판정 오류
- **디버깅 어려움**: 에러가 여러 시스템에 걸쳐 발생

### 구체적 구현 플랜

**1단계: 백엔드 확장**
```python
# PEGProcessingService에 단계 추가
self.processing_steps = [
    "data_retrieval",
    "data_validation", 
    "filtering",           # 신규: 6장 구현
    "aggregation",
    "derived_calculation",
    "judgement",           # 신규: 4장, 5장 구현
    "result_formatting",
]
```

**2단계: MCP 연동**
- 백엔드 판정 결과를 받아 자연어 설명 생성
- 사용자 질의 시 판정 근거 설명

**3단계: 프론트엔드 렌더링**
- 백엔드에서 받은 풍부한 판정 데이터를 3장 UI로 표현
- 툴팁, 드릴다운 등 인터랙션 구현

### 예상 데이터 플로우

```
원시 데이터 → [백엔드] → 판정 결과 + 근거 → [MCP] → 자연어 설명 → [프론트] → UI 렌더링
            필터링→집계→판정              해석 생성              표시+상호작용
```

**결론**: Choi 알고리즘은 **명확한 수학적 규칙**이므로 백엔드가 담당하고, AI는 **결과 해석**에 집중하는 것이 각 기술의 장점을 최대한 활용하는 아키텍처입니다. 성능, 안정성, 유지보수성을 모두 확보할 수 있습니다.

```python
# 신규 추가 서비스들
services/
├── choi_filtering_service.py      # 6장: 필터링 알고리즘
├── choi_judgement_service.py      # 4장, 5장: 이상탐지 + KPI 판정
└── peg_processing_service.py      # 기존: 오케스트레이션 확장

# 설정 외부화
config/
└── choi_algorithm.yml             # α, β 상수 + KPI 정의
```

```python
# PEGProcessingService에 단계 추가
self.processing_steps = [
    "data_retrieval",
    "data_validation", 
    "filtering",           # 신규: 6장 구현
    "aggregation",
    "derived_calculation",
    "judgement",           # 신규: 4장, 5장 구현
    "result_formatting",
]
```

```plaintext
원시 데이터 → [백엔드] → 판정 결과 + 근거 → [MCP] → 자연어 설명 → [프론트] → UI 렌더링
            필터링→집계→판정              해석 생성              표시+상호작용
```

---

**User**

난 지금 이 채팅 세션을 백엔드 폴더에서 유지하고 싶어. 옮기는 방법있어? 

---

**Cursor**

현재 채팅 세션을 `D:\Coding\kpi_dashboard\backend` 폴더로 옮기는 방법을 알려드리겠습니다.

PowerShell에서는 다른 명령어를 사용해야 합니다:

---

**User**

PS D:\Coding\kpi_dashboard\backend> Get-Location

Path
----
D:\Coding\kpi_dashboard\backend


PS D:\Coding\kpi_dashboard\backend> 

---

**Cursor**

완벽합니다! 🎉 

현재 작업 디렉토리가 `D:\Coding\kpi_dashboard\backend`로 성공적으로 변경되었습니다. 이제 이 채팅 세션에서 실행하는 모든 터미널 명령어들이 backend 폴더를 기준으로 동작할 것입니다.

현재 `D:\Coding\kpi_dashboard\backend` 폴더에 있지만, 실제 파일 목록을 보니 `3gpp_analysis_mcp` 프로젝트 구조가 보입니다. 

이제 이 백엔드 환경에서 **Choi 알고리즘 구현**을 진행할 수 있습니다! 앞서 분석한 대로:

- **`analysis_llm/services/`** 폴더에 `choi_filtering_service.py`, `choi_judgement_service.py` 추가
- **`analysis_llm/config/`** 폴더에 `choi_algorithm.yml` 설정 파일 추가  
- **`analysis_llm/models/`** 폴더에 `judgement.py` 도메인 모델 추가

필요하시면 지금부터 실제 구현을 시작해보겠습니다. 어떤 부분부터 시작하고 싶으신가요?

1. **설정 파일** (`choi_algorithm.yml`) 부터?
2. **도메인 모델** (`judgement.py`) 부터?
3. **필터링 서비스** (6장) 부터?
4. **판정 서비스** (4장, 5장) 부터?

