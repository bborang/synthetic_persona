# 합성 페르소나 대화 테스트 & H3 실험

`ko_KR.parquet`에 담긴 약 100만 명의 합성 페르소나로 OpenAI Chat Completions API와 대화합니다.
두 가지 방식으로 쓸 수 있습니다.

1. **`main.py`** — 터미널에서 페르소나 1명을 골라 자유롭게(또는 H3 주제로) 대화
2. **`scripts/`** — H3 실험(질문유형×정보량×세션유형×반복)을 배치로 돌리고 결과를 분석

## 0. 시작하기 전에 (필수 준비물)

이 저장소에는 아래 2가지가 **포함되어 있지 않습니다.** (`.gitignore`로 제외됨)
각자 아래 방법으로 직접 준비해야 프로그램이 동작합니다.

1. **`.env` 파일 (본인 OpenAI API 키)**
   `.env.example`을 복사해 `.env`를 만들고, 본인의 API 키를 입력하세요.

2. **`ko_KR.parquet` (원본 페르소나 데이터, 약 2.8GB)**
  원본파일을 폴더 **루트**에 그대로 넣어주세요.

## 1. 가상환경 생성 및 활성화

```bash
python3 -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

## 2. 패키지 설치

```bash
pip install -r requirements.txt
```

## 3. API 키 설정

`.env.example` 파일을 복사해 `.env` 파일을 만들고, 본인의 OpenAI API 키를 입력합니다.

```bash
cp .env.example .env
```

`.env` 파일 내용:

```
OPENAI_API_KEY=sk-여기에_본인의_키_입력
```

## 4. 실행: 대화형 챗봇 (`main.py`)

```bash
python main.py
```

1. 실행하면 먼저 탐색용 인덱스(uuid + 대표 컬럼 몇 개)를 100만 행 전체에서 불러옵니다.
   (수 초 정도 걸릴 수 있습니다.)
2. 아래 메뉴 중 하나를 선택해 대화할 페르소나 1명을 찾습니다.
   - **1) 랜덤으로 몇 명 보기**: 원하는 인원 수(기본 10명)를 무작위로 뽑아 목록으로 보여줍니다.
   - **2) 조건으로 검색하기**: 나이 범위, 직업(단어 포함), 지역(region), 시/군/구(district), 성별 중 하나로 필터링합니다.
     결과가 많으면 상위 30건만 보여줍니다.
   - **3) 인덱스/ID로 직접 조회하기**: 행 번호(0~999,999) 또는 uuid 값을 직접 입력해 바로 조회합니다.
   - 각 화면에서 `b` 입력 시 메뉴로 되돌아갈 수 있습니다.
3. 번호를 입력해 페르소나를 확정하면, 그 uuid 1건에 대해서만 51개 컬럼 전체를 조회합니다.
4. **대화 주제를 선택합니다** (`experiments/configs/h3_questions.json`/`h3_stimuli.json`이 있을 때만 표시됨):
   - 등록된 주제(청년 월세 지원 / DDP 철거 후 재개발 / 고령자 AI 돌봄 / 의대 정원 확대) 중 하나를 고르면,
     질문 유형(원본/다른 표현/반박형, 엔터 시 원본)과 정보량 단계(개요만/구체적 수치/반론 포함, 엔터 시 개요만)를 물어본 뒤
     해당 자극문+질문이 첫 메시지로 자동 전송됩니다.
   - 마지막 번호: 주제 없이 바로 자유 대화 시작.
   - 두 JSON 파일이 없으면 이 단계는 자동으로 생략되고 바로 자유 대화로 들어갑니다.
   - 두 파일에 같은 키로 주제를 추가하면 메뉴에 자동으로 반영됩니다 (코드 수정 불필요).
5. 선택한 페르소나가 되어 자유롭게 대화를 이어갑니다.
6. `exit` 또는 `종료`를 입력하면 대화가 끝납니다.

### 참고: 대용량(100만 행) 처리 방식

`ko_KR.parquet`는 100만 행 × 51개 컬럼(약 2.8GB)이라, 매번 전체를 pandas로 불러오면
느리고 메모리도 많이 사용합니다. 그래서:

- 목록 탐색(랜덤/검색/인덱스 조회) 단계에서는 `이름, 성별, 나이, 직업, 지역` 등 가벼운 컬럼만 우선 로드합니다. 
- 사용자가 페르소나 1명을 최종 확정하면, 그 uuid로 호출해 해당 1행에 대해서만 51개 컬럼 전체를 조회합니다.

### 모델 변경 (`main.py`)

기본 모델은 `gpt-4o-mini`입니다. 
다른 모델을 쓰려면 `main.py` 상단의 `MODEL_NAME` 값을 바꿔주세요.

```python
MODEL_NAME = "gpt-4o-mini"
```

## 5. H3 실험 배치 실행 (`scripts/run_h3_experiment.py`)

`experiments/configs/h3_config.json`에 지정된 **주제 1개**에 대해, 페르소나별로
질문유형 3가지 × 정보량 3단계 × 세션유형 3가지 × `repetitions`회를 자동으로 실행합니다
(`repetitions`가 5면 페르소나당 135회, 현재 설정값인 1이면 페르소나당 27회).

```bash
python scripts/run_h3_experiment.py --model gpt4o_mini
python scripts/run_h3_experiment.py --model gpt4o_mini --max-personas 2 --max-reps 1   # 소규모 테스트용
python scripts/run_h3_experiment.py --config experiments/configs/h3_config_spotcheck.json --model gpt56
```

- **인자**
  - `--model` (필수): config 파일의 `models`에 등록된 라벨. **하드코딩된 목록이 아니라, 지정한 config 파일에서 동적으로 읽습니다.**
    잘못된 라벨을 넣으면 그 config에 실제로 등록된 라벨 목록을 에러 메시지로 보여줍니다.
  - `--config` (기본값: `experiments/configs/h3_config.json`): 다른 실험 설정 파일을 쓰고 싶을 때 지정 (예: gpt-5 소규모 예비 확인용 `h3_config_spotcheck.json`)
  - `--max-personas N` (기본: 전체): 페르소나 목록에서 앞에서 N명만 사용 — 소규모 테스트용
  - `--max-reps N` (기본: config의 `repetitions`): 반복 횟수를 강제로 N회로 덮어씀 — 소규모 테스트용
- **설정 파일**
  - `experiments/configs/h3_config.json` (기본): 실행할 `topic`, 페르소나 목록 경로 `personas_file`, `repetitions`,
    모델 라벨별 `model_id`/`generation_params`/`pricing_usd_per_1k_tokens`
  - `experiments/configs/h3_questions.json`, `h3_stimuli.json`: 주제별 질문 3종/자극문 3단계
  - `personas_file`이 가리키는 JSON: 실험 대상 페르소나 목록. `{"personas": [{"persona_id": "...", ...}, ...]}` 형태면 되고,
    `group` 같은 추가 메타데이터 필드를 넣어도 됩니다 (아래 "페르소나 샘플링" 참고). 현재 값:
    - `topic`: `medical_school_quota` (의대 정원 확대)
    - `personas_file`: `experiments/h3/sampled_personas_medical.json`
  - `results_topic` (선택): 결과 저장 폴더명을 `topic`과 다르게 쓰고 싶을 때 지정. 지정 안 하면 `topic`과 동일
    (기존 동작과 100% 동일). 같은 주제 콘텐츠(질문/자극문)를 재사용하면서 본 실험 결과 폴더는 건드리지 않고
    별도로 저장하고 싶을 때 사용 — 예: `h3_config_spotcheck.json`은 `topic: medical_school_quota`로 실제 질문/자극문을
    그대로 쓰되, `results_topic: medical_school_quota_spotcheck`로 결과만 분리 저장.
- **세션유형**: `new_session`(매 회 새 대화) / `same_session_followup`(같은 세션에서 같은 질문 재확인) / `same_session_pressure`(같은 세션에서 반박 질문 추가)
- **결과 저장**: `experiments/h3/results/{results_topic 또는 topic}/{model_label}/raw_responses/{persona_id}_{question_type}_{info_level}_{session}_{rep}.json`
  (한 조합이 1~2턴이면 그 턴들을 모두 담아 1파일로 저장)
- **재실행(resume) 지원**: 이미 저장된 파일은 건너뛰므로, 중단 후 같은 명령으로 다시 실행하면 이어서 진행됩니다.
- **완료 후**: `experiments/h3/results/{results_topic 또는 topic}/{model_label}/run_meta.json`에
  `total_calls`/`completed_calls`/`failed_calls`/토큰 사용량/`estimated_cost_usd`/실행 시간이 저장됩니다.
- **진행률**: `tqdm`으로 표시됩니다.
- **비용 로그**: 실행 중 `agent.py`가 API 호출 10회마다 `[호출 N/전체] 누적 비용: $X.XX (입력: $X.XX, 출력: $X.XX)` 를 출력합니다.

> `h3_config.json`의 `pricing_usd_per_1k_tokens`는 참고용 근사치입니다. 실행 전 OpenAI 최신 요금과 맞춰 갱신하세요.
> gpt-5 계열은 아직 공식 가격이 공개되지 않아 0으로 되어 있어, `estimated_cost_usd`가 실제 비용을 반영하지 못합니다.

### 조건별 페르소나 샘플링 (`scripts/sample_medical_personas.py`)

`ko_KR.parquet`에서 지역(서울/지방) × 연령(청년/고령) 2×2 집단별로 조건에 맞는 페르소나를 뽑아
`experiments/h3/sampled_personas_medical.json`을 생성합니다 (`h3_config.json`의 `personas_file`이 가리키는 파일).

```bash
python scripts/sample_medical_personas.py
```

- **집단**: `서울_청년`(region에 "서울" 포함, 20~29세) / `서울_고령`(서울, 55세 이상) /
  `지방_청년`(서울 미포함, 20~29세) / `지방_고령`(서울 미포함, 55세 이상), 각 2명씩 총 8명
- 집단 안에 의료 관련 직업(의사/간호사/약사/의료/병원 키워드) 종사자가 있으면 최소 1명은 우선 포함
- `seed=42`로 고정되어 있어 다시 실행해도 동일한 8명이 뽑힙니다
- 출력 JSON의 각 페르소나에 `group` 라벨이 붙어 있어, `analyze_h3.py` 등에서 집단별로 나눠 분석할 때 바로 활용 가능합니다
- 다른 주제·조건으로 새 샘플이 필요하면 이 스크립트를 복사해서 집단 정의(`GROUPS`)만 바꾸면 됩니다

### GPT-5 계열 소규모 예비 확인 (`h3_config_spotcheck.json`)

4계열(gpt-4o 등)에서 발견한 패턴이 최신 추론 모델(gpt-5 계열)에서도 유지되는지 싸게 먼저 확인해보기 위한
소규모 설정입니다. `medical_school_quota` 주제의 질문/자극문을 그대로 쓰되, 결과는 본 실험과 분리 저장합니다.

- `experiments/configs/h3_config_spotcheck.json`: 페르소나 1명 × 질문유형 1개(`original`) × 정보량 3단계 ×
  세션유형 1개(`new_session`) × 반복 1회 = 모델당 3회 호출. `results_topic: "medical_school_quota_spotcheck"`로
  지정되어 있어 `experiments/h3/results/medical_school_quota/`(본 실험)에는 전혀 영향을 주지 않습니다.

- 실행 예시:
  ```bash
  python scripts/run_h3_experiment.py --config experiments/configs/h3_config_spotcheck.json --model gpt56
  python scripts/run_h3_experiment.py --config experiments/configs/h3_config_spotcheck.json --model gpt54
  ```
- 실측 결과 (2026-07 기준, 페르소나 1명·정보량 3단계 실행): `gpt-5.6`은 `temperature`를 거부하고(자동 드롭)
  `reasoning_tokens`를 실제로 소비하는 반면, `gpt-5.4`는 `temperature`를 그대로 받아들이고 `reasoning_tokens`가
  0으로 보고됨 — 같은 "gpt-5.x" 계열이라도 내부적으로 다르게 동작할 수 있음을 시사합니다.
  또한 요청한 모델 ID(`gpt-5.6` 등)가 실제로는 다른 ID(`gpt-5.6-sol` 등)로 라우팅되는 경우가 있어,
  분석 시 `actual_model` 컬럼(API가 실제로 응답한 모델명)을 함께 확인하는 게 안전합니다.

## 6. H3 결과 분석 (`scripts/analyze_h3.py`)

`experiments/configs/h3_config.json`의 `topic`을 읽어, `experiments/h3/results/{topic}/` 아래 모든 모델의
원본 응답을 로드하고 지표를 계산한 뒤 `experiments/h3/analysis/{topic}/`에 CSV 6종과 그래프(PNG) 최대 6종을 저장합니다.

```bash
python scripts/analyze_h3.py
```

- **태도 점수(찬성/반대/중립/판단불가, 1~5점)**는 `gpt-5-mini`로 판정합니다. `topic`을 프롬프트에 주입해서
  판정 모델이 어떤 정책을 이야기하는지 알고 판단하게 합니다.
  - **1세대(키워드 카운팅) 한계**: 키워드 개수로 다수결 판정했는데, "우려도 있었지만 그래도 잘한 일이라고 봐"처럼
    **반론을 언급한 뒤 결론을 내리는 양보문**에서 언급된 반론 키워드 때문에 실제 결론과 반대로 오분류됐습니다.
  - **2세대(`gpt-4o-mini` LLM 분류)**: 양보문 문제는 해결했지만, 1/3/5점만 기준을 정의하고 2/4점 기준이
    없어서 척도가 불명확했습니다 (실제 데이터의 72%가 2·4점에 몰려 있었는데도).
  - **3세대(현재, `gpt-5-mini`)**: 1~5점 전부의 기준을 명시한 프롬프트로 교체
    (1=명확한 반대, 2=소극적 반대, 3=중립, 4=조건부 찬성, 5=명확한 찬성).
    `gpt-5-mini`는 추론 모델이라 `max_tokens` 대신 `max_completion_tokens`를 써야 하고 `temperature` 등
    일부 파라미터를 거부하는데, 이 처리는 새로 만들지 않고 `agent.py`의 `_call_chat_completions`를
    그대로 재사용합니다. `response_format=json_object`를 쓰려면 프롬프트에 "json"이라는 단어가 있어야
    한다는 API 제약도 있어 프롬프트 끝에 한 줄을 추가했습니다.
  - **"판단불가"(모델이 직접 판정) vs "판정실패"(빈 응답/API 오류/파싱 실패로 우리 쪽이 판정을 못한 경우) 구분**:
    둘 다 평균 계산에서는 제외되지만 원인이 다르므로 `n_undecidable`/`n_failed`로 각각 따로 집계합니다.
  - **핵심 근거 추출/페르소나 속성 언급 여부**는 여전히 문장·키워드 기반 휴리스틱입니다 (OpenAI 호출 없음).
- **핵심 근거 유지율**은 `text-embedding-3-small` 임베딩(코사인 유사도 ≥ 0.8)을 사용합니다 — 실행 시 소액의 실제 API 비용이 발생합니다.
- **페르소나 단위 집계와 집단 비교**: 같은 페르소나의 반복(repetition) 응답은 서로 독립이 아니므로,
  응답 단위로 바로 t-test/ANOVA를 하면 안 됩니다. 그래서 먼저 페르소나×조건 단위 대표값(평균 등)을 만들고,
  집단 비교는 그 대표값을 입력으로 계산합니다.
- **평균과 분포를 항상 함께 확인**: 평균만 보면 "다들 3점 근처"인지 "1점·5점으로 양극화"됐는지 구분할 수
  없어서, 응답 단위 원본 점수 분포(%)도 별도로 계산합니다.
- **출력** (`experiments/h3/analysis/{topic}/`, 이 폴더는 `.gitignore`로 제외됨 — 언제든 재생성 가능):
  - `attitude_scores.csv`: 응답 단위 원본 (페르소나×조건×모델), `actual_model`(API가 실제로 응답한 모델명) 포함
  - `persona_level_scores.csv`: 페르소나×모델×질문유형×정보량×세션유형×턴 단위 대표값
    (`mean_score`/`std_score`/`min_score`/`max_score`/`mode_direction`/`direction_consistency`,
    `n_undecidable`/`n_failed`로 판단불가·판정실패 별도 집계, `group`/`occupation`/`age`/`age_bin`/`sex`/`region` 등
    인구통계 컬럼 포함). **이후 모든 집단 비교의 입력**이 되는 파일입니다.
  - `group_comparison.csv`: `persona_level_scores.csv`의 `mean_score`를 입력으로, 같은 실험 조건 안에서
    `group`/`sex`/`age_bin`별 ANOVA(전체 비교) + 모든 쌍 t-test 결과
  - `score_distribution.csv`: 모델×정보량×집단 조건별 1~5점 비율(%) (판단불가·판정실패 제외)
  - `consistency_metrics.csv`: 모델별 초기 입장 일치율/태도 변화량/근거 유지율/방어성/카이제곱 검정
  - `cross_analysis.csv`: 질문유형×모델, 정보량×모델 교차표 (long format)
  - `plots/`: 모델별 입장 일치율 bar, 정보량×모델 방어성 line, 모델별 근거 수 box, 모델 크기별 페르소나 정합성 bar,
    **모델별/집단별 1~5점 분포 100% 스택 막대(`score_distribution_stacked.png`/`score_distribution_by_group.png`,
    `group` 메타데이터 없으면 후자는 생성 안 함)** (PNG 최대 6개)
- `experiments/h3/results/{topic}/`가 비어 있으면 "분석할 결과 파일이 없습니다" 안내만 출력하고 종료합니다.

## `agent.py` — 페르소나 질의 모듈 (재사용 가능)

`main.py`의 `build_system_prompt`/`load_api_key`를 재사용해서, 스크립트에서 바로 불러 쓸 수 있는 함수를 제공합니다.

```python
from agent import ask_persona

result = ask_persona(
    persona_row,                 # main.py 스키마의 pd.Series (uuid 포함 51개 컬럼)
    "정부의 청년 월세 지원 정책에 대해 어떻게 생각하시나요?",
    model_id="gpt-4o-mini",       # SUPPORTED_MODELS(=PRICING_USD_PER_1M_TOKENS 키)에 등록된 모델 중 하나
    generation_params=None,       # temperature/top_p/max_tokens/frequency_penalty/presence_penalty 일부만 넘겨도 됨
    history=None,                 # 이전 턴 이어가려면 [{"role": "user"/"assistant", "content": ...}, ...]
)
# result: model_id, timestamp, persona_id, system_prompt, user_message,
#         raw_response(API 응답 전체), parsed_response(텍스트),
#         usage(prompt/completion_tokens, 있으면 reasoning_tokens도),
#         dropped_params(자동 제거된 파라미터 목록), truncated(응답이 비었거나 잘렸으면 True)
```

- 모델 가격표는 `agent.py` 상단 `PRICING_USD_PER_1M_TOKENS` dict 하나로 관리합니다 — 가격이 바뀌면 여기만 수정하면 됩니다.
  현재 `gpt-4o`/`gpt-4o-mini`/`gpt-4.1`/`gpt-5.4`/`gpt-5.6`/`gpt-5-mini`가 등록되어 있습니다.
- 호출 실패 시 `AgentAPIError`를 발생시킵니다 (인증 오류/한도 초과/네트워크 오류/API 오류를 구분해서 메시지 제공).
- **gpt-5 계열(추론 모델) 대응**:
  - `model_id`가 `"gpt-5"`로 시작하면 `max_tokens` 대신 `max_completion_tokens`를 자동으로 사용합니다.
  - `temperature`/`top_p`/`frequency_penalty`/`presence_penalty` 중 모델이 거부하는 값이 있으면
    (예: gpt-5.6은 `temperature`가 기본값 1 외에는 400 에러) 해당 파라미터만 제거하고 자동 재시도하며,
    실제로 제거된 파라미터 이름을 `dropped_params`에 남깁니다 (통제 조건이 깨졌는지 보고서에 명시할 때 사용).
  - 이 재시도 로직은 `_call_chat_completions(client, model_id, messages, params, **extra_kwargs)`라는
    내부 함수로 분리되어 있어, `agent.py` 밖에서도 재사용할 수 있습니다 (`**extra_kwargs`로
    `response_format` 등 추가 인자를 그대로 전달 가능). `scripts/analyze_h3.py`의 태도 분류기(`gpt-5-mini`)가
    이 함수를 그대로 가져다 씁니다 — 같은 파라미터 호환성 문제를 두 곳에서 따로 구현하지 않기 위함입니다.

## 파일 구성

| 파일/폴더 | 설명 |
| --- | --- |
| `main.py` | 대화형 챗봇 실행 파일 |
| `agent.py` | `ask_persona()` — 모델/파라미터를 외부에서 주입 가능한 API 호출 모듈, 비용 누적 로그 포함 |
| `scripts/run_h3_experiment.py` | H3 실험 배치 실행. `--model`(config에서 동적 검증) `--config` `--max-personas` `--max-reps` |
| `scripts/analyze_h3.py` | H3 실험 결과 분석 → CSV 6종 + 그래프 최대 6종 (topic은 `h3_config.json` 고정 경로에서 읽음) |
| `scripts/sample_medical_personas.py` | 지역×연령 2×2 집단별 페르소나 샘플링 (seed=42, 의료 직업 우선 포함) |
| `experiments/configs/h3_config.json` | 본 실험 설정 (topic, personas_file, repetitions, 모델별 파라미터/가격) |
| `experiments/configs/h3_config_spotcheck.json` | gpt-5 계열 소규모 예비 확인용 설정 (`results_topic`으로 결과를 본 실험과 분리 저장) |
| `experiments/configs/h3_questions.json` | 주제별 질문 3종(원본/다른 표현/반박형) — 청년 월세 지원, DDP 재개발, 고령자 AI 돌봄, 의대 정원 확대 |
| `experiments/configs/h3_stimuli.json` | 주제별 자극문 3단계(개요/상세/반론 포함) — 위와 동일한 4개 주제 |
| `experiments/h3/sampled_personas.json` | 기존 실험(청년 월세 지원 등)용 페르소나 uuid 목록 |
| `experiments/h3/sampled_personas_medical.json` | 의대 정원 확대 실험용 페르소나 목록 (지역×연령 그룹 라벨 포함) |
| `experiments/h3/sampled_personas_spotcheck.json` | gpt-5 소규모 예비 확인용 페르소나 1명 (medical 샘플 중 1명 추출) |
| `experiments/h3/results/` | 실행 결과 원본. `{topic}/{model_label}/...` 구조 (Git 제외, 재생성 가능) |
| `experiments/h3/analysis/` | 분석 결과 CSV/그래프. `{topic}/...` 구조 (Git 제외, 재생성 가능) |
| `requirements.txt` | 필요 패키지 목록 |
| `.env.example` | 환경변수 예시 (실제 키는 `.env`에 입력, `.env`는 Git에 포함되지 않음) |
| `ko_KR.parquet` | 페르소나 데이터 (약 100만 행). **Git에 포함되지 않음** — 별도로 전달받아 직접 추가 필요 |
