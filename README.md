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
   ```bash
   cp .env.example .env
   ```
   ```
   OPENAI_API_KEY=본인의_실제_API_키
   ```
   `.env`는 절대 커밋하지 마세요. (`.gitignore`에 이미 등록되어 있습니다.)

2. **`ko_KR.parquet` (원본 페르소나 데이터, 약 2.8GB)**
   용량 문제뿐 아니라 개인정보/민감정보 성격상 Git에 올리지 않습니다.
   전달받은 파일을 `test_API` 폴더 **루트**에 그대로 넣어주세요.
   > TODO: 원본 데이터 전달 방법/경로를 여기에 채워주세요. (예: 사내 스토리지 링크, 담당자 문의 등)

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

`main.py` 실행에는 `openai`/`pandas`/`pyarrow`/`python-dotenv`만 있으면 됩니다.
`numpy`/`scipy`/`matplotlib`/`scikit-learn`/`tqdm`은 `scripts/` 아래 H3 실험·분석 스크립트에서만 사용합니다.

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

- 목록 탐색(랜덤/검색/인덱스 조회) 단계에서는 `uuid, first_name, last_name, sex, age, occupation, region, district` 등
  가벼운 컬럼만 우선 로드합니다(`pandas.read_parquet(..., columns=[...])`).
- 사용자가 페르소나 1명을 최종 확정하면, 그 uuid로 `pandas.read_parquet(..., filters=[("uuid", "=", 값)])`를 호출해
  해당 1행에 대해서만 51개 컬럼 전체를 조회합니다.

### 모델 변경 (`main.py`)

기본 모델은 `gpt-4o-mini`입니다. 다른 모델을 쓰려면 `main.py` 상단의 `MODEL_NAME` 값을 바꿔주세요.

```python
MODEL_NAME = "gpt-4o-mini"
```

## 5. H3 실험 배치 실행 (`scripts/run_h3_experiment.py`)

`experiments/configs/h3_config.json`에 지정된 **주제 1개**에 대해, 페르소나별로
질문유형 3가지 × 정보량 3단계 × 세션유형 3가지 × `repetitions`회를 자동으로 실행합니다
(`repetitions`가 5면 페르소나당 135회, 현재 설정값인 1이면 페르소나당 27회).

```bash
python scripts/run_h3_experiment.py --model gpt4o_mini   # gpt4o | gpt4o_mini | gpt41
```

- **설정 파일**
  - `experiments/configs/h3_config.json`: 실행할 `topic`, 페르소나 목록 경로 `personas_file`, `repetitions`,
    모델 라벨별 `model_id`/`generation_params`/`pricing_usd_per_1k_tokens`
  - `experiments/configs/h3_questions.json`, `h3_stimuli.json`: 주제별 질문 3종/자극문 3단계
  - `personas_file`이 가리키는 JSON: 실험 대상 페르소나 목록. `{"personas": [{"persona_id": "...", ...}, ...]}` 형태면 되고,
    `group` 같은 추가 메타데이터 필드를 넣어도 됩니다 (아래 "페르소나 샘플링" 참고). 현재 값:
    - `topic`: `medical_school_quota` (의대 정원 확대)
    - `personas_file`: `experiments/h3/sampled_personas_medical.json`
- **세션유형**: `new_session`(매 회 새 대화) / `same_session_followup`(같은 세션에서 같은 질문 재확인) / `same_session_pressure`(같은 세션에서 반박 질문 추가)
- **결과 저장**: `experiments/h3/results/{model_label}/raw_responses/{persona_id}_{question_type}_{info_level}_{session}_{rep}.json`
  (한 조합이 1~2턴이면 그 턴들을 모두 담아 1파일로 저장)
- **재실행(resume) 지원**: 이미 저장된 파일은 건너뛰므로, 중단 후 같은 명령으로 다시 실행하면 이어서 진행됩니다.
- **완료 후**: `experiments/h3/results/{model_label}/run_meta.json`에 `total_calls`/`completed_calls`/`failed_calls`/토큰 사용량/`estimated_cost_usd`/실행 시간이 저장됩니다.
- **진행률**: `tqdm`으로 표시됩니다.
- **비용 로그**: 실행 중 `agent.py`가 API 호출 10회마다 `[호출 N/전체] 누적 비용: $X.XX (입력: $X.XX, 출력: $X.XX)` 를 출력합니다.

> `h3_config.json`의 `pricing_usd_per_1k_tokens`는 참고용 근사치입니다. 실행 전 OpenAI 최신 요금과 맞춰 갱신하세요.

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

## 6. H3 결과 분석 (`scripts/analyze_h3.py`)

`experiments/h3/results/` 아래 모든 모델의 원본 응답을 읽어 지표를 계산하고, CSV/그래프로 저장합니다.

```bash
python scripts/analyze_h3.py
```

- **태도 점수(찬성/반대/중립, 1~5점)**는 `gpt-4o-mini`로 판정합니다.
  - **이전 버전(키워드 카운팅 방식)의 한계**: 응답 텍스트에서 "찬성"/"반대" 등 키워드 개수를 세서 다수결로 판정했는데,
    "예산 낭비 우려도 있었지만, 그래도 잘한 일이라고 봐"처럼 **반론을 언급한 뒤 결론을 내리는 양보문**에서
    실제 결론과 무관하게 언급된 반론 키워드("우려") 때문에 반대로 오분류되는 문제가 있었습니다.
    키워드 매칭은 문장 구조·양보 관계를 전혀 이해하지 못하기 때문에 구조적으로 해결이 불가능했습니다.
  - **해결**: `gpt-4o-mini`에게 "응답자의 최종 결론을 기준으로 판단하라"고 명시한 프롬프트로 분류하도록 교체했습니다.
    비용은 응답 1건당 약 $0.00009로 무시할 수 있는 수준이며(실험 생성 자체의 1/50~1/100), 동일 텍스트는 캐시로
    재사용해 중복 호출도 방지합니다. 실행 중 총 분류 비용이 터미널에 출력됩니다.
  - **핵심 근거 추출/페르소나 속성 언급 여부**는 여전히 문장·키워드 기반 휴리스틱입니다 (OpenAI 호출 없음).
- **핵심 근거 유지율**은 `text-embedding-3-small` 임베딩(코사인 유사도 ≥ 0.8)을 사용합니다 — 실행 시 소액의 실제 API 비용이 발생합니다.
- **출력** (`experiments/h3/analysis/`, 이 폴더는 `.gitignore`로 제외됨 — 언제든 재생성 가능):
  - `attitude_scores.csv`: 응답 단위 원본 (페르소나×조건×모델)
  - `consistency_metrics.csv`: 모델별 초기 입장 일치율/태도 변화량/근거 유지율/방어성/카이제곱 검정
  - `cross_analysis.csv`: 질문유형×모델, 정보량×모델 교차표 (long format)
  - `plots/`: 모델별 입장 일치율 bar, 정보량×모델 방어성 line, 모델별 근거 수 box, 모델 크기별 페르소나 정합성 bar (PNG 4개)
- `experiments/h3/results/`가 비어 있으면 "분석할 결과 파일이 없습니다" 안내만 출력하고 종료합니다.

## `agent.py` — 페르소나 질의 모듈 (재사용 가능)

`main.py`의 `build_system_prompt`/`load_api_key`를 재사용해서, 스크립트에서 바로 불러 쓸 수 있는 함수를 제공합니다.

```python
from agent import ask_persona

result = ask_persona(
    persona_row,                 # main.py 스키마의 pd.Series (uuid 포함 51개 컬럼)
    "정부의 청년 월세 지원 정책에 대해 어떻게 생각하시나요?",
    model_id="gpt-4o-mini",       # "gpt-4o" | "gpt-4o-mini" | "gpt-4.1"
    generation_params=None,       # temperature/top_p/max_tokens/frequency_penalty/presence_penalty 일부만 넘겨도 됨
    history=None,                 # 이전 턴 이어가려면 [{"role": "user"/"assistant", "content": ...}, ...]
)
# result: model_id, timestamp, persona_id, system_prompt, user_message,
#         raw_response(API 응답 전체), parsed_response(텍스트), usage(prompt/completion_tokens)
```

- 모델 가격표는 `agent.py` 상단 `PRICING_USD_PER_1M_TOKENS` dict 하나로 관리합니다 — 가격이 바뀌면 여기만 수정하면 됩니다.
- 호출 실패 시 `AgentAPIError`를 발생시킵니다 (인증 오류/한도 초과/네트워크 오류/API 오류를 구분해서 메시지 제공).

## 파일 구성

| 파일/폴더 | 설명 |
| --- | --- |
| `main.py` | 대화형 챗봇 실행 파일 |
| `agent.py` | `ask_persona()` — 모델/파라미터를 외부에서 주입 가능한 API 호출 모듈, 비용 누적 로그 포함 |
| `scripts/run_h3_experiment.py` | H3 실험 배치 실행 (`--model gpt4o\|gpt4o_mini\|gpt41`) |
| `scripts/analyze_h3.py` | H3 실험 결과 분석 → CSV 3종 + 그래프 4종 |
| `scripts/sample_medical_personas.py` | 지역×연령 2×2 집단별 페르소나 샘플링 (seed=42, 의료 직업 우선 포함) |
| `experiments/configs/h3_config.json` | 실험 설정 (topic, personas_file, repetitions, 모델별 파라미터/가격) |
| `experiments/configs/h3_questions.json` | 주제별 질문 3종(원본/다른 표현/반박형) — 청년 월세 지원, DDP 재개발, 고령자 AI 돌봄, 의대 정원 확대 |
| `experiments/configs/h3_stimuli.json` | 주제별 자극문 3단계(개요/상세/반론 포함) — 위와 동일한 4개 주제 |
| `experiments/h3/sampled_personas.json` | 기존 실험(청년 월세 지원 등)용 페르소나 uuid 목록 |
| `experiments/h3/sampled_personas_medical.json` | 의대 정원 확대 실험용 페르소나 목록 (지역×연령 그룹 라벨 포함) |
| `experiments/h3/results/` | 실행 결과 원본 (Git 제외, 재생성 가능) |
| `experiments/h3/analysis/` | 분석 결과 CSV/그래프 (Git 제외, 재생성 가능) |
| `requirements.txt` | 필요 패키지 목록 |
| `.env.example` | 환경변수 예시 (실제 키는 `.env`에 입력, `.env`는 Git에 포함되지 않음) |
| `ko_KR.parquet` | 페르소나 데이터 (약 100만 행). **Git에 포함되지 않음** — 별도로 전달받아 직접 추가 필요 |
