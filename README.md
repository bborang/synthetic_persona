# 합성 페르소나 대화 테스트

`ko_KR.parquet`에 담긴 약 100만 명의 합성 페르소나 중 1명을 선택하면,
그 인물이 되어 OpenAI Chat Completions API로 대화할 수 있는 터미널 프로그램입니다.

## 0. 시작하기 전에 (팀원 필수 준비물)

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

## 3. API 키 설정

`.env.example` 파일을 복사해 `.env` 파일을 만들고, 본인의 OpenAI API 키를 입력합니다.

```bash
cp .env.example .env
```

`.env` 파일 내용:

```
OPENAI_API_KEY=sk-여기에_본인의_키_입력
```

## 4. 실행

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
3. 목록에서 번호를 입력해 페르소나를 확정하면(각 화면에서 `b` 입력 시 메뉴로 되돌아갈 수 있습니다),
   그 uuid 1건에 대해서만 51개 컬럼 전체를 조회해 대화를 시작합니다.
4. 선택한 페르소나가 되어 자유롭게 대화를 나눕니다.
5. `exit` 또는 `종료`를 입력하면 대화가 끝납니다.

### 참고: 대용량(100만 행) 처리 방식

`ko_KR.parquet`는 100만 행 × 51개 컬럼(약 2.8GB)이라, 매번 전체를 pandas로 불러오면
느리고 메모리도 많이 사용합니다. 그래서 이 프로그램은:

- 목록 탐색(랜덤/검색/인덱스 조회) 단계에서는 `uuid, first_name, last_name, sex, age, occupation, region, district` 등
  가벼운 컬럼만 우선 로드합니다(`pandas.read_parquet(..., columns=[...])`).
- 사용자가 페르소나 1명을 최종 확정하면, 그 uuid로 `pandas.read_parquet(..., filters=[("uuid", "=", 값)])`를 호출해
  해당 1행에 대해서만 51개 컬럼 전체를 조회합니다.

이 정도 수준이면 개인 PC에서도 충분히 동작하며, 더 빠른 응답이 필요하다면 별도의 DB나 인덱싱(예: DuckDB)
도입을 고려할 수 있습니다.

## 모델 변경

기본 모델은 `gpt-4o-mini`입니다. 다른 모델을 쓰려면 `main.py` 상단의
`MODEL_NAME` 값을 원하는 모델명으로 바꿔주세요.

```python
MODEL_NAME = "gpt-4o-mini"
```

## 파일 구성

| 파일 | 설명 |
| --- | --- |
| `main.py` | 실행 파일 |
| `requirements.txt` | 필요 패키지 목록 |
| `.env.example` | 환경변수 예시 (실제 키는 `.env`에 입력, `.env`는 Git에 포함되지 않음) |
| `ko_KR.parquet` | 페르소나 데이터 (약 100만 행). **Git에 포함되지 않음** — 별도로 전달받아 직접 추가 필요 |
