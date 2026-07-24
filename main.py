"""합성 페르소나 대화 테스트 프로그램.

ko_KR.parquet(약 100만 행)에서 페르소나 1명을 선택하면, 그 페르소나가 되어
OpenAI Chat Completions API로 대화할 수 있는 터미널 챗봇을 실행한다.
"""

import ast
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import os

# ── 설정 (필요하면 여기만 바꾸면 됨) ─────────────────────────────
MODEL_NAME = "gpt-4o-mini"
PARQUET_PATH = Path(__file__).resolve().parent / "ko_KR.parquet"
EXIT_COMMANDS = {"exit", "종료"}

# 100만 행 전체를 51개 컬럼 모두 메모리에 올리면 느리고 무거우므로,
# 목록 탐색(랜덤/검색/인덱스) 단계에서는 아래 가벼운 컬럼만 우선 로드한다.
# 실제 대화용 전체 컬럼은 사용자가 1명을 확정한 뒤 uuid로 딱 1행만 조회한다.
LIGHT_COLUMNS = ["uuid", "first_name", "last_name", "sex", "age", "occupation", "region", "district"]
SEARCH_RESULT_LIMIT = 30

H3_QUESTIONS_PATH = Path(__file__).resolve().parent / "experiments" / "configs" / "h3_questions.json"
H3_STIMULI_PATH = Path(__file__).resolve().parent / "experiments" / "configs" / "h3_stimuli.json"

QUESTION_TYPE_ORDER = ["original", "paraphrase", "pressure"]
QUESTION_TYPE_LABELS = {"original": "원본 질문", "paraphrase": "다른 표현으로 바꾼 질문", "pressure": "반박/압박형 질문"}

INFO_LEVEL_ORDER = ["overview", "detailed", "with_counterarguments"]
INFO_LEVEL_LABELS = {"overview": "개요만", "detailed": "구체적 수치 포함", "with_counterarguments": "반론/위험 정보 포함"}

BIG5_LABELS = {
    "openness": "개방성",
    "conscientiousness": "성실성",
    "extraversion": "외향성",
    "agreeableness": "우호성",
    "neuroticism": "신경성",
}

PERSONA_TEXT_COLUMNS = [
    ("professional_persona", "직업 생활"),
    ("finance_persona", "경제·소비 생활"),
    ("healthcare_persona", "건강 관리"),
    ("sports_persona", "운동·스포츠"),
    ("arts_persona", "문화·예술"),
    ("travel_persona", "여행"),
    ("culinary_persona", "음식·식생활"),
    ("family_persona", "가족 관계"),
]


def safe_input(prompt: str) -> str:
    """Ctrl+C / Ctrl+D 입력 시 트레이스백 대신 안내 후 정상 종료."""
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        print("\n프로그램을 종료합니다.")
        sys.exit(0)


def load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        print(
            "[오류] OPENAI_API_KEY를 찾을 수 없습니다.\n"
            "  1) .env.example 파일을 .env로 복사하세요.\n"
            "  2) .env 파일을 열어 OPENAI_API_KEY=본인의_API_키 형태로 입력하세요.\n"
            "  3. 다시 실행해주세요.",
            file=sys.stderr,
        )
        sys.exit(1)
    return api_key


def load_persona_index() -> pd.DataFrame:
    """탐색용 가벼운 인덱스(uuid + 대표 컬럼 몇 개)만 100만 행 전체에서 로드."""
    if not PARQUET_PATH.exists():
        print(f"[오류] {PARQUET_PATH.name} 파일을 찾을 수 없습니다. 프로그램과 같은 폴더에 두어주세요.", file=sys.stderr)
        sys.exit(1)
    try:
        print(f"페르소나 데이터를 불러오는 중입니다... ({PARQUET_PATH.name})")
        return pd.read_parquet(PARQUET_PATH, columns=LIGHT_COLUMNS)
    except Exception as e:
        print(f"[오류] parquet 파일을 읽는 중 문제가 발생했습니다: {e}", file=sys.stderr)
        sys.exit(1)


def fetch_full_persona(uuid_value: str) -> pd.Series:
    """확정된 uuid 1건에 대해서만 51개 컬럼 전체를 조회 (전체 로드 방지)."""
    try:
        row_df = pd.read_parquet(PARQUET_PATH, filters=[("uuid", "=", uuid_value)])
    except Exception as e:
        print(f"[오류] 페르소나 상세 정보를 불러오는 중 문제가 발생했습니다: {e}", file=sys.stderr)
        sys.exit(1)
    if row_df.empty:
        print(f"[오류] uuid={uuid_value} 에 해당하는 페르소나를 찾을 수 없습니다.", file=sys.stderr)
        sys.exit(1)
    return row_df.iloc[0]


def safe_str(value) -> str | None:
    """NaN/빈 문자열/'해당없음'을 None으로 정리."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text in ("", "nan", "해당없음"):
        return None
    return text


def parse_trait(raw) -> tuple[str, str] | None:
    """Big5 성격 컬럼(JSON 문자열)에서 (label, description) 추출."""
    text = safe_str(raw)
    if text is None:
        return None
    try:
        data = json.loads(text)
        return data.get("label"), data.get("description")
    except (json.JSONDecodeError, TypeError):
        return None


def parse_list_field(raw) -> list[str]:
    """"['a', 'b']" 형태의 문자열을 리스트로 변환. 실패 시 빈 리스트."""
    text = safe_str(raw)
    if text is None:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed]
    except (ValueError, SyntaxError):
        pass
    return []


def get_full_name(row: pd.Series) -> str:
    last = safe_str(row.get("last_name")) or ""
    first = safe_str(row.get("first_name")) or ""
    name = f"{last}{first}".strip()
    return name if name else "이름 미상"


def describe_light_row(row: pd.Series) -> str:
    name = get_full_name(row)
    age = safe_str(row.get("age"))
    sex = safe_str(row.get("sex"))
    occupation = safe_str(row.get("occupation")) or "직업 정보 없음"
    age_sex = f"{age}세 {sex}".strip() if age or sex else "정보 없음"
    return f"{name} ({age_sex}) - {occupation}"


def pick_from_subset(subset_df: pd.DataFrame) -> str | None:
    """번호가 붙은 목록을 보여주고 선택된 행의 uuid를 반환. 'b' 입력 시 None(뒤로가기)."""
    if subset_df.empty:
        print("조건에 맞는 페르소나가 없습니다.\n")
        return None

    for local_num, (_, row) in enumerate(subset_df.iterrows(), start=1):
        print(f"{local_num:3d}. {describe_light_row(row)}")
    print()

    total = len(subset_df)
    while True:
        choice = safe_input(f"번호 입력 (1-{total}, 뒤로가기는 b): ").strip().lower()
        if choice == "b":
            return None
        if not choice.isdigit():
            print("숫자 또는 b를 입력해주세요.")
            continue
        num = int(choice)
        if not (1 <= num <= total):
            print(f"1에서 {total} 사이의 번호를 입력해주세요.")
            continue
        return subset_df.iloc[num - 1]["uuid"]


def random_view(index_df: pd.DataFrame) -> str | None:
    raw = safe_input("몇 명을 무작위로 보시겠어요? (기본 10): ").strip()
    n = int(raw) if raw.isdigit() and int(raw) > 0 else 10
    n = min(n, len(index_df))
    sample_df = index_df.sample(n=n).reset_index(drop=True)
    print(f"\n=== 무작위 {n}명 ===")
    return pick_from_subset(sample_df)


def search_view(index_df: pd.DataFrame) -> str | None:
    print("\n어떤 조건으로 검색할까요?")
    print("  1) 나이 범위")
    print("  2) 직업 (일부 단어 포함)")
    print("  3) 지역 (region, 예: 서울)")
    print("  4) 시/군/구 (district, 일부 단어 포함)")
    print("  5) 성별")
    print("  b) 뒤로가기")
    field_choice = safe_input("선택: ").strip().lower()

    if field_choice == "b":
        return None

    mask = None
    if field_choice == "1":
        min_age_raw = safe_input("최소 나이 (엔터 시 제한 없음): ").strip()
        max_age_raw = safe_input("최대 나이 (엔터 시 제한 없음): ").strip()
        mask = pd.Series(True, index=index_df.index)
        if min_age_raw.isdigit():
            mask &= index_df["age"] >= int(min_age_raw)
        if max_age_raw.isdigit():
            mask &= index_df["age"] <= int(max_age_raw)
    elif field_choice == "2":
        keyword = safe_input("직업에 포함될 단어: ").strip()
        mask = index_df["occupation"].str.contains(keyword, na=False)
    elif field_choice == "3":
        keyword = safe_input("지역명 (예: 서울, 경기): ").strip()
        mask = index_df["region"].str.contains(keyword, na=False)
    elif field_choice == "4":
        keyword = safe_input("시/군/구에 포함될 단어: ").strip()
        mask = index_df["district"].str.contains(keyword, na=False)
    elif field_choice == "5":
        keyword = safe_input("성별 (남자/여자): ").strip()
        mask = index_df["sex"] == keyword
    else:
        print("올바른 항목을 선택해주세요.\n")
        return None

    result_df = index_df[mask]
    total_hits = len(result_df)
    if total_hits == 0:
        print("조건에 맞는 페르소나가 없습니다.\n")
        return None

    shown_df = result_df.head(SEARCH_RESULT_LIMIT).reset_index(drop=True)
    if total_hits > SEARCH_RESULT_LIMIT:
        print(f"\n=== 검색 결과 총 {total_hits}건 중 상위 {SEARCH_RESULT_LIMIT}건 표시 ===")
    else:
        print(f"\n=== 검색 결과 {total_hits}건 ===")
    return pick_from_subset(shown_df)


def index_id_view(index_df: pd.DataFrame) -> str | None:
    total = len(index_df)
    raw = safe_input(f"행 번호(0-{total - 1}) 또는 uuid 값을 입력하세요 (뒤로가기는 b): ").strip()
    if raw.lower() == "b":
        return None
    if raw.isdigit() and 0 <= int(raw) < total:
        return index_df.iloc[int(raw)]["uuid"]
    matched = index_df[index_df["uuid"] == raw]
    if matched.empty:
        print("해당 행 번호/uuid를 찾을 수 없습니다.\n")
        return None
    return matched.iloc[0]["uuid"]


def choose_persona_uuid(index_df: pd.DataFrame) -> str:
    """메뉴를 반복 표시하며 사용자가 최종 uuid 1개를 확정할 때까지 진행."""
    while True:
        print("\n=== 페르소나 선택 ===")
        print(f"(전체 {len(index_df):,}명)")
        print("  1) 랜덤으로 몇 명 보기")
        print("  2) 조건으로 검색하기")
        print("  3) 인덱스/ID로 직접 조회하기")
        menu_choice = safe_input("메뉴 선택 (1/2/3): ").strip()

        if menu_choice == "1":
            uuid_value = random_view(index_df)
        elif menu_choice == "2":
            uuid_value = search_view(index_df)
        elif menu_choice == "3":
            uuid_value = index_id_view(index_df)
        else:
            print("1, 2, 3 중 하나를 입력해주세요.")
            continue

        if uuid_value is not None:
            return uuid_value


def load_h3_content() -> tuple[dict, dict] | tuple[None, None]:
    """H3 실험용 질문/자극 JSON을 로드. 파일이 없으면 (None, None)."""
    if not H3_QUESTIONS_PATH.exists() or not H3_STIMULI_PATH.exists():
        return None, None
    with open(H3_QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    with open(H3_STIMULI_PATH, encoding="utf-8") as f:
        stimuli = json.load(f)
    return questions, stimuli


def choose_from_labels(title: str, order: list[str], labels: dict[str, str], default: str) -> str:
    print(f"\n{title}을 선택하세요 (엔터 시 기본값: {labels[default]})")
    for i, key in enumerate(order, start=1):
        print(f"  {i}) {labels[key]}")
    choice = safe_input(f"선택 (1-{len(order)}, 엔터=기본값): ").strip()
    if not choice:
        return default
    if choice.isdigit() and 1 <= int(choice) <= len(order):
        return order[int(choice) - 1]
    print("올바르지 않은 입력이라 기본값을 사용합니다.")
    return default


def choose_opening_message() -> str | None:
    """H3 토픽을 고르고 질문유형/정보량을 선택해 오프닝 메시지를 구성.
    토픽 파일이 없거나 '주제 없이 자유롭게 대화'를 고르면 None."""
    questions, stimuli = load_h3_content()
    if questions is None:
        return None

    topics = [t for t in questions if t in stimuli]
    if not topics:
        return None

    print("\n=== 대화 주제를 선택하세요 ===")
    for i, topic in enumerate(topics, start=1):
        label = stimuli[topic].get("topic") or questions[topic].get("topic") or topic
        print(f"  {i}) {label}")
    free_chat_option = len(topics) + 1
    print(f"  {free_chat_option}) 주제 없이 자유롭게 대화")

    while True:
        choice = safe_input(f"선택 (1-{free_chat_option}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(topics):
            topic = topics[int(choice) - 1]
            break
        if choice.isdigit() and int(choice) == free_chat_option:
            return None
        print("올바른 번호를 입력해주세요.")

    question_type = choose_from_labels("질문 유형", QUESTION_TYPE_ORDER, QUESTION_TYPE_LABELS, default="original")
    info_level = choose_from_labels("정보량 단계", INFO_LEVEL_ORDER, INFO_LEVEL_LABELS, default="overview")

    stimulus_text = stimuli[topic][info_level]
    question_text = questions[topic][question_type]
    return f"{stimulus_text}\n\n{question_text}"


def build_system_prompt(row: pd.Series) -> str:
    name = get_full_name(row)
    sections: list[str] = [f"당신은 '{name}'입니다. 지금부터 아래 설명에 맞는 인물이 되어 사용자와 대화합니다."]

    # 1) 기본 인적 정보
    basic_facts = []
    age = safe_str(row.get("age"))
    if age:
        basic_facts.append(f"나이는 {age}세")
    sex = safe_str(row.get("sex"))
    if sex:
        basic_facts.append(f"성별은 {sex}")
    district = safe_str(row.get("district"))
    if district:
        basic_facts.append(f"거주지는 {district.replace('-', ' ')}")
    marital_status = safe_str(row.get("marital_status"))
    if marital_status:
        basic_facts.append(f"혼인 상태는 {marital_status}")
    family_type = safe_str(row.get("family_type"))
    if family_type:
        basic_facts.append(f"가족 형태는 {family_type}")
    if basic_facts:
        sections.append("당신은 " + ", ".join(basic_facts) + "인 사람입니다.")

    # 2) 학력·직업·경제 상태
    social_facts = []
    education_level = safe_str(row.get("education_level"))
    if education_level:
        social_facts.append(f"최종 학력은 {education_level}")
    bachelors_field = safe_str(row.get("bachelors_field"))
    if bachelors_field:
        social_facts.append(f"전공은 {bachelors_field}")
    occupation = safe_str(row.get("occupation"))
    if occupation:
        social_facts.append(f"직업은 {occupation}")
    economic_activity_status = safe_str(row.get("economic_activity_status"))
    if economic_activity_status:
        social_facts.append(f"경제활동 상태는 {economic_activity_status}")
    income_bracket = safe_str(row.get("income_bracket"))
    if income_bracket:
        social_facts.append(f"소득 수준은 {income_bracket}")
    housing_type = safe_str(row.get("housing_type"))
    housing_tenure = safe_str(row.get("housing_tenure"))
    if housing_type or housing_tenure:
        housing = " ".join(filter(None, [housing_type, housing_tenure]))
        social_facts.append(f"주거 형태는 {housing}")
    military_status = safe_str(row.get("military_status"))
    if military_status:
        social_facts.append(f"병역 상태는 {military_status}")
    if social_facts:
        sections.append(", ".join(social_facts) + "입니다.")

    # 3) 건강 상태
    health_facts = []
    for col, label in [
        ("bmi_status", "체질량지수(BMI)"),
        ("blood_pressure_status", "혈압"),
        ("blood_sugar_status", "혈당"),
        ("waist_status", "허리둘레"),
    ]:
        val = safe_str(row.get(col))
        if val:
            health_facts.append(f"{label}은 {val} 수준")
    smoking_status = safe_str(row.get("smoking_status"))
    if smoking_status:
        health_facts.append(f"흡연 여부는 {smoking_status}")
    drinking_status = safe_str(row.get("drinking_status"))
    if drinking_status:
        health_facts.append(f"음주 여부는 {drinking_status}")
    if health_facts:
        sections.append("건강 상태는 " + ", ".join(health_facts) + "입니다.")

    # 4) 성격 (Big5)
    personality_sentences = []
    for col, kor_label in BIG5_LABELS.items():
        parsed = parse_trait(row.get(col))
        if parsed:
            label, description = parsed
            desc_part = f" {description}" if description else ""
            personality_sentences.append(f"{kor_label}은 '{label}' 수준으로,{desc_part}")
    if personality_sentences:
        sections.append("성격 특성은 다음과 같습니다.\n" + "\n".join(f"- {s}" for s in personality_sentences))

    # 5) 성장 배경
    cultural_background = safe_str(row.get("cultural_background"))
    if cultural_background:
        sections.append(f"성장 배경과 가치관: {cultural_background}")

    # 6) 역량·커리어
    skills_text = safe_str(row.get("skills_and_expertise"))
    skills_list = parse_list_field(row.get("skills_and_expertise_list"))
    career_goals = safe_str(row.get("career_goals_and_ambitions"))
    career_parts = []
    if skills_text:
        career_parts.append(skills_text)
    if skills_list:
        career_parts.append("주요 역량: " + ", ".join(skills_list))
    if career_goals:
        career_parts.append(f"목표와 야망: {career_goals}")
    if career_parts:
        sections.append("역량과 커리어:\n" + "\n".join(career_parts))

    # 7) 취미·관심사
    hobbies_text = safe_str(row.get("hobbies_and_interests"))
    hobbies_list = parse_list_field(row.get("hobbies_and_interests_list"))
    hobby_parts = []
    if hobbies_text:
        hobby_parts.append(hobbies_text)
    if hobbies_list:
        hobby_parts.append("관심사 목록: " + ", ".join(hobbies_list))
    if hobby_parts:
        sections.append("취미와 관심사:\n" + "\n".join(hobby_parts))

    # 8) 영역별 상세 페르소나
    domain_parts = []
    for col, label in PERSONA_TEXT_COLUMNS:
        text = safe_str(row.get(col))
        if text:
            domain_parts.append(f"[{label}] {text}")
    if domain_parts:
        sections.append("세부 생활 모습:\n" + "\n".join(domain_parts))

    # 9) 전체 요약
    detailed_persona = safe_str(row.get("detailed_persona")) or safe_str(row.get("persona"))
    if detailed_persona:
        sections.append(f"전체 요약: {detailed_persona}")

    sections.append(
        "이 페르소나로서 1인칭('나')으로 자연스럽게 대화하세요. "
        "위 배경, 성격, 말투에 어울리는 관점과 어휘를 유지하고, "
        "당신이 AI라는 사실은 언급하지 마세요."
    )

    return "\n\n".join(sections)


def run_chat_loop(
    client, model_name: str, persona_name: str, system_prompt: str, opening_message: str | None = None
) -> None:
    messages = [{"role": "system", "content": system_prompt}]

    print(f"\n=== '{persona_name}' 페르소나와 대화를 시작합니다 ===")
    print("(종료하려면 'exit' 또는 '종료'를 입력하세요)\n")

    import openai  # 예외 클래스 접근용

    def send(user_text: str) -> bool:
        """user_text를 보내고 응답을 출력. 대화를 계속할 수 있으면 True, 중단해야 하면 False."""
        messages.append({"role": "user", "content": user_text})
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
            )
            reply = response.choices[0].message.content
            messages.append({"role": "assistant", "content": reply})
            print(f"\n{persona_name}: {reply}\n")
            return True
        except openai.AuthenticationError:
            print("[오류] API 키가 유효하지 않습니다. .env의 OPENAI_API_KEY를 확인해주세요.")
            messages.pop()
            return False
        except openai.RateLimitError:
            print("[오류] API 요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요.")
            messages.pop()
            return True
        except openai.APIConnectionError:
            print("[오류] 네트워크 연결에 문제가 발생했습니다. 인터넷 연결을 확인해주세요.")
            messages.pop()
            return True
        except openai.APIStatusError as e:
            print(f"[오류] OpenAI API 오류가 발생했습니다 (status={e.status_code}). 잠시 후 다시 시도해주세요.")
            messages.pop()
            return True
        except Exception as e:
            print(f"[오류] 예상치 못한 문제가 발생했습니다: {e}")
            messages.pop()
            return True

    if opening_message:
        print(f"You: {opening_message}\n")
        if not send(opening_message):
            return

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n대화를 종료합니다.")
            break

        if not user_input:
            continue
        if user_input.lower() in EXIT_COMMANDS:
            print("대화를 종료합니다.")
            break

        if not send(user_input):
            break


def main() -> None:
    api_key = load_api_key()
    index_df = load_persona_index()

    selected_uuid = choose_persona_uuid(index_df)
    selected_row = fetch_full_persona(selected_uuid)
    persona_name = get_full_name(selected_row)
    system_prompt = build_system_prompt(selected_row)
    opening_message = choose_opening_message()

    from openai import OpenAI

    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        print(f"[오류] OpenAI 클라이언트를 초기화하지 못했습니다: {e}", file=sys.stderr)
        sys.exit(1)

    run_chat_loop(client, MODEL_NAME, persona_name, system_prompt, opening_message=opening_message)


if __name__ == "__main__":
    main()
