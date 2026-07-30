"""H3 실험 실행 스크립트.

experiments/configs/h3_config.json 의 설정으로,
config의 "personas_file"이 가리키는 JSON에 담긴 페르소나들에 대해
질문유형 3가지 x 정보량 3단계 x 세션유형 3가지 x repetitions회를 실행하고,
각 결과를 experiments/h3/results/{topic}/{model_label}/raw_responses/ 에 저장한다.

기본적으로 --concurrency(기본 5)만큼 스레드로 조합(combo)을 동시에 처리한다.
같은 조합 안의 턴(turn)은 순서대로(동기) 실행되고, 서로 다른 조합끼리만 병렬로 실행된다.
429(RateLimitError)가 나면 지수 백오프로 자동 재시도한다.

사용법:
    python scripts/run_h3_experiment.py --model gpt4o_mini
    python scripts/run_h3_experiment.py --model gpt4o_mini --max-personas 2 --max-reps 1  # 소규모 테스트용
    python scripts/run_h3_experiment.py --config experiments/configs/h3_config_spotcheck.json --model gpt56nano
    python scripts/run_h3_experiment.py --config experiments/configs/h3_config_v4.json --model gpt4o --concurrency 10
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import openai
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import ask_persona, AgentAPIError  # noqa: E402
from main import PARQUET_PATH, build_system_prompt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = "experiments/configs/h3_config.json"
QUESTIONS_PATH = PROJECT_ROOT / "experiments" / "configs" / "h3_questions.json"
STIMULI_PATH = PROJECT_ROOT / "experiments" / "configs" / "h3_stimuli.json"
RESULTS_ROOT = PROJECT_ROOT / "experiments" / "h3" / "results"

MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 2.0


def fetch_personas_batch(uuid_list: list[str]) -> dict:
    """개별 fetch_full_persona()를 N번 부르면 매번 전체 parquet를 스캔해서 느리다
    (150명 기준 실측 약 16분). "in" 필터로 한 번에 읽으면 같은 150명이 약 8초.
    반환: {uuid: pd.Series}
    """
    df = pd.read_parquet(PARQUET_PATH, filters=[("uuid", "in", uuid_list)])
    return {row["uuid"]: row for _, row in df.iterrows()}


def call_with_backoff(fn, max_retries: int = MAX_RETRIES, base_delay: float = BACKOFF_BASE_SECONDS):
    """fn()을 실행하다 429(RateLimitError)를 만나면 지수 백오프 후 재시도.
    다른 종류의 실패는 재시도하지 않고 그대로 올린다."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except AgentAPIError as e:
            if not isinstance(e.__cause__, openai.RateLimitError) or attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            tqdm.write(f"[429] 요청 한도 초과, {delay:.0f}초 후 재시도 ({attempt + 1}/{max_retries})")
            time.sleep(delay)


def load_json(path: Path) -> dict:
    if not path.exists():
        print(f"[오류] 파일을 찾을 수 없습니다: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_first_message(stimulus_text: str, question_text: str) -> str:
    return f"{stimulus_text}\n\n{question_text}"


# 실측 보정치: gpt-5.6 spotcheck에서 system_prompt+첫 메시지 3643자가 실제 prompt_tokens=2232로
# 찍힌 것을 기준으로 계산 (문자수/토큰수 ≈ 1.63). tiktoken 없이 쓰는 근사치라 정확하지 않을 수 있음 —
# 여유 있게(과소평가 방지) completion은 max_tokens(상한)로 잡아서 "최악의 경우" 비용을 추정한다.
CHARS_PER_TOKEN_ESTIMATE = 1.63


def estimate_prompt_tokens(text: str) -> int:
    return max(1, round(len(text) / CHARS_PER_TOKEN_ESTIMATE))


def dry_run_report(
    personas, question_types, info_levels, session_types, repetitions,
    topic_questions, topic_stimuli, persona_rows, model_id, generation_params, pricing,
) -> dict:
    """실제 API를 호출하지 않고, 예상 호출 수와 비용(상한 추정)을 계산.

    prompt_tokens는 실제 system_prompt+첫 메시지 텍스트 길이를 문자 기반으로 추정하고(로컬 계산,
    API 호출 없음), completion_tokens는 generation_params의 max_tokens(상한)를 그대로 써서
    "이 이상은 안 나온다"는 보수적인 예산 상한을 낸다 (실제 비용은 보통 이보다 낮음).
    """
    max_tokens = generation_params.get("max_tokens", 1024)
    combos = list(product(personas, question_types, info_levels, session_types, range(1, repetitions + 1)))
    total_calls = len(combos)

    prompt_estimate_cache: dict[tuple, int] = {}
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_api_requests = 0

    for persona, question_type, info_level, session_type, _rep in combos:
        persona_id = persona["persona_id"]
        cache_key = (persona_id, question_type, info_level)
        if cache_key not in prompt_estimate_cache:
            system_prompt = build_system_prompt(persona_rows[persona_id])
            first_message = build_first_message(topic_stimuli[info_level], topic_questions[question_type])
            prompt_estimate_cache[cache_key] = estimate_prompt_tokens(system_prompt + first_message)
        turn1_prompt = prompt_estimate_cache[cache_key]

        total_prompt_tokens += turn1_prompt
        total_completion_tokens += max_tokens
        total_api_requests += 1

        if session_type != "new_session":
            # 2턴째: 히스토리(직전 프롬프트+응답)가 쌓이므로 대략 turn1_prompt + max_tokens를
            # 두 번째 호출의 prompt로 잡는다 (상한 기준 추정이라 실제보다 크게 나올 수 있음).
            total_prompt_tokens += turn1_prompt + max_tokens
            total_completion_tokens += max_tokens
            total_api_requests += 1

    estimated_cost = (
        total_prompt_tokens / 1000 * pricing["prompt"]
        + total_completion_tokens / 1000 * pricing["completion"]
    )

    return {
        "model_id": model_id,
        "total_calls": total_calls,
        "total_api_requests": total_api_requests,
        "estimated_prompt_tokens": total_prompt_tokens,
        "estimated_completion_tokens_upper_bound": total_completion_tokens,
        "estimated_cost_usd_upper_bound": round(estimated_cost, 2),
    }


def run_session(
    persona_row, topic_questions, question_type, model_id, generation_params, session_type, first_message,
    total_api_requests=None,
):
    """session_type에 따라 1~2턴을 호출하고, 각 턴의 ask_persona() 결과 리스트를 반환."""
    turns = []

    turn1 = ask_persona(
        persona_row, first_message, model_id=model_id, generation_params=generation_params,
        total_calls=total_api_requests,
    )
    turns.append(turn1)

    if session_type == "new_session":
        return turns

    history = [
        {"role": "user", "content": first_message},
        {"role": "assistant", "content": turn1["parsed_response"]},
    ]

    if session_type == "same_session_followup":
        followup_message = topic_questions[question_type]
    elif session_type == "same_session_pressure":
        followup_message = topic_questions["pressure"]
    else:
        raise ValueError(f"알 수 없는 session_type입니다: {session_type!r}")

    turn2 = ask_persona(
        persona_row,
        followup_message,
        model_id=model_id,
        generation_params=generation_params,
        history=history,
        total_calls=total_api_requests,
    )
    turns.append(turn2)
    return turns


def compute_run_meta(model_id: str, results_dir: Path, total_calls: int, pricing: dict, start_time: datetime, end_time: datetime) -> dict:
    """resume를 감안해, results_dir에 실제로 저장된 결과 파일들을 스캔해서 메타데이터를 계산."""
    result_files = list(results_dir.glob("*.json"))
    completed_calls = len(result_files)

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_api_requests = 0
    turns_with_dropped_params = 0
    dropped_params_counts: dict[str, int] = {}
    for path in result_files:
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        for turn in record.get("turns", []):
            total_prompt_tokens += turn["usage"]["prompt_tokens"]
            total_completion_tokens += turn["usage"]["completion_tokens"]
            total_api_requests += 1
            dropped = turn.get("dropped_params") or []
            if dropped:
                turns_with_dropped_params += 1
            for name in dropped:
                dropped_params_counts[name] = dropped_params_counts.get(name, 0) + 1

    estimated_cost_usd = (
        total_prompt_tokens / 1000 * pricing["prompt"]
        + total_completion_tokens / 1000 * pricing["completion"]
    )

    return {
        "model_id": model_id,
        "total_calls": total_calls,
        "completed_calls": completed_calls,
        "failed_calls": total_calls - completed_calls,
        "total_api_requests": total_api_requests,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "estimated_cost_usd": round(estimated_cost_usd, 4),
        "turns_with_dropped_params": turns_with_dropped_params,
        "dropped_params_counts": dropped_params_counts,
        "start_time": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_time": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_seconds": round((end_time - start_time).total_seconds(), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="H3 실험 실행")
    parser.add_argument("--model", required=True, help="config 파일의 models에 등록된 라벨 (예: gpt4o_mini)")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"h3 실험 설정 파일 경로, 프로젝트 루트 기준 (기본: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument("--max-personas", type=int, default=None, help="페르소나 목록에서 앞에서 N명만 사용 (기본: 전체)")
    parser.add_argument("--max-reps", type=int, default=None, help="config의 repetitions 값을 무시하고 N회만 반복 (기본: config 값)")
    parser.add_argument("--concurrency", type=int, default=5, help="동시에 처리할 조합(combo) 스레드 수 (기본: 5)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="실제 API를 호출하지 않고 예상 호출 수/비용(상한 추정)만 계산해서 출력하고 종료",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = load_json(config_path)
    questions = load_json(QUESTIONS_PATH)
    stimuli = load_json(STIMULI_PATH)

    available_labels = sorted(config.get("models", {}).keys())
    if args.model not in available_labels:
        print(
            f"[오류] --model={args.model!r} 이 {config_path.name}의 models에 없습니다. "
            f"사용 가능한 라벨: {available_labels}",
            file=sys.stderr,
        )
        sys.exit(1)

    personas_file = config.get("personas_file")
    if not personas_file:
        print(f"[오류] {config_path.name}에 \"personas_file\" 필드가 없습니다.", file=sys.stderr)
        sys.exit(1)
    personas_data = load_json(PROJECT_ROOT / personas_file)

    topic = config["topic"]
    if topic not in questions or topic not in stimuli:
        print(f"[오류] topic={topic!r} 이 h3_questions.json / h3_stimuli.json 에 없습니다.", file=sys.stderr)
        sys.exit(1)
    topic_questions = questions[topic]
    topic_stimuli = stimuli[topic]

    # results_topic: 결과 저장 폴더명. 지정 안 하면 topic과 동일(기존 동작과 100% 동일).
    # 같은 topic 콘텐츠(질문/자극문)를 쓰면서 결과만 별도 폴더에 저장하고 싶을 때 사용
    # (예: 본 실험 medical_school_quota/ 를 건드리지 않는 spotcheck 실행).
    results_topic = config.get("results_topic", topic)

    model_config = config["models"][args.model]
    model_id = model_config["model_id"]
    generation_params = model_config["generation_params"]
    pricing = model_config["pricing_usd_per_1k_tokens"]

    question_types = config["question_types"]
    info_levels = config["info_levels"]
    session_types = config["session_types"]
    repetitions = args.max_reps if args.max_reps is not None else config["repetitions"]

    personas = personas_data["personas"]
    if args.max_personas is not None:
        personas = personas[: args.max_personas]

    print(f"페르소나 {len(personas)}명 로딩 중... (배치 조회)")
    persona_rows = fetch_personas_batch([p["persona_id"] for p in personas])

    if args.dry_run:
        report = dry_run_report(
            personas, question_types, info_levels, session_types, repetitions,
            topic_questions, topic_stimuli, persona_rows, model_id, generation_params, pricing,
        )
        print(f"\n=== DRY RUN ({args.model}) — 실제 API 호출 없음 ===")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    results_dir = RESULTS_ROOT / results_topic / args.model / "raw_responses"
    results_dir.mkdir(parents=True, exist_ok=True)

    combos = list(
        product(
            personas,
            question_types,
            info_levels,
            session_types,
            range(1, repetitions + 1),
        )
    )
    total_calls = len(combos)
    # 세션 유형별로 실제 API 호출(턴) 수가 다르므로(new_session=1, 나머지=2),
    # agent.py의 누적 비용 로그에 표시할 "전체 예상 호출 수"는 따로 계산한다.
    total_api_requests_estimate = sum(1 if s == "new_session" else 2 for *_, s, _ in combos)

    start_time = datetime.now(timezone.utc)
    failed_this_run = 0

    def process_combo(combo) -> tuple[str, bool, str | None]:
        """조합 1개를 처리해서 (파일명, 성공 여부, 에러 메시지) 반환.
        스레드에서 실행되며, 결과 파일 경로가 combo마다 고유하므로 쓰기 충돌은 없다."""
        persona, question_type, info_level, session_type, rep = combo
        persona_id = persona["persona_id"]
        filename = f"{persona_id}_{question_type}_{info_level}_{session_type}_{rep}.json"
        result_path = results_dir / filename

        if result_path.exists():
            return filename, True, None  # resume: 이미 완료된 결과는 건너뛰기

        persona_row = persona_rows[persona_id]
        stimulus_text = topic_stimuli[info_level]
        question_text = topic_questions[question_type]
        first_message = build_first_message(stimulus_text, question_text)

        try:
            turns = call_with_backoff(lambda: run_session(
                persona_row,
                topic_questions,
                question_type,
                model_id,
                generation_params,
                session_type,
                first_message,
                total_api_requests=total_api_requests_estimate,
            ))
        except (AgentAPIError, ValueError) as e:
            return filename, False, str(e)
        except Exception as e:  # 예상치 못한 오류도 전체 실행을 멈추지 않고 계속 진행
            return filename, False, f"예상치 못한 오류: {e}"

        record = {
            "persona_id": persona_id,
            "topic": topic,
            "question_type": question_type,
            "info_level": info_level,
            "session_type": session_type,
            "repetition": rep,
            "model_label": args.model,
            "model_id": model_id,
            "turns": turns,
        }
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        return filename, True, None

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(process_combo, combo) for combo in combos]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"H3 실험 ({args.model})"):
            filename, success, error = future.result()
            if not success:
                failed_this_run += 1
                tqdm.write(f"[실패] {filename}: {error}")

    end_time = datetime.now(timezone.utc)

    run_meta = compute_run_meta(model_id, results_dir, total_calls, pricing, start_time, end_time)
    run_meta["failed_calls_this_run"] = failed_this_run

    meta_path = RESULTS_ROOT / results_topic / args.model / "run_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    print(f"\n완료: {meta_path}")
    print(json.dumps(run_meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
