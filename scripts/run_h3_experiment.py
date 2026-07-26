"""H3 실험 실행 스크립트.

experiments/configs/h3_config.json 의 설정으로,
config의 "personas_file"이 가리키는 JSON에 담긴 페르소나들에 대해
질문유형 3가지 x 정보량 3단계 x 세션유형 3가지 x repetitions회를 실행하고,
각 결과를 experiments/h3/results/{topic}/{model_label}/raw_responses/ 에 저장한다.

사용법:
    python scripts/run_h3_experiment.py --model gpt4o_mini
    python scripts/run_h3_experiment.py --model gpt4o_mini --max-personas 2 --max-reps 1  # 소규모 테스트용
    python scripts/run_h3_experiment.py --config experiments/configs/h3_config_spotcheck.json --model gpt56nano
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import ask_persona, AgentAPIError  # noqa: E402
from main import fetch_full_persona  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = "experiments/configs/h3_config.json"
QUESTIONS_PATH = PROJECT_ROOT / "experiments" / "configs" / "h3_questions.json"
STIMULI_PATH = PROJECT_ROOT / "experiments" / "configs" / "h3_stimuli.json"
RESULTS_ROOT = PROJECT_ROOT / "experiments" / "h3" / "results"


def load_json(path: Path) -> dict:
    if not path.exists():
        print(f"[오류] 파일을 찾을 수 없습니다: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_first_message(stimulus_text: str, question_text: str) -> str:
    return f"{stimulus_text}\n\n{question_text}"


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
    for path in result_files:
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        for turn in record.get("turns", []):
            total_prompt_tokens += turn["usage"]["prompt_tokens"]
            total_completion_tokens += turn["usage"]["completion_tokens"]
            total_api_requests += 1

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

    results_dir = RESULTS_ROOT / results_topic / args.model / "raw_responses"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"페르소나 {len(personas)}명 로딩 중...")
    persona_rows = {p["persona_id"]: fetch_full_persona(p["persona_id"]) for p in personas}

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

    for persona, question_type, info_level, session_type, rep in tqdm(combos, desc=f"H3 실험 ({args.model})"):
        persona_id = persona["persona_id"]
        filename = f"{persona_id}_{question_type}_{info_level}_{session_type}_{rep}.json"
        result_path = results_dir / filename

        if result_path.exists():
            continue  # resume: 이미 완료된 결과는 건너뛰기

        persona_row = persona_rows[persona_id]
        stimulus_text = topic_stimuli[info_level]
        question_text = topic_questions[question_type]
        first_message = build_first_message(stimulus_text, question_text)

        try:
            turns = run_session(
                persona_row,
                topic_questions,
                question_type,
                model_id,
                generation_params,
                session_type,
                first_message,
                total_api_requests=total_api_requests_estimate,
            )
        except (AgentAPIError, ValueError) as e:
            failed_this_run += 1
            tqdm.write(f"[실패] {filename}: {e}")
            continue
        except Exception as e:  # 예상치 못한 오류도 전체 실행을 멈추지 않고 계속 진행
            failed_this_run += 1
            tqdm.write(f"[예상치 못한 오류] {filename}: {e}")
            continue

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
