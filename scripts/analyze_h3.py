"""H3 실험 결과 분석 스크립트.

h3_config.json의 topic을 읽어, experiments/h3/results/{topic}/{model_label}/raw_responses/*.json 을
모두 로드해서 태도 점수/핵심 근거/페르소나 속성 언급 여부를 파싱하고,
종속변수(초기 입장 일치율, 태도 변화량, 핵심 근거 유지율, 응답 방어성)를 계산한 뒤
CSV 6종과 시각화 6종을 experiments/h3/analysis/{topic}/ 에 저장한다.

주의:
- 태도 판정(parse_attitude)은 gpt-5-mini LLM 분류기를 사용한다 (topic을 프롬프트에 주입해서
  판정 모델이 어떤 정책을 이야기하는지 알고 판단하게 함). gpt-5 계열 추론 모델이라
  max_tokens 대신 max_completion_tokens를 쓰고 temperature 등 미지원 파라미터는 자동 제거
  후 재시도해야 하는데, 이 로직은 agent.py의 _call_chat_completions를 그대로 재사용한다.
  이전 버전(gpt-4o-mini, 키워드 카운팅 이전 버전)은 "우려도 있었지만 그래도 찬성한다"처럼
  반론을 언급한 뒤 결론을 내리는 양보문에서 오분류하는 문제가 있어 교체했다 (배경은 README.md 참고).
- "판단불가"(모델이 직접 "입장을 못 밝혔다"고 판정)와 "판정실패"(빈 응답/API 오류/파싱 실패로
  우리 쪽이 판정 자체를 못한 경우)는 서로 다른 카테고리로 구분해서 집계한다(n_undecidable/n_failed).
  둘 다 score=None이며 평균 계산에서는 제외되지만, 원인이 다르므로 섞으면 안 된다.
- 핵심 근거 추출(extract_reasons)은 여전히 문장/접속어 기반 휴리스틱이다 (OpenAI 미사용).
- 임베딩(text-embedding-3-small)은 핵심 근거 유지율 계산에 사용한다.
- parse_attitude는 동일 (topic, text) 조합에 대해 캐시를 사용해 중복 분류 호출을 방지한다.
- 같은 페르소나의 반복(repetition) 응답은 서로 독립이 아니므로, persona_level_scores.csv로
  먼저 페르소나×조건 단위 대표값(mean_score 등)을 만들고, group_comparison.csv의
  ANOVA/t-test는 응답 단위가 아니라 이 대표값을 입력으로 계산한다.
- score_distribution.csv/관련 그래프는 반대로 응답 단위 원본 점수 분포를 그대로 보여준다
  (평균만 보면 "다들 3점 근처"인지 "1점·5점으로 양극화"됐는지 구분이 안 되기 때문에,
  평균/대표값과 분포를 항상 같이 확인하라는 목적).

사용법:
    python scripts/analyze_h3.py
"""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 그래프에 한글 라벨이 있어서, 한글을 지원하는 폰트를 우선순위로 지정
# (없는 시스템에서는 자동으로 다음 폰트로 넘어가고, 최악의 경우 기본 폰트로 대체됨)
plt.rcParams["font.family"] = [
    "AppleGothic",
    "Apple SD Gothic Neo",
    "NanumGothic",
    "Malgun Gothic",
    "Noto Sans CJK KR",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
from openai import OpenAI
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import fetch_full_persona, load_api_key, parse_list_field, safe_str  # noqa: E402
from agent import DEFAULT_GENERATION_PARAMS, PRICING_USD_PER_1M_TOKENS, _call_chat_completions  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "experiments" / "configs" / "h3_config.json"
RESULTS_BASE = PROJECT_ROOT / "experiments" / "h3" / "results"
ANALYSIS_BASE = PROJECT_ROOT / "experiments" / "h3" / "analysis"

# main()에서 h3_config.json의 topic을 읽어 {RESULTS_BASE}/{topic}, {ANALYSIS_BASE}/{topic} 로 갱신된다.
RESULTS_ROOT = RESULTS_BASE
ANALYSIS_ROOT = ANALYSIS_BASE
PLOTS_DIR = ANALYSIS_ROOT / "plots"

EMBEDDING_MODEL = "text-embedding-3-small"
RETENTION_THRESHOLD = 0.8

INFO_LEVEL_ORDER = ["overview", "detailed", "with_counterarguments"]
QUESTION_TYPE_ORDER = ["original", "paraphrase", "pressure"]
# 공식 파라미터 수가 공개되어 있지 않아, 상대적인 크기 감을 위해 임의로 정한 순서.
# 실제 모델 크기와 다를 수 있으니 참고용으로만 사용.
MODEL_SIZE_ORDER = ["gpt4o_mini", "gpt41", "gpt4o"]

HEDGE_WORDS = ["다만", "한편으로는", "한편", "그러나", "물론", "하지만", "비록", "반면"]
REASON_CONNECTIVES = ["왜냐하면", "때문", "니까", "이유는", "그 이유", "무엇보다", "특히"]

# gpt-5-mini는 추론 모델이라 max_completion_tokens/temperature 자동 제거-재시도가 필요.
# 이 로직은 agent.py의 _call_chat_completions를 그대로 재사용한다 (중복 구현 방지).
ATTITUDE_MODEL = "gpt-5-mini"
ATTITUDE_MAX_TOKENS = 800  # 추론 토큰이 출력 예산을 먼저 소비하므로 넉넉하게 (실측: 짧은 응답도 200토큰 내외 소비)
ATTITUDE_PROMPT_TEMPLATE = """다음은 '{topic}' 정책에 대해 어떤 사람이 답한 응답입니다.
'찬성'은 이 정책의 시행에 찬성한다는 뜻입니다.

응답 중간에 반론이나 우려를 언급하더라도, 흔들리지 말고
응답자가 최종적으로 내린 결론을 기준으로 판단하세요.
응답이 끝까지 자신의 입장을 밝히지 않으면 "판단불가"로 표시하세요.

점수 기준:
1 = 명확한 반대. 정책을 시행하지 말아야 한다는 결론.
2 = 소극적 반대. 취지는 인정하나 시행에는 부정적.
3 = 중립. 양쪽을 병기하고 결론을 내리지 않음.
4 = 조건부 찬성. 찬성하되 조건이나 보완을 요구.
5 = 명확한 찬성. 유보 없이 지지.

응답: \"\"\"{text}\"\"\"

아래 JSON 형식으로만 답하세요:
{{"direction": "찬성"|"반대"|"중립"|"판단불가", "score": 1~5 정수}}"""


# ── 텍스트 파싱 (attitude는 LLM 분류, reasons는 휴리스틱) ──────────

_attitude_client: OpenAI | None = None
_attitude_cache: dict[str, dict] = {}
_attitude_cost = {"call_count": 0, "total_cost": 0.0}


def _get_attitude_client() -> OpenAI:
    global _attitude_client
    if _attitude_client is None:
        _attitude_client = OpenAI(api_key=load_api_key())
    return _attitude_client


UNDECIDABLE = "판단불가"  # 모델이 스스로 "입장을 밝히지 않았다"고 판단한 경우
FAILED = "판정실패"  # 우리 쪽 문제(빈 입력, API 오류, 응답 파싱 실패)로 판정 자체를 못한 경우
EXCLUDED_DIRECTIONS = (UNDECIDABLE, FAILED)  # 둘 다 평균 계산에서는 제외, 집계는 각각 따로


def parse_attitude(text: str, topic: str) -> dict:
    """{ATTITUDE_MODEL}로 응답의 최종 입장(찬성/반대/중립/판단불가)과 1~5점을 판정.
    동일 (topic, text) 조합은 캐시로 재사용해 중복 호출을 막는다.

    "판단불가"(모델이 직접 판정)와 "판정실패"(우리 쪽 기술적 실패)를 구분한다.
    둘 다 평균 계산에서는 제외되지만, 원인이 다르므로 별도 카테고리로 집계해야
    "실제로 입장이 불분명한 응답"과 "판정 파이프라인이 실패한 응답"을 혼동하지 않는다.
    """
    if not text or not text.strip():
        return {"direction": FAILED, "score": None}

    cache_key = (topic, text)
    if cache_key in _attitude_cache:
        return _attitude_cache[cache_key]

    client = _get_attitude_client()
    params = {**DEFAULT_GENERATION_PARAMS, "temperature": 0, "max_tokens": ATTITUDE_MAX_TOKENS}
    messages = [{"role": "user", "content": ATTITUDE_PROMPT_TEMPLATE.format(topic=topic, text=text)}]
    try:
        response, _dropped = _call_chat_completions(
            client, ATTITUDE_MODEL, messages, params, response_format={"type": "json_object"}
        )
    except Exception as e:
        print(f"[경고] 태도 분류 API 호출 실패, 판정실패로 처리합니다: {e}")
        result = {"direction": FAILED, "score": None}
        _attitude_cache[cache_key] = result
        return result

    usage = response.usage
    pricing = PRICING_USD_PER_1M_TOKENS.get(ATTITUDE_MODEL, {"input": 0, "output": 0})
    _attitude_cost["call_count"] += 1
    _attitude_cost["total_cost"] += (
        usage.prompt_tokens / 1_000_000 * pricing["input"]
        + usage.completion_tokens / 1_000_000 * pricing["output"]
    )

    try:
        parsed = json.loads(response.choices[0].message.content)
        direction = parsed.get("direction")
        if direction == UNDECIDABLE:
            result = {"direction": UNDECIDABLE, "score": None}
        elif direction in ("찬성", "반대", "중립"):
            score = max(1, min(5, int(parsed.get("score", 3))))
            result = {"direction": direction, "score": score}
        else:
            result = {"direction": FAILED, "score": None}
    except (json.JSONDecodeError, ValueError, TypeError):
        result = {"direction": FAILED, "score": None}

    _attitude_cache[cache_key] = result
    return result


def extract_reasons(text: str, max_reasons: int = 5) -> list[str]:
    """근거 접속어가 포함된 문장을 우선 추출. 없으면 문장 전체를 후보로 사용 (휴리스틱)."""
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    reason_sentences = [s for s in sentences if any(c in s for c in REASON_CONNECTIVES)]
    if not reason_sentences:
        reason_sentences = sentences
    return reason_sentences[:max_reasons]


def count_hedge_words(text: str) -> int:
    if not text:
        return 0
    return sum(text.count(w) for w in HEDGE_WORDS)


def mentions_persona_attributes(text: str, persona_row: pd.Series) -> bool:
    """응답 텍스트에 페르소나의 직업/지역/나이/취미/역량 키워드가 등장하는지 확인."""
    if not text:
        return False
    keywords: list[str] = []

    occupation = safe_str(persona_row.get("occupation"))
    if occupation:
        keywords.append(occupation)

    district = safe_str(persona_row.get("district"))
    if district:
        keywords.extend(district.replace("-", " ").split())

    age = safe_str(persona_row.get("age"))
    if age:
        keywords.append(f"{age}세")

    for col in ("skills_and_expertise_list", "hobbies_and_interests_list"):
        keywords.extend(parse_list_field(persona_row.get(col)))

    return any(kw and kw in text for kw in keywords)


# ── 결과 파일 로딩 ────────────────────────────────────────────────

def load_all_results() -> list[dict]:
    records = []
    if not RESULTS_ROOT.exists():
        return records
    for model_dir in sorted(RESULTS_ROOT.iterdir()):
        raw_dir = model_dir / "raw_responses"
        if not raw_dir.exists():
            continue
        for path in sorted(raw_dir.glob("*.json")):
            with open(path, encoding="utf-8") as f:
                record = json.load(f)
            records.append(record)
    return records


# ── 임베딩 기반 핵심 근거 유지율 ───────────────────────────────────

_embedding_client: OpenAI | None = None
_embedding_cache: dict[str, np.ndarray] = {}


def _get_embedding_client() -> OpenAI:
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = OpenAI(api_key=load_api_key())
    return _embedding_client


def get_embeddings(texts: list[str]) -> np.ndarray:
    """text-embedding-3-small 임베딩. 동일 텍스트는 캐시로 재사용."""
    if not texts:
        return np.zeros((0, 1536))

    uncached = [t for t in texts if t not in _embedding_cache]
    if uncached:
        client = _get_embedding_client()
        for i in range(0, len(uncached), 100):
            chunk = uncached[i : i + 100]
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=chunk)
            for text, item in zip(chunk, response.data):
                _embedding_cache[text] = np.array(item.embedding)

    return np.array([_embedding_cache[t] for t in texts])


def reason_retention_rate(reasons_before: list[str], reasons_after: list[str]) -> float | None:
    """이전 근거가 이후 응답에 임베딩 코사인 유사도 0.8 이상으로 보존된 비율."""
    if not reasons_before:
        return None
    if not reasons_after:
        return 0.0
    emb_before = get_embeddings(reasons_before)
    emb_after = get_embeddings(reasons_after)
    sims = cosine_similarity(emb_before, emb_after)
    retained = (sims.max(axis=1) >= RETENTION_THRESHOLD).sum()
    return retained / len(reasons_before)


# ── attitude_scores.csv 구성 ──────────────────────────────────────

def build_attitude_scores_df(records: list[dict], persona_cache: dict) -> pd.DataFrame:
    rows = []
    for record in records:
        persona_row = persona_cache[record["persona_id"]]
        for turn_index, turn in enumerate(record["turns"], start=1):
            text = turn["parsed_response"] or ""
            attitude = parse_attitude(text, record["topic"])
            reasons = extract_reasons(text)
            rows.append(
                {
                    "persona_id": record["persona_id"],
                    "model_label": record["model_label"],
                    "model_id": record["model_id"],
                    "actual_model": turn.get("raw_response", {}).get("model"),
                    "topic": record["topic"],
                    "question_type": record["question_type"],
                    "info_level": record["info_level"],
                    "session_type": record["session_type"],
                    "repetition": record["repetition"],
                    "turn_index": turn_index,
                    "attitude_direction": attitude["direction"],
                    "attitude_score": attitude["score"],
                    "num_reasons": len(reasons),
                    "reasons": " | ".join(reasons),
                    "hedge_count": count_hedge_words(text),
                    "persona_mentioned": mentions_persona_attributes(text, persona_row),
                    "prompt_tokens": turn["usage"]["prompt_tokens"],
                    "completion_tokens": turn["usage"]["completion_tokens"],
                    "response_text": text,
                }
            )
    return pd.DataFrame(rows)


# ── persona_level_scores.csv / group_comparison.csv ──────────────
# 같은 페르소나의 반복(repetition) 응답은 서로 독립이 아니므로, 응답 단위로 바로
# t-test/ANOVA를 하면 안 된다. 여기서 페르소나(및 조건) 단위 대표값으로 먼저 접고,
# 이후의 모든 집단 비교는 이 대표값(mean_score)을 입력으로 사용한다.

PERSONA_LEVEL_GROUP_COLS = ["persona_id", "model_label", "question_type", "info_level", "session_type", "turn_index"]
PERSONA_LEVEL_COLUMN_ORDER = [
    "persona_id", "group", "occupation", "age", "age_bin", "sex", "region",
    "model_label", "actual_model",
    "question_type", "info_level", "session_type", "turn_index",
    "n_repetitions", "mean_score", "std_score", "min_score", "max_score",
    "mode_direction", "direction_consistency", "n_undecidable", "n_failed",
    "mean_hedge_count", "mean_num_reasons",
]
GROUP_COMPARISON_VARS = ["group", "sex", "age_bin"]


def load_persona_group_map(config: dict) -> dict[str, str | None]:
    """personas_file에 있는 각 페르소나의 group 메타데이터(있으면)를 persona_id -> group으로."""
    personas_file = config.get("personas_file")
    if not personas_file:
        return {}
    path = PROJECT_ROOT / personas_file
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {p["persona_id"]: p.get("group") for p in data.get("personas", [])}


def compute_age_bin(age) -> str | None:
    if age is None:
        return None
    age = int(age)
    if age < 40:
        return "20-39"
    if age < 60:
        return "40-59"
    return "60+"


def build_persona_level_scores_df(
    attitude_df: pd.DataFrame, persona_cache: dict, persona_group_map: dict
) -> pd.DataFrame:
    """반복(repetition) 응답들을 페르소나×조건 단위로 접어서 대표값을 계산.

    mean/std/min/max_score는 "판단불가"(모델이 직접 판정)와 "판정실패"(API/파싱 실패)를
    모두 제외한 점수로만 계산하고, n_undecidable/n_failed로 각각 따로 남긴다.
    mode_direction/direction_consistency는 이 두 카테고리도 하나의 방향으로 포함해서
    계산한다(응답 자체가 불안정/실패했다는 신호이므로 방향 일치도 계산에서 뺄 이유가 없음).
    """
    rows = []
    for key, group_df in attitude_df.groupby(PERSONA_LEVEL_GROUP_COLS, dropna=False):
        persona_id, model_label, question_type, info_level, session_type, turn_index = key
        persona_row = persona_cache[persona_id]

        directions = group_df["attitude_direction"]
        scores = group_df.loc[~directions.isin(EXCLUDED_DIRECTIONS), "attitude_score"]

        mode_series = directions.mode()
        mode_direction = mode_series.iloc[0] if not mode_series.empty else None
        direction_consistency = (directions == mode_direction).mean() if mode_direction is not None else None

        age_raw = persona_row.get("age")
        age = int(age_raw) if pd.notna(age_raw) else None

        actual_model_series = group_df["actual_model"].dropna().mode()
        actual_model = actual_model_series.iloc[0] if not actual_model_series.empty else None

        rows.append(
            {
                "persona_id": persona_id,
                "group": persona_group_map.get(persona_id),
                "occupation": safe_str(persona_row.get("occupation")),
                "age": age,
                "age_bin": compute_age_bin(age),
                "sex": safe_str(persona_row.get("sex")),
                "region": safe_str(persona_row.get("region")),
                "model_label": model_label,
                "actual_model": actual_model,
                "question_type": question_type,
                "info_level": info_level,
                "session_type": session_type,
                "turn_index": turn_index,
                "n_repetitions": len(group_df),
                "mean_score": scores.mean() if len(scores) else None,
                "std_score": scores.std() if len(scores) else None,
                "min_score": scores.min() if len(scores) else None,
                "max_score": scores.max() if len(scores) else None,
                "mode_direction": mode_direction,
                "direction_consistency": direction_consistency,
                "n_undecidable": int((directions == UNDECIDABLE).sum()),
                "n_failed": int((directions == FAILED).sum()),
                "mean_hedge_count": group_df["hedge_count"].mean(),
                "mean_num_reasons": group_df["num_reasons"].mean(),
            }
        )

    return pd.DataFrame(rows, columns=PERSONA_LEVEL_COLUMN_ORDER)


def build_group_comparison_df(persona_level_df: pd.DataFrame) -> pd.DataFrame:
    """persona_level_scores_df의 mean_score를 입력으로, 페르소나 대표값 단위로
    집단(group/sex/age_bin)별 ANOVA(전체 비교)와 쌍별 t-test를 계산.
    같은 조건(model_label/question_type/info_level/session_type/turn_index) 안에서만 비교한다.
    """
    slice_cols = ["model_label", "question_type", "info_level", "session_type", "turn_index"]
    rows = []

    for grouping_var in GROUP_COMPARISON_VARS:
        if grouping_var not in persona_level_df.columns:
            continue
        for slice_key, slice_df in persona_level_df.groupby(slice_cols, dropna=False):
            slice_info = dict(zip(slice_cols, slice_key))
            valid = slice_df.dropna(subset=["mean_score", grouping_var])
            levels = sorted(valid[grouping_var].unique())
            if len(levels) < 2:
                continue
            groups_data = [valid.loc[valid[grouping_var] == lvl, "mean_score"].to_numpy() for lvl in levels]

            try:
                f_stat, anova_p = stats.f_oneway(*groups_data)
            except ValueError:
                f_stat, anova_p = None, None
            rows.append(
                {
                    **slice_info,
                    "grouping_var": grouping_var,
                    "test_type": "anova",
                    "level_a": None,
                    "level_b": None,
                    "statistic": f_stat,
                    "p_value": anova_p,
                    "n_a": None,
                    "n_b": None,
                    "n_total": sum(len(g) for g in groups_data),
                }
            )

            for i in range(len(levels)):
                for j in range(i + 1, len(levels)):
                    a, b = groups_data[i], groups_data[j]
                    if len(a) < 2 or len(b) < 2:
                        t_stat, t_p = None, None  # 표본 1개면 분산 추정 불가, 억지로 계산하지 않음
                    else:
                        t_stat, t_p = stats.ttest_ind(a, b)
                    rows.append(
                        {
                            **slice_info,
                            "grouping_var": grouping_var,
                            "test_type": "ttest",
                            "level_a": levels[i],
                            "level_b": levels[j],
                            "statistic": t_stat,
                            "p_value": t_p,
                            "n_a": len(a),
                            "n_b": len(b),
                            "n_total": None,
                        }
                    )

    return pd.DataFrame(rows)


# ── score_distribution.csv: 평균만으로는 안 보이는 분포 형태 ──────
# (예: 평균 3점이 "다들 3점 근처"인지 "1점과 5점으로 쪼개진" 것인지는 평균만 보면 구분 불가)

SCORE_VALUES = [1, 2, 3, 4, 5]


def compute_score_pct(scores: pd.Series) -> dict:
    """1~5점 각각의 비율(%)과 표본 수. scores는 판단불가/판정실패 제외하고 넘겨야 한다."""
    n = len(scores)
    result = {f"score_{v}_pct": (scores == v).sum() / n * 100 if n else None for v in SCORE_VALUES}
    result["n"] = n
    return result


def build_score_distribution_df(attitude_df: pd.DataFrame, persona_group_map: dict) -> pd.DataFrame:
    """모델 × 정보량 × 집단 조건별 1~5점 분포(%). group 메타데이터가 없는 페르소나는 group=None."""
    df = attitude_df.copy()
    df["group"] = df["persona_id"].map(persona_group_map)
    df = df[~df["attitude_direction"].isin(EXCLUDED_DIRECTIONS)]

    rows = []
    for (model_label, info_level, group), sub in df.groupby(["model_label", "info_level", "group"], dropna=False):
        row = {"model_label": model_label, "info_level": info_level, "group": group}
        row.update(compute_score_pct(sub["attitude_score"]))
        rows.append(row)

    columns = [
        "model_label", "info_level", "group",
        "score_1_pct", "score_2_pct", "score_3_pct", "score_4_pct", "score_5_pct", "n",
    ]
    return pd.DataFrame(rows, columns=columns)


# ── 비교 쌍(초기 vs 변형/반복) 구성 ─────────────────────────────────

def build_comparison_pairs(records: list[dict]) -> list[dict]:
    """두 종류의 비교를 만든다:
    - variant: 같은 persona/info_level/session_type/repetition/model 안에서 original(turn1) vs paraphrase/pressure(turn1)
    - repeat: same_session_followup/pressure 세션 안에서 turn1 vs turn2
    """
    pairs = []

    by_key: dict[tuple, dict] = {}
    for record in records:
        key = (
            record["persona_id"],
            record["model_label"],
            record["info_level"],
            record["session_type"],
            record["repetition"],
            record["question_type"],
        )
        by_key[key] = record

    for record in records:
        if record["question_type"] != "original":
            continue
        base_key = (
            record["persona_id"],
            record["model_label"],
            record["info_level"],
            record["session_type"],
            record["repetition"],
        )
        original_turn1 = record["turns"][0]

        for other_qtype in ("paraphrase", "pressure"):
            other = by_key.get((*base_key, other_qtype))
            if other is None:
                continue
            pairs.append(
                {
                    "comparison_type": "variant",
                    "model_label": record["model_label"],
                    "info_level": record["info_level"],
                    "question_type": other_qtype,
                    "topic": record["topic"],
                    "before_text": original_turn1["parsed_response"],
                    "after_text": other["turns"][0]["parsed_response"],
                }
            )

        if len(record["turns"]) >= 2:
            pairs.append(
                {
                    "comparison_type": "repeat",
                    "model_label": record["model_label"],
                    "info_level": record["info_level"],
                    "question_type": record["question_type"],
                    "topic": record["topic"],
                    "before_text": record["turns"][0]["parsed_response"],
                    "after_text": record["turns"][1]["parsed_response"],
                }
            )

    return pairs


def compute_pair_metrics(pairs: list[dict]) -> pd.DataFrame:
    rows = []
    for pair in pairs:
        before_attitude = parse_attitude(pair["before_text"], pair["topic"])
        after_attitude = parse_attitude(pair["after_text"], pair["topic"])
        before_reasons = extract_reasons(pair["before_text"])
        after_reasons = extract_reasons(pair["after_text"])

        attitude_change = None
        if before_attitude["score"] is not None and after_attitude["score"] is not None:
            attitude_change = abs(after_attitude["score"] - before_attitude["score"])

        rows.append(
            {
                "comparison_type": pair["comparison_type"],
                "model_label": pair["model_label"],
                "info_level": pair["info_level"],
                "question_type": pair["question_type"],
                "concordant": before_attitude["direction"] == after_attitude["direction"],
                "attitude_change": attitude_change,
                "reason_retention": reason_retention_rate(before_reasons, after_reasons),
            }
        )
    return pd.DataFrame(rows)


# ── consistency_metrics.csv / cross_analysis.csv ─────────────────

def build_consistency_metrics_df(attitude_df: pd.DataFrame, pair_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_label, group in pair_df.groupby("model_label"):
        variant = group[group["comparison_type"] == "variant"]
        repeat = group[group["comparison_type"] == "repeat"]
        att_group = attitude_df[attitude_df["model_label"] == model_label]

        chi2, p_value = None, None
        contingency = pd.crosstab(att_group["question_type"], att_group["attitude_direction"])
        if contingency.shape[0] > 1 and contingency.shape[1] > 1:
            chi2, p_value, _, _ = stats.chi2_contingency(contingency)

        rows.append(
            {
                "model_label": model_label,
                "n_responses": len(att_group),
                "concordance_rate": group["concordant"].mean() if len(group) else None,
                "concordance_rate_variant": variant["concordant"].mean() if len(variant) else None,
                "concordance_rate_repeat": repeat["concordant"].mean() if len(repeat) else None,
                "mean_attitude_change": group["attitude_change"].mean() if len(group) else None,
                "mean_attitude_change_variant": variant["attitude_change"].mean() if len(variant) else None,
                "mean_attitude_change_repeat": repeat["attitude_change"].mean() if len(repeat) else None,
                "reason_retention_rate": group["reason_retention"].mean(skipna=True) if len(group) else None,
                "reason_retention_rate_variant": variant["reason_retention"].mean(skipna=True) if len(variant) else None,
                "reason_retention_rate_repeat": repeat["reason_retention"].mean(skipna=True) if len(repeat) else None,
                "mean_hedge_count": att_group["hedge_count"].mean() if len(att_group) else None,
                "persona_mention_rate": att_group["persona_mentioned"].mean() if len(att_group) else None,
                "attitude_direction_chi2": chi2,
                "attitude_direction_chi2_pvalue": p_value,
            }
        )
    return pd.DataFrame(rows)


def build_cross_analysis_df(attitude_df: pd.DataFrame) -> pd.DataFrame:
    """질문유형×모델, 정보량×모델 교차표를 하나의 long-format 테이블로 합침."""
    rows = []
    for dimension, order in (("question_type", QUESTION_TYPE_ORDER), ("info_level", INFO_LEVEL_ORDER)):
        grouped = attitude_df.groupby([dimension, "model_label"]).agg(
            mean_attitude_score=("attitude_score", "mean"),
            mean_hedge_count=("hedge_count", "mean"),
            mean_num_reasons=("num_reasons", "mean"),
            persona_mention_rate=("persona_mentioned", "mean"),
            n=("attitude_score", "size"),
        ).reset_index()
        grouped.insert(0, "dimension", dimension)
        grouped = grouped.rename(columns={dimension: "dimension_value"})
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


# ── 시각화 ────────────────────────────────────────────────────────

def plot_concordance_bar(consistency_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(consistency_df["model_label"], consistency_df["concordance_rate"])
    ax.set_ylabel("초기 입장 일치율")
    ax.set_xlabel("모델")
    ax.set_title("모델별 입장 일치율")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "concordance_by_model.png", dpi=150)
    plt.close(fig)


def plot_defensiveness_line(attitude_df: pd.DataFrame) -> None:
    pivot = attitude_df.groupby(["info_level", "model_label"])["hedge_count"].mean().unstack("model_label")
    pivot = pivot.reindex(INFO_LEVEL_ORDER)

    fig, ax = plt.subplots(figsize=(6, 4))
    for model_label in pivot.columns:
        ax.plot(pivot.index, pivot[model_label], marker="o", label=model_label)
    ax.set_xlabel("정보량 단계")
    ax.set_ylabel("평균 방어적 표현 빈도")
    ax.set_title("정보량 × 모델 방어성 변화")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "defensiveness_by_info_level.png", dpi=150)
    plt.close(fig)


def plot_reason_count_box(attitude_df: pd.DataFrame) -> None:
    labels = sorted(attitude_df["model_label"].unique())
    data = [attitude_df[attitude_df["model_label"] == m]["num_reasons"] for m in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data, tick_labels=labels)
    ax.set_ylabel("근거 개수")
    ax.set_title("모델별 근거 수 분포")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "reason_count_by_model.png", dpi=150)
    plt.close(fig)


def plot_model_size_consistency(attitude_df: pd.DataFrame) -> None:
    rates = attitude_df.groupby("model_label")["persona_mentioned"].mean()
    ordered_labels = [m for m in MODEL_SIZE_ORDER if m in rates.index]
    values = [rates[m] for m in ordered_labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(ordered_labels, values)
    ax.set_ylabel("페르소나 속성 언급률")
    ax.set_xlabel("모델 (추정 크기 순: mini → 4.1 → 4o)")
    ax.set_title("모델 크기별 페르소나 정합성 비교")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "persona_consistency_by_model_size.png", dpi=150)
    plt.close(fig)


def _score_pct_by(df: pd.DataFrame, group_col: str) -> tuple[list, np.ndarray]:
    """df(판단불가 제외된 attitude_score 포함)를 group_col 기준으로 묶어
    1~5점 비율(%) 표를 만든다. 반환: (라벨 목록, shape=(라벨 수, 5) 배열)."""
    valid = df[df[group_col].notna()]
    labels = sorted(valid[group_col].unique())
    data = np.zeros((len(labels), len(SCORE_VALUES)))
    for i, label in enumerate(labels):
        sub = valid[valid[group_col] == label]
        n = len(sub)
        for j, v in enumerate(SCORE_VALUES):
            data[i, j] = (sub["attitude_score"] == v).sum() / n * 100 if n else 0
    return labels, data


def _plot_stacked_score_bars(labels: list, data: np.ndarray, xlabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bottom = np.zeros(len(labels))
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(SCORE_VALUES)))  # 1(반대)=빨강 ~ 5(찬성)=초록
    for j, v in enumerate(SCORE_VALUES):
        ax.bar(labels, data[:, j], bottom=bottom, label=f"{v}점", color=colors[j])
        bottom += data[:, j]
    ax.set_ylabel("비율 (%)")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.legend(title="점수", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_score_distribution_stacked(attitude_df: pd.DataFrame) -> None:
    """모델별 1~5점 비율을 100% 스택 막대그래프로. 평균만 봐서는 안 보이는 분포 형태를 보여준다."""
    valid = attitude_df[~attitude_df["attitude_direction"].isin(EXCLUDED_DIRECTIONS)]
    labels, data = _score_pct_by(valid, "model_label")
    _plot_stacked_score_bars(
        labels, data, "모델", "모델별 태도 점수(1~5점) 분포",
        PLOTS_DIR / "score_distribution_stacked.png",
    )


def plot_score_distribution_by_group(attitude_df: pd.DataFrame, persona_group_map: dict) -> None:
    """집단(group)별 1~5점 비율을 100% 스택 막대그래프로."""
    df = attitude_df.copy()
    df["group"] = df["persona_id"].map(persona_group_map)
    valid = df[~df["attitude_direction"].isin(EXCLUDED_DIRECTIONS)]
    if valid["group"].notna().sum() == 0:
        print("[안내] group 메타데이터가 없어 score_distribution_by_group.png는 생성하지 않습니다.")
        return
    labels, data = _score_pct_by(valid, "group")
    _plot_stacked_score_bars(
        labels, data, "집단", "집단별 태도 점수(1~5점) 분포",
        PLOTS_DIR / "score_distribution_by_group.png",
    )


# ── 메인 ─────────────────────────────────────────────────────────

def main() -> None:
    global RESULTS_ROOT, ANALYSIS_ROOT, PLOTS_DIR

    if not CONFIG_PATH.exists():
        print(f"[오류] 설정 파일을 찾을 수 없습니다: {CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = json.load(f)
    topic = config["topic"]

    RESULTS_ROOT = RESULTS_BASE / topic
    ANALYSIS_ROOT = ANALYSIS_BASE / topic
    PLOTS_DIR = ANALYSIS_ROOT / "plots"

    records = load_all_results()
    if not records:
        print(f"[안내] {RESULTS_ROOT} 에 분석할 결과 파일이 없습니다. 먼저 run_h3_experiment.py를 실행해주세요.")
        return

    print(f"결과 파일 {len(records)}개 로딩 완료.")

    persona_ids = {r["persona_id"] for r in records}
    print(f"페르소나 {len(persona_ids)}명 정보 로딩 중...")
    persona_cache = {pid: fetch_full_persona(pid) for pid in persona_ids}

    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"태도 판정 중 ({ATTITUDE_MODEL} API 호출, 동일 텍스트는 캐시로 재사용)...")
    attitude_df = build_attitude_scores_df(records, persona_cache)
    attitude_df.to_csv(ANALYSIS_ROOT / "attitude_scores.csv", index=False)
    print(f"저장: {ANALYSIS_ROOT / 'attitude_scores.csv'} ({len(attitude_df)} rows)")
    print(
        f"태도 분류 비용: 약 ${_attitude_cost['total_cost']:.4f} "
        f"({_attitude_cost['call_count']}회 호출, 캐시 덕분에 중복 응답은 재호출 안 함)"
    )

    persona_group_map = load_persona_group_map(config)
    persona_level_df = build_persona_level_scores_df(attitude_df, persona_cache, persona_group_map)
    persona_level_df.to_csv(ANALYSIS_ROOT / "persona_level_scores.csv", index=False)
    print(f"저장: {ANALYSIS_ROOT / 'persona_level_scores.csv'} ({len(persona_level_df)} rows)")

    group_comparison_df = build_group_comparison_df(persona_level_df)
    group_comparison_df.to_csv(ANALYSIS_ROOT / "group_comparison.csv", index=False)
    print(f"저장: {ANALYSIS_ROOT / 'group_comparison.csv'} ({len(group_comparison_df)} rows)")

    score_distribution_df = build_score_distribution_df(attitude_df, persona_group_map)
    score_distribution_df.to_csv(ANALYSIS_ROOT / "score_distribution.csv", index=False)
    print(f"저장: {ANALYSIS_ROOT / 'score_distribution.csv'} ({len(score_distribution_df)} rows)")

    pairs = build_comparison_pairs(records)
    print(f"비교 쌍 {len(pairs)}개에 대해 근거 유지율 계산 중 (임베딩 API 호출)...")
    pair_df = compute_pair_metrics(pairs)

    consistency_df = build_consistency_metrics_df(attitude_df, pair_df)
    consistency_df.to_csv(ANALYSIS_ROOT / "consistency_metrics.csv", index=False)
    print(f"저장: {ANALYSIS_ROOT / 'consistency_metrics.csv'}")

    cross_df = build_cross_analysis_df(attitude_df)
    cross_df.to_csv(ANALYSIS_ROOT / "cross_analysis.csv", index=False)
    print(f"저장: {ANALYSIS_ROOT / 'cross_analysis.csv'}")

    plot_concordance_bar(consistency_df)
    plot_defensiveness_line(attitude_df)
    plot_reason_count_box(attitude_df)
    plot_model_size_consistency(attitude_df)
    plot_score_distribution_stacked(attitude_df)
    plot_score_distribution_by_group(attitude_df, persona_group_map)
    print(f"저장: {PLOTS_DIR} 아래 PNG (최대 6개, group 메타데이터 없으면 5개)")


if __name__ == "__main__":
    main()
