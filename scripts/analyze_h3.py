"""H3 실험 결과 분석 스크립트.

experiments/h3/results/{model_label}/raw_responses/*.json 을 모두 로드해서
태도 점수/핵심 근거/페르소나 속성 언급 여부를 파싱하고,
종속변수(초기 입장 일치율, 태도 변화량, 핵심 근거 유지율, 응답 방어성)를 계산한 뒤
CSV 3종과 시각화 4종을 experiments/h3/analysis/ 에 저장한다.

주의: 태도 점수·핵심 근거 추출은 OpenAI 호출 없이 키워드/문장 기반 휴리스틱으로 처리한다
(비용 절감 목적). 정밀도가 필요하면 이 부분을 LLM 기반 분류기로 교체하는 것을 권장한다.
임베딩(text-embedding-3-small)은 핵심 근거 유지율 계산에만 사용한다.

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "experiments" / "h3" / "results"
ANALYSIS_ROOT = PROJECT_ROOT / "experiments" / "h3" / "analysis"
PLOTS_DIR = ANALYSIS_ROOT / "plots"

EMBEDDING_MODEL = "text-embedding-3-small"
RETENTION_THRESHOLD = 0.8

INFO_LEVEL_ORDER = ["overview", "detailed", "with_counterarguments"]
QUESTION_TYPE_ORDER = ["original", "paraphrase", "pressure"]
# 공식 파라미터 수가 공개되어 있지 않아, 상대적인 크기 감을 위해 임의로 정한 순서.
# 실제 모델 크기와 다를 수 있으니 참고용으로만 사용.
MODEL_SIZE_ORDER = ["gpt4o_mini", "gpt41", "gpt4o"]

ATTITUDE_POSITIVE_KEYWORDS = [
    "찬성", "지지", "필요하다고 생각", "좋다고 생각", "긍정적", "환영", "도움이 될", "바람직", "동의"
]
ATTITUDE_NEGATIVE_KEYWORDS = [
    "반대", "우려", "문제가 있다고 생각", "부정적", "걱정", "위험", "부작용", "신중해야", "회의적"
]
HEDGE_WORDS = ["다만", "한편으로는", "한편", "그러나", "물론", "하지만", "비록", "반면"]
REASON_CONNECTIVES = ["왜냐하면", "때문", "니까", "이유는", "그 이유", "무엇보다", "특히"]


# ── 텍스트 파싱 휴리스틱 ─────────────────────────────────────────

def parse_attitude(text: str) -> dict:
    """찬성/반대/중립 키워드 빈도를 비교해 방향과 1~5점 점수를 추정 (휴리스틱)."""
    if not text:
        return {"direction": "중립", "score": 3}
    pos = sum(text.count(k) for k in ATTITUDE_POSITIVE_KEYWORDS)
    neg = sum(text.count(k) for k in ATTITUDE_NEGATIVE_KEYWORDS)
    if pos == neg:
        return {"direction": "중립", "score": 3}
    if pos > neg:
        return {"direction": "찬성", "score": min(5, 3 + (pos - neg))}
    return {"direction": "반대", "score": max(1, 3 - (neg - pos))}


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
            attitude = parse_attitude(text)
            reasons = extract_reasons(text)
            rows.append(
                {
                    "persona_id": record["persona_id"],
                    "model_label": record["model_label"],
                    "model_id": record["model_id"],
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
                    "before_text": record["turns"][0]["parsed_response"],
                    "after_text": record["turns"][1]["parsed_response"],
                }
            )

    return pairs


def compute_pair_metrics(pairs: list[dict]) -> pd.DataFrame:
    rows = []
    for pair in pairs:
        before_attitude = parse_attitude(pair["before_text"])
        after_attitude = parse_attitude(pair["after_text"])
        before_reasons = extract_reasons(pair["before_text"])
        after_reasons = extract_reasons(pair["after_text"])

        rows.append(
            {
                "comparison_type": pair["comparison_type"],
                "model_label": pair["model_label"],
                "info_level": pair["info_level"],
                "question_type": pair["question_type"],
                "concordant": before_attitude["direction"] == after_attitude["direction"],
                "attitude_change": abs(after_attitude["score"] - before_attitude["score"]),
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


# ── 메인 ─────────────────────────────────────────────────────────

def main() -> None:
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

    attitude_df = build_attitude_scores_df(records, persona_cache)
    attitude_df.to_csv(ANALYSIS_ROOT / "attitude_scores.csv", index=False)
    print(f"저장: {ANALYSIS_ROOT / 'attitude_scores.csv'} ({len(attitude_df)} rows)")

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
    print(f"저장: {PLOTS_DIR} 아래 PNG 4개")


if __name__ == "__main__":
    main()
