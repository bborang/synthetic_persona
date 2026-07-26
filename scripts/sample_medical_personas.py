"""의대 정원 확대 주제용 페르소나 샘플링 스크립트.

ko_KR.parquet에서 지역(서울/지방) x 연령(청년/고령) 2x2 집단별로 2명씩,
총 8명을 뽑아 experiments/h3/sampled_personas_medical.json으로 저장한다.
집단 안에 의료 관련 직업(의사/간호사/약사/의료/병원) 종사자가 있으면
그 중 최소 1명은 우선 포함시킨다 (없으면 그냥 무작위 2명).

사용법:
    python scripts/sample_medical_personas.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import LIGHT_COLUMNS, PARQUET_PATH, get_full_name  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "experiments" / "h3" / "sampled_personas_medical.json"

SEED = 42
N_PER_GROUP = 2
MEDICAL_KEYWORDS = ["의사", "간호사", "약사", "의료", "병원"]

GROUPS = [
    ("서울_청년", lambda df: df["region"].str.contains("서울", na=False) & df["age"].between(20, 29)),
    ("서울_고령", lambda df: df["region"].str.contains("서울", na=False) & (df["age"] >= 55)),
    ("지방_청년", lambda df: ~df["region"].str.contains("서울", na=False) & df["age"].between(20, 29)),
    ("지방_고령", lambda df: ~df["region"].str.contains("서울", na=False) & (df["age"] >= 55)),
]


def sample_group(subset: pd.DataFrame, group_label: str, rng: np.random.Generator) -> pd.DataFrame:
    if subset.empty:
        print(f"[경고] {group_label}: 조건에 맞는 페르소나가 없습니다.")
        return subset

    medical_mask = subset["occupation"].str.contains("|".join(MEDICAL_KEYWORDS), na=False)
    medical_candidates = subset[medical_mask]

    picked_indices: list = []
    if not medical_candidates.empty:
        idx = rng.choice(medical_candidates.index.to_numpy(), size=1, replace=False)
        picked_indices.extend(idx.tolist())

    remaining = subset.drop(index=picked_indices)
    n_more = min(N_PER_GROUP - len(picked_indices), len(remaining))
    if n_more > 0:
        more_idx = rng.choice(remaining.index.to_numpy(), size=n_more, replace=False)
        picked_indices.extend(more_idx.tolist())

    if len(picked_indices) < N_PER_GROUP:
        print(f"[경고] {group_label}: 조건에 맞는 인원이 {len(picked_indices)}명뿐입니다 (요청 {N_PER_GROUP}명).")

    return subset.loc[picked_indices]


def main() -> None:
    print(f"페르소나 인덱스 로딩 중... ({PARQUET_PATH.name})")
    df = pd.read_parquet(PARQUET_PATH, columns=LIGHT_COLUMNS)

    rng = np.random.default_rng(SEED)
    personas = []

    for group_label, mask_fn in GROUPS:
        subset = df[mask_fn(df)]
        picked = sample_group(subset, group_label, rng)
        for _, row in picked.iterrows():
            name = get_full_name(row)
            personas.append(
                {
                    "persona_id": row["uuid"],
                    "group": group_label,
                    "note": f"{name}, {row['age']}세 {row['sex']}, {row['occupation']}",
                }
            )
        print(f"{group_label}: {len(picked)}명 선정 (조건 대상 {len(subset)}명 중)")

    output = {"personas": personas}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료: {OUTPUT_PATH} (총 {len(personas)}명)")


if __name__ == "__main__":
    main()
