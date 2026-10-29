"""
Dataset loading for UWNSL/SafeChain, allenai/wildguardmix and StrongREJECT,
shared by eval_safety.py (evaluation) and the quantize/ scripts (GPTQ/W8A8/W4A8
calibration data).
"""
import os

import datasets
import pandas as pd

DATASETS = ("safechain", "wildguard", "strongreject")

DATASET_HF_ID = {
    "safechain": "UWNSL/SafeChain",
    "wildguard": "allenai/wildguardmix (wildguardtest)",
    "strongreject": "StrongREJECT (alexandrasouly/strongreject)",
}

# The GPTQ/AWQ checkpoints were calibrated on load_safechain(256, seed=42),
# i.e. the first 256 rows of the seed-42 shuffle. Evaluation skips them.
SAFECHAIN_CALIB_SAMPLES = 256

# walledai/StrongREJECT on the Hub is gated; this is the benchmark authors'
# CSV (313 forbidden prompts) from github.com/alexandrasouly/strongreject.
# UWNSL/SafeChain train split exported with Dataset.to_parquet (same row
# order as the Hub split). The flatquant env's older `datasets` can't read the
# cache the reason env wrote, and offline it won't fall back to the Hub.
SAFECHAIN_PARQUET = os.path.join(os.path.dirname(__file__), "..", "data", "safechain", "train.parquet")

STRONGREJECT_CSV = os.path.join(
    os.path.dirname(__file__), "..", "data", "strongreject", "strongreject_dataset.csv"
)


def load_safechain(n: int | None, seed: int = 42, offset: int = 0) -> list[dict]:
    """offset skips the first `offset` rows of the seed-shuffled data; eval
    uses SAFECHAIN_CALIB_SAMPLES so it never sees a calibration prompt."""
    if os.path.exists(SAFECHAIN_PARQUET):
        ds = datasets.Dataset.from_parquet(SAFECHAIN_PARQUET)
    else:
        ds = datasets.load_dataset("UWNSL/SafeChain", split="train")
    if n is not None or offset:
        ds = ds.shuffle(seed=seed)
        end = len(ds) if n is None else min(offset + n, len(ds))
        ds = ds.select(range(offset, end))
    return list(ds)


def load_wildguardmix(n: int | None, seed: int = 42) -> list[dict]:
    """Load allenai/wildguardmix (WildGuardTest split) and reshape it to the
    same {instruction, label} schema used for SafeChain, so the rest of the
    pipeline (generation, metrics) is dataset-agnostic."""
    ds = datasets.load_dataset("allenai/wildguardmix", "wildguardtest", split="test")
    ds = ds.filter(lambda ex: ex["prompt_harm_label"] is not None)
    if n is not None:
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    items = []
    for ex in ds:
        harm = "harmful" if ex["prompt_harm_label"] == "harmful" else "benign"
        adv = "adversarial" if ex["adversarial"] else "vanilla"
        items.append({
            "instruction": ex["prompt"],
            "label": f"{adv}_{harm}",
        })
    return items


def load_strongreject(n: int | None, seed: int = 42) -> list[dict]:
    """All 313 StrongREJECT prompts are harmful requests without a jailbreak
    wrapper, so every item is labeled vanilla_harmful."""
    df = pd.read_csv(STRONGREJECT_CSV)
    if n is not None:
        df = df.sample(n=min(n, len(df)), random_state=seed)
    return [
        {"instruction": row.forbidden_prompt, "label": "vanilla_harmful", "category": row.category}
        for row in df.itertuples()
    ]


def load_dataset(name: str, n: int | None, seed: int = 42, offset: int = 0) -> list[dict]:
    if name == "safechain":
        return load_safechain(n, seed=seed, offset=offset)
    elif name == "wildguard":
        return load_wildguardmix(n, seed=seed)
    elif name == "strongreject":
        return load_strongreject(n, seed=seed)
    raise ValueError(f"Unknown dataset: {name!r}. Choose from {DATASETS}")
