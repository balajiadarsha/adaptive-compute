"""
Dataset loading for UWNSL/SafeChain and allenai/wildguardmix, shared by
eval_safety.py (evaluation) and the quantize/ scripts (GPTQ/W8A8/W4A8
calibration data).
"""
import datasets

DATASETS = ("safechain", "wildguard")

DATASET_HF_ID = {
    "safechain": "UWNSL/SafeChain",
    "wildguard": "allenai/wildguardmix (wildguardtest)",
}


def load_safechain(n: int | None, seed: int = 42) -> list[dict]:
    ds = datasets.load_dataset("UWNSL/SafeChain", split="train")
    if n is not None:
        ds = ds.shuffle(seed=seed).select(range(n))
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


def load_dataset(name: str, n: int | None, seed: int = 42) -> list[dict]:
    if name == "safechain":
        return load_safechain(n, seed=seed)
    elif name == "wildguard":
        return load_wildguardmix(n, seed=seed)
    raise ValueError(f"Unknown dataset: {name!r}. Choose from {DATASETS}")
