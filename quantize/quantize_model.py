"""
Post-training GPTQ quantization of a base PyTorch/HuggingFace checkpoint into
standalone 8-bit, 4-bit, or 3-bit checkpoints that eval_safety.py can load
directly via --model-id (vLLM auto-detects GPTQ checkpoints from the
quantize_config.json written into the output directory).

BF16 needs no quantization step -- just point --model-id at the original
base checkpoint when running eval_safety.py.

Calibration prompts are drawn from the same SafeChain / WildGuardMix
datasets used for evaluation, formatted through the model's chat template,
so the calibration distribution matches the eval distribution.

Usage:
    python quantize_model.py --model-id l3lab/L1-Qwen-1.5B-Exact --bits 4 \
        --output-dir ./checkpoints/L1-Qwen-1.5B-Exact-gptq-int4

    python quantize_model.py --model-id l3lab/L1-Qwen-1.5B-Exact --bits 8 \
        --calib-dataset wildguard --calib-samples 256 \
        --output-dir ./checkpoints/L1-Qwen-1.5B-Exact-gptq-int8
"""
import argparse
import glob
import json
import os
import shutil

import datasets
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer

BITS = (8, 4, 3)
DATASETS = ("safechain", "wildguard")


def load_safechain(n: int | None, seed: int = 42) -> list[dict]:
    ds = datasets.load_dataset("UWNSL/SafeChain", split="train")
    if n is not None:
        ds = ds.shuffle(seed=seed).select(range(n))
    return list(ds)


def load_wildguardmix(n: int | None, seed: int = 42) -> list[dict]:
    """Load allenai/wildguardmix (WildGuardTest split) and reshape it to the
    same {instruction, label} schema used for SafeChain, so calibration
    sampling is dataset-agnostic."""
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


def build_calibration_data(
    tokenizer, dataset: str, n: int, seed: int
) -> list[str]:
    items = load_dataset(dataset, n, seed=seed)
    texts = []
    for item in items:
        msg = [{"role": "user", "content": item["instruction"]}]
        if tokenizer.chat_template is not None:
            text = tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
        else:
            text = f"## Instruction: {item['instruction']}\n## Response: "
        texts.append(text)
    return texts


def quantize(
    model_id: str,
    bits: int,
    output_dir: str,
    calib_dataset: str,
    calib_samples: int,
    seed: int,
    group_size: int,
) -> None:
    print(f"\nLoading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, use_fast=True
    )

    print(f"Building {calib_samples}-sample calibration set from {calib_dataset}...")
    calibration_data = build_calibration_data(
        tokenizer, calib_dataset, calib_samples, seed
    )

    quant_config = QuantizeConfig(bits=bits, group_size=group_size)

    print(f"\nLoading {model_id} for GPTQ quantization (bits={bits})...")
    model = GPTQModel.load(model_id, quant_config, trust_remote_code=True)

    print("Quantizing (this may take a while)...")
    model.quantize(calibration_data)

    os.makedirs(output_dir, exist_ok=True)
    model.save(output_dir)

    if os.path.isdir(model_id):
        # Copy tokenizer files verbatim instead of tokenizer.save_pretrained().
        # GPTQ quantization never touches the tokenizer, and re-serializing it
        # ties the checkpoint to whichever `tokenizers` library version wrote
        # it here -- a newer tokenizer.json can fail to parse ("data did not
        # match any variant of untagged enum ModelWrapper") in eval
        # environments pinned to an older `tokenizers` for vLLM compatibility.
        # A verbatim copy of the source files sidesteps that entirely.
        tokenizer_patterns = (
            "tokenizer*.json", "tokenizer*.model", "vocab*.json",
            "merges.txt", "special_tokens_map.json", "chat_template*",
        )
        for pattern in tokenizer_patterns:
            for src in glob.glob(os.path.join(model_id, pattern)):
                shutil.copy2(src, output_dir)
    else:
        tokenizer.save_pretrained(output_dir)

        # tokenizer.save_pretrained() carries forward whatever
        # `tokenizer_class` string was in the source checkpoint's config even
        # when a fast (tokenizers-backed) tokenizer was actually
        # loaded/saved. A stale slow class name (e.g. GemmaTokenizer) then
        # forces AutoTokenizer to try loading a sentencepiece tokenizer.model
        # that was never saved. Strip it so AutoTokenizer auto-resolves
        # PreTrainedTokenizerFast from tokenizer.json instead.
        if tokenizer.is_fast:
            tokenizer_config_path = os.path.join(output_dir, "tokenizer_config.json")
            with open(tokenizer_config_path) as f:
                tokenizer_config = json.load(f)
            tokenizer_config.pop("tokenizer_class", None)
            with open(tokenizer_config_path, "w") as f:
                json.dump(tokenizer_config, f, indent=2)

    print(f"\nSaved {bits}-bit GPTQ checkpoint to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", type=str, required=True,
                        help="HuggingFace model ID or local path of the BF16 base model")
    parser.add_argument("--bits", type=int, required=True, choices=BITS,
                        help="Target quantization bit-width")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save the quantized checkpoint "
                             "(default: ./checkpoints/<model_name>-gptq-int<bits>)")
    parser.add_argument("--calib-dataset", type=str, default="safechain", choices=DATASETS,
                        help="Dataset to draw calibration prompts from (default: safechain)")
    parser.add_argument("--calib-samples", type=int, default=256,
                        help="Number of calibration prompts (default: 256)")
    parser.add_argument("--group-size", type=int, default=128,
                        help="GPTQ quantization group size (default: 128)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        model_name = args.model_id.rstrip("/").split("/")[-1]
        output_dir = os.path.join("checkpoints", f"{model_name}-gptq-int{args.bits}")

    quantize(
        model_id=args.model_id,
        bits=args.bits,
        output_dir=output_dir,
        calib_dataset=args.calib_dataset,
        calib_samples=args.calib_samples,
        seed=args.seed,
        group_size=args.group_size,
    )


if __name__ == "__main__":
    main()
