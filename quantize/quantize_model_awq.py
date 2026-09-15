"""
Post-training AWQ (Activation-aware Weight Quantization) of a base
PyTorch/HuggingFace checkpoint into a standalone weight-only INT4
checkpoint using llmcompressor (the compressed-tensors format vLLM loads
natively via its "compressed-tensors" quant backend -- no extra flags
needed on eval_safety.py/eval_skill.py, vLLM auto-detects the format from
config.json).

Unlike GPTQ (see quantize_model.py), AWQ picks per-channel weight scales
from activation magnitude statistics gathered during calibration (rather
than the Hessian-based approach GPTQ uses), then quantizes weights only --
activations stay full precision at inference time.

Calibration prompts are drawn from the same SafeChain / WildGuardMix
datasets used for evaluation, matching quantize_model.py /
quantize_model_activation.py.

Usage:
    python quantize_model_awq.py --model-id l3lab/L1-Qwen-1.5B-Exact \
        --output-dir ./checkpoints/L1-Qwen-1.5B-Exact-awq-w4a16_asym

    python quantize_model_awq.py --model-id l3lab/L1-Qwen-1.5B-Exact \
        --scheme W4A16 --calib-dataset wildguard --calib-samples 256 \
        --output-dir ./checkpoints/L1-Qwen-1.5B-Exact-awq-w4a16
"""
import argparse
import glob
import json
import os
import sys
import shutil

from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from llmcompressor.modifiers.transform.awq import AWQModifier
except ImportError:
    from llmcompressor.modifiers.awq import AWQModifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from dataset_utils import DATASETS, load_dataset  # noqa: E402

SCHEMES = ("W4A16", "W4A16_ASYM")


def build_calibration_dataset(
    tokenizer, dataset: str, n: int, seed: int, max_seq_length: int
) -> Dataset:
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

    ds = Dataset.from_dict({"text": texts})

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            truncation=True,
            max_length=max_seq_length,
        )

    return ds.map(tokenize, remove_columns=ds.column_names)


def quantize(
    model_id: str,
    scheme: str,
    output_dir: str,
    calib_dataset: str,
    calib_samples: int,
    seed: int,
    max_seq_length: int,
) -> None:
    print(f"\nLoading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype="auto"
    )

    print(f"Building {calib_samples}-sample calibration set from {calib_dataset}...")
    calibration_data = build_calibration_dataset(
        tokenizer, calib_dataset, calib_samples, seed, max_seq_length
    )

    recipe = [
        AWQModifier(duo_scaling="both"),
        QuantizationModifier(ignore=["lm_head"], scheme=scheme, targets=["Linear"]),
    ]

    print(f"\nRunning AWQ quantization (scheme={scheme})...")
    oneshot(
        model=model,
        tokenizer=tokenizer,
        recipe=recipe,
        dataset=calibration_data,
        num_calibration_samples=calib_samples,
        max_seq_length=max_seq_length,
        output_dir=output_dir,
    )

    # Tokenizer saving mirrors quantize_model.py's more careful path rather
    # than a bare tokenizer.save_pretrained(): re-serializing tokenizer.json
    # with a newer `tokenizers` library than eval environments are pinned to
    # can fail to parse ("data did not match any variant of untagged enum
    # ModelWrapper"), and save_pretrained() can carry forward a stale
    # `tokenizer_class` that forces AutoTokenizer down a slow-tokenizer path
    # with files that were never saved.
    if os.path.isdir(model_id):
        tokenizer_patterns = (
            "tokenizer*.json", "tokenizer*.model", "vocab*.json",
            "merges.txt", "special_tokens_map.json", "chat_template*",
        )
        for pattern in tokenizer_patterns:
            for src in glob.glob(os.path.join(model_id, pattern)):
                shutil.copy2(src, output_dir)
    else:
        tokenizer.save_pretrained(output_dir)
        if tokenizer.is_fast:
            tokenizer_config_path = os.path.join(output_dir, "tokenizer_config.json")
            with open(tokenizer_config_path) as f:
                tokenizer_config = json.load(f)
            tokenizer_config.pop("tokenizer_class", None)
            with open(tokenizer_config_path, "w") as f:
                json.dump(tokenizer_config, f, indent=2)

    print(f"\nSaved AWQ ({scheme}) checkpoint to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", type=str, required=True,
                        help="HuggingFace model ID or local path of the BF16 base model")
    parser.add_argument("--scheme", type=str, default="W4A16_ASYM", choices=SCHEMES,
                        help="AWQ weight-only quantization scheme (default: W4A16_ASYM)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save the quantized checkpoint "
                             "(default: ./checkpoints/<model_name>-awq-<scheme>)")
    parser.add_argument("--calib-dataset", type=str, default="safechain", choices=DATASETS,
                        help="Dataset to draw calibration prompts from (default: safechain)")
    parser.add_argument("--calib-samples", type=int, default=256,
                        help="Number of calibration prompts (default: 256)")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Max sequence length for calibration samples (default: 2048)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        model_name = args.model_id.rstrip("/").split("/")[-1]
        output_dir = os.path.join("checkpoints", f"{model_name}-awq-{args.scheme.lower()}")

    quantize(
        model_id=args.model_id,
        scheme=args.scheme,
        output_dir=output_dir,
        calib_dataset=args.calib_dataset,
        calib_samples=args.calib_samples,
        seed=args.seed,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
