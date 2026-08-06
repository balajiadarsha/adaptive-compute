"""
Post-training weight-activation quantization of a base PyTorch/HuggingFace
checkpoint into standalone W8A8 or W4A8 checkpoints using llmcompressor
(the compressed-tensors format vLLM loads natively via its
"compressed-tensors" quant backend -- no extra flags needed on
eval_safety.py, vLLM auto-detects the format from config.json).

Unlike GPTQ weight-only quantization (see quantize_model.py), these schemes
also quantize activations at inference time:
  - W8A8: INT8 weights (per-channel, static) + INT8 activations (per-token, dynamic)
  - W4A8: INT4 weights (per-group, static)   + INT8 activations (per-token, dynamic)

There is no plain INT4/INT4 (W4A4) scheme in compressed-tensors -- the
closest 4-bit-weight/4-bit-activation formats are NVFP4 and MXFP4, which
need FP4 tensor-core support (Blackwell-class GPUs) and are out of scope
here.

Weights are calibrated with GPTQ; activations use dynamic per-token
quantization (no calibration needed for the activation scale itself, but
GPTQ's weight calibration still needs sample data). SmoothQuant runs first
to migrate activation outlier magnitude into the weights, which materially
improves W8A8/W4A8 accuracy on transformer models.

Calibration prompts are drawn from the same SafeChain / WildGuardMix
datasets used for evaluation, matching quantize_model.py.

Usage:
    python quantize_model_activation.py --model-id l3lab/L1-Qwen-1.5B-Exact --scheme W8A8 \
        --output-dir ./checkpoints/L1-Qwen-1.5B-Exact-w8a8

    python quantize_model_activation.py --model-id l3lab/L1-Qwen-1.5B-Exact --scheme W4A8 \
        --calib-dataset wildguard --calib-samples 256 \
        --output-dir ./checkpoints/L1-Qwen-1.5B-Exact-w4a8
"""
import argparse
import os
import sys

from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from dataset_utils import DATASETS, load_dataset  # noqa: E402

SCHEMES = ("W8A8", "W4A8")


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
        model_id, trust_remote_code=True, use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype="auto"
    )

    print(f"Building {calib_samples}-sample calibration set from {calib_dataset}...")
    calibration_data = build_calibration_dataset(
        tokenizer, calib_dataset, calib_samples, seed, max_seq_length
    )

    recipe = [
        SmoothQuantModifier(smoothing_strength=0.8),
        GPTQModifier(scheme=scheme, targets="Linear", ignore=["lm_head"]),
    ]

    print(f"\nRunning {scheme} quantization (SmoothQuant + GPTQ)...")
    oneshot(
        model=model,
        tokenizer=tokenizer,
        recipe=recipe,
        dataset=calibration_data,
        num_calibration_samples=calib_samples,
        max_seq_length=max_seq_length,
        output_dir=output_dir,
    )
    tokenizer.save_pretrained(output_dir)
    print(f"\nSaved {scheme} checkpoint to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", type=str, required=True,
                        help="HuggingFace model ID or local path of the BF16 base model")
    parser.add_argument("--scheme", type=str, required=True, choices=SCHEMES,
                        help="Weight-activation quantization scheme")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save the quantized checkpoint "
                             "(default: ./checkpoints/<model_name>-<scheme>)")
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
        output_dir = os.path.join("checkpoints", f"{model_name}-{args.scheme.lower()}")

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
