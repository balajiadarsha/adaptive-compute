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
import os
import sys

from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
from eval_safety import DATASETS, load_dataset  # noqa: E402

BITS = (8, 4, 3)


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
        model_id, trust_remote_code=True, use_fast=False
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
    tokenizer.save_pretrained(output_dir)
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
