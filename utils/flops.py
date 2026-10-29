"""
Analytical, precision-independent FLOPs-per-token estimation for the dense
Qwen2-architecture 1.5B models evaluated in this repo (DeepSeek-R1-Distill-
Qwen-1.5B and L1-Qwen-1.5B-Exact -- both `Qwen2ForCausalLM`, same backbone
shape, only fine-tuning/quantization differ).

Quantization changes which kernel runs the matmul (and therefore achieved
throughput/memory traffic), not the number of multiply-adds the algorithm
performs -- so a single FLOPs-per-token formula, parameterized only by
architecture + sequence length, applies to every precision variant of a
given model. Multiply the result by measured tokens/sec to get achieved
FLOP/s, and compare that against the hardware peak for whatever precision
the matmul actually executes in (see eval/mfu_report.py) -- NOT a peak for
the nominal quantization bit-width, since e.g. GPTQ/AWQ weight-only
quantization dequantizes to bf16 before the GEMM runs.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelFlopsConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int
    tie_word_embeddings: bool = False


# deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B and l3lab/L1-Qwen-1.5B-Exact
# share this exact backbone shape (both Qwen2ForCausalLM fine-tunes);
# tie_word_embeddings=False confirmed in both models' real config.json.
QWEN2_1_5B = ModelFlopsConfig(
    hidden_size=1536,
    num_hidden_layers=28,
    num_attention_heads=12,
    num_key_value_heads=2,
    intermediate_size=8960,
    vocab_size=151936,
    tie_word_embeddings=False,
)


def backbone_linear_flops_per_token(cfg: ModelFlopsConfig) -> float:
    """Precision-independent FLOPs for one token through every linear layer
    EXCEPT lm_head (QKVO attention projections + SwiGLU MLP), summed over
    all layers. Uses the 2*in*out multiply-add convention. GQA: K/V project
    to num_key_value_heads*head_dim, not hidden_size."""
    H, L, I = cfg.hidden_size, cfg.num_hidden_layers, cfg.intermediate_size
    head_dim = H // cfg.num_attention_heads
    kv_dim = cfg.num_key_value_heads * head_dim
    per_layer = (
        2 * H * H         # Q proj (H -> H, since n_heads*head_dim == H)
        + 2 * H * kv_dim  # K proj (H -> kv_dim)
        + 2 * H * kv_dim  # V proj (H -> kv_dim)
        + 2 * H * H       # O proj (H -> H)
        + 2 * H * I * 3   # SwiGLU: gate + up (H->I each) + down (I->H)
    )
    return per_layer * L


def lm_head_flops_per_token(cfg: ModelFlopsConfig) -> float:
    return 2 * cfg.hidden_size * cfg.vocab_size


def attention_flops_for_sequence(cfg: ModelFlopsConfig, prompt_len: int, output_len: int) -> float:
    """QK^T + softmax*V FLOPs for the full request (causal), summed across
    all layers, from position 1 through prompt_len+output_len. GQA does NOT
    reduce this term -- attention score compute scales with
    num_attention_heads (queries), not num_key_value_heads; GQA only
    shrinks the K/V projection matmuls (already counted in
    backbone_linear_flops_per_token) and the KV-cache memory footprint, not
    this compute term. Per-position-t cost (1-indexed) = 4*hidden_size*t
    per layer (QK^T and attn@V, each 2*n_heads*head_dim*t = 2*hidden_size*t
    FLOPs since n_heads*head_dim == hidden_size); summed 1..N via
    N*(N+1)/2."""
    H, L = cfg.hidden_size, cfg.num_hidden_layers
    N = prompt_len + output_len
    return 4 * H * L * (N * (N + 1) / 2)


def total_flops_for_response(cfg: ModelFlopsConfig, prompt_len: int, output_len: int) -> dict:
    """Total algorithmic FLOPs for one (prompt, response) pair.

    IMPORTANT accounting detail: the backbone runs on every processed token
    (prompt_len prefill tokens + output_len decode tokens), but lm_head is
    only evaluated once per SAMPLED token (output_len times total) -- not
    once per prompt token. Getting this wrong double-counts lm_head FLOPs
    by a large factor for long-prompt/short-output requests (e.g. GPQA)."""
    n_tokens = prompt_len + output_len
    backbone = backbone_linear_flops_per_token(cfg) * n_tokens
    lm_head = lm_head_flops_per_token(cfg) * output_len
    attn = attention_flops_for_sequence(cfg, prompt_len, output_len)
    return {
        "backbone_flops": backbone,
        "lm_head_flops": lm_head,
        "attention_flops": attn,
        "total_flops": backbone + lm_head + attn,
    }


if __name__ == "__main__":
    import json
    import os
    import sys

    cfg = QWEN2_1_5B

    # Sanity check: backbone_linear_flops_per_token(cfg) / 2 should equal
    # the real non-embedding parameter count of the checkpoint (standard
    # "FLOPs (forward) ~= 2*N" identity). Compare against the actual
    # checkpoint if a path is given, else just print the implied count.
    implied_non_embed_params = backbone_linear_flops_per_token(cfg) / 2
    print(f"Implied non-embedding params (backbone_flops/2): {implied_non_embed_params:,.0f}")

    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    if ckpt:
        from safetensors import safe_open

        st_path = os.path.join(ckpt, "model.safetensors")
        total_params = 0
        embed_params = 0
        with safe_open(st_path, framework="pt") as f:
            for key in f.keys():
                shape = f.get_slice(key).get_shape()
                numel = 1
                for d in shape:
                    numel *= d
                total_params += numel
                if "embed_tokens" in key or "lm_head" in key:
                    embed_params += numel
        non_embed = total_params - embed_params
        print(f"Checkpoint: {ckpt}")
        print(f"  total params (incl. embed/lm_head): {total_params:,}")
        print(f"  embed_tokens + lm_head params:       {embed_params:,}")
        print(f"  non-embedding params:                {non_embed:,}")
        pct_diff = 100 * (implied_non_embed_params - non_embed) / non_embed
        print(f"  formula vs actual non-embed params:  {pct_diff:+.2f}% difference")
    else:
        print("(pass a checkpoint dir as argv[1] to cross-check against real param counts)")

    print()
    print("FLOPs/token at representative sequence lengths (prompt_len=0 for this table; "
          "attention term grows with total sequence length, shown separately):")
    for seq_len in (512, 4096, 30000):
        r = total_flops_for_response(cfg, prompt_len=0, output_len=seq_len)
        per_token = r["total_flops"] / seq_len
        print(f"  seq_len={seq_len:>6}: total={r['total_flops']:.3e} FLOPs, "
              f"per-token={per_token:.3e} FLOPs/tok "
              f"(backbone {r['backbone_flops']:.3e}, lm_head {r['lm_head_flops']:.3e}, "
              f"attn {r['attention_flops']:.3e})")
