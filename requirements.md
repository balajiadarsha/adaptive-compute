torch>=2.3.0        # verified with 2.13.0+cu130
transformers>=4.40.2  # verified with 5.15.0
vllm>=0.4.2         # verified with 0.27.1
datasets            # verified with 5.0.1
pandas              # verified with 3.0.5
matplotlib
numpy               # verified with 2.2.6
openpyxl
tqdm                # verified with 4.70.0
gptqmodel>=7.3.2    # verified with 7.3.2
llmcompressor>=0.12.0
compressed-tensors>=0.17.1  # verified with 0.17.0

# Known conflict: installing vllm>=0.27 pulls in protobuf 6.33.6, which is
# older than gptqmodel's protobuf>=7.34.0 requirement (pip warns but does
# not fail). Not an issue for eval_safety.py / eval_skill.py, which don't
# import gptqmodel; only matters if quantize_model.py runs in the same env.