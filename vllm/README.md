# vLLM BiomedRNA Model Plugin

Running Inference at scale for BiomedRNA models via vLLM plugin.

## Installation

```bash
uv pip install -e ".[vllm]"
```

## Supported Models

| Constant | HuggingFace repo | Parameters |
|----------|-----------------|------------|
| `WCED_MULTITASK_MODEL` | `ibm-research/biomed.rna.llama.47m.wced.multitask.v1.vllm` | 47M |
| `MLM_MULTITASK_MODEL` | `ibm-research/biomed.rna.llama.32m.mlm.multitask.v1.vllm` | 32M |

## Offline Mode

Generate embeddings directly from Python:

```bash
python examples/offline_biomed_rna_example.py
python examples/offline_biomed_rna_example.py --pooling-method mean_pooling
```

## Online Mode (vLLM Server)

**1. Start the server:**

```bash
vllm serve ibm-research/biomed.rna.llama.47m.wced.multitask.v1.vllm \
    --runner pooling \
    --trust-remote-code \
    --enforce-eager \
    --no-enable-prefix-caching \
    --dtype float32 \
    --gpu-memory-utilization 0.1 \
    --io-processor-plugin biomed_rna \
    --enable-mm-embeds \
     > vllm_server.log 2>&1 &
```

Required flags:
- `--io-processor-plugin biomed_rna` — enables the custom RNA IO processor
- `--enable-mm-embeds` — enables multi-modal embeddings
- `--dtype float32` — BMFM models require float32

> Do **not** use `--skip-tokenizer-init`. vLLM's pooling endpoint requires a tokenizer even though RNA processing doesn't use one.

> The client must use the full model path as `model` in requests. Override with `--served-model-name` or set `VLLM_MODEL_NAME` in the client environment.

**2. Run the example:**

```bash
python examples/online_biomed_rna_example.py
python examples/online_biomed_rna_example.py --pooling-method mean_pooling
```

## Runtime Pooling Methods

Both modes support the same pooling methods as `bmfm_targets.inference()`. List of pooling methods is not supported.

| Method | Description |
|--------|-------------|
| `"first_token"` | CLS token at position 0 (model default) |
| `"mean_pooling"` | Average of all non-CLS tokens |
| `"pooling_layer"` | Trained pooler layer output |
| `int` | CLS token at a specific position |


**Online** payload format:

```json
{
  "model": "ibm-research/biomed.rna.llama.47m.wced.multitask.v1.vllm",
  "data": {
    "gene_ids": [...],
    "expr_values": [...],
    "pooling_method": "mean_pooling"
  }
}
```

## Testing

```bash
python -m pytest tests/ -v -s
```
