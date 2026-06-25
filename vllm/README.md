# vLLM BiomedRNA Model Plugin

Running Inference at scale for BiomedRNA models via vLLM plugin.

## Installation

Install biomed-multi-omic with vLLM support:

```bash
uv pip install -e ".[vllm]"
```

This installs the base package plus the vLLM plugin with all required dependencies.

## Supported Models

The plugin supports multiple BiomedRNA model variants:

1. **47M WCED Model** (default): `ibm-research/biomed.rna.llama.47m.wced.multitask.v1.vllm`
   - 47M parameters, WCED Multitask
   - Default model used in examples

2. **32M MLM Model**: `ibm-research/biomed.rna.llama.32m.mlm.multitask.v1.vllm`
   - 32M parameters, MLM Multitask


## Usage

Both models can be accessed using the constants defined in [`vllm_biomed_rna_plugin.utils`](vllm_biomed_rna_plugin/utils.py):
- `WCED_MULTITASK_MODEL` - Default 47M WCED model
- `MLM_MULTITASK_MODEL` - 32M MLM model

### Offline Mode (Direct LLM)

To analyze h5ad file and generate embeddings directly using the LLM instance, run [`examples/offline_biomed_rna_example.py`](examples/offline_biomed_rna_example.py):

```bash
# Run on GPU (uses WCED model by default):
python examples/offline_biomed_rna_example.py
```

### Online Mode (vLLM Server with IO Processor Plugin)

For production deployments, use the vLLM server mode with the custom IO processor plugin. This plugin handles the serialization/deserialization of multi-modal RNA data (gene IDs + expression values) between HTTP JSON and vLLM's internal format.

**1. Start the vLLM server:**

```bash
# Use default 47M WCED model
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

# Or use 32M MLM model
vllm serve ibm-research/biomed.rna.llama.32m.mlm.multitask.v1.vllm \
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

**Important Flags**:
- `--io-processor-plugin biomed_rna`: **Required** - Enables the custom IO processor for RNA data
- `--enable-mm-embeds`: **Required** - Enables multi-modal embeddings support
- `--dtype float32`: **Required** - BMFM model requires float32 precision

**IO Processor**: The `biomed_rna` IO processor handles conversion between HTTP JSON (gene IDs + expression values) and vLLM's internal tensor format. It processes RNA data through the `/pooling` endpoint.

**Note on Tokenizer**: Do NOT use `--skip-tokenizer-init`. While the biomed_rna IO processor doesn't use a tokenizer, vLLM's pooling endpoint initializes a default scoring processor that requires one. The tokenizer will be created from the `vocab_size` in config.json but won't be used for RNA processing.

**Note on Model Name**: By default, vLLM uses the full model path as the `served_model_name`. The client must use this exact name in requests. You can override this with `--served-model-name` flag when starting the server, or set the `VLLM_MODEL_NAME` environment variable when running the client.

**2. Run the online example:**

```bash
python examples/online_biomed_rna_example.py
```

The example uses the `/pooling` endpoint with the IO processor plugin.


## Testing

```bash
python -m pytest tests/ -v -s
```
