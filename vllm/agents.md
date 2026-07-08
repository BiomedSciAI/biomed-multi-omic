# BiomedRNA vLLM Plugin

A vLLM plugin for single-cell RNA sequencing (scRNA-seq) embeddings using IBM's BiomedRNA foundation model.

## Overview

This plugin enables vLLM to process scRNA-seq data and generate cell embeddings using pre-trained BMFM (Biomedical Foundation Model) checkpoints.

**Key Features:**
- Loads pre-trained BMFM checkpoints from HuggingFace or local paths
- Supports multiple pooling strategies for embedding extraction
- Integrates seamlessly with vLLM's multimodal data pipeline
- Maintains numerical consistency with reference implementation

## Architecture

**Model Specifications:**
- HuggingFace: `ibm-research/biomed.rna.llama.47m.wced.multitask.v1`
- Architecture: LLaMa-based encoder (12 layers, 384 hidden dim, 12 attention heads)
- Parameters: 47M
- Output: 384-dimensional cell embeddings

**Plugin Components:**
- `BiomedRnaForSequenceEmbedding`: Main model class wrapping BMFM's `LlamaForMultiTaskModel`.
  Pooling is applied inside `forward()` using a `pooling_method_id` field carried through
  the multimodal kwargs pipeline. `BiomedRnaPooler` is a simple pass-through.
- `BiomedRnaPooler`: Pass-through pooler — `forward()` already returns pooled embeddings.
- `preprocess.py`: Converts gene expression data to vLLM format. Accepts an optional
  `pooling_method` argument that is embedded in each cell's `multi_modal_data` dict.

## Installation
First instell bmfm-targets per its instructions.
The intall the plugin:
```bash
cd vllm
pip install -e .
```

## Input Format

The plugin expects RNA data with:
- **gene_ids**: Integer tensor of tokenized gene names `[seq_len]`
- **expr_values**: Float tensor of expression values `[seq_len]`
- **attention_mask** (optional): Binary mask for padding `[seq_len]`

Gene names are tokenized using BMFM's vocabulary (19,321 genes).

## Batch Processing

The `forward()` method handles two data paths:

**Offline (DataModule):**
- Pre-padded tensors: `[batch, seq_len]`
- All sequences same length
- Direct tensor operations

**Online (API requests):**
- Variable-length lists of tensors
- Dynamic padding to batch max length
- Handles mixed sequence lengths efficiently

Padding uses:
- `PAD_TOKEN_ID = 2` for gene_ids
- `0.0` for expression values
- `0` for attention masks

## Usage Examples

### Offline Mode (`offline_biomed_rna_example.py`)
Direct inference using the vLLM model instance without a server. Best for:
- Development and testing
- Batch processing of h5ad files
- Single-machine workflows

**Features:**
- Loads model directly via `get_vllm_biomed_rna_model()`
- Processes h5ad files using `preprocess_anndata()`
- Supports batch processing and full file iteration
- No server setup required

### Online Mode (`online_biomed_rna_example.py`)
Production deployment using vLLM server with IO processor plugin. Best for:
- Production APIs
- Multi-client access
- Scalable inference

**Features:**
- HTTP API via `/pooling` endpoint
- JSON input/output format
- Automatic batching and optimization
- Requires IO processor plugin

## IO Processor Plugin

The `BiomedRnaIOProcessor` enables online inference by handling data conversion between HTTP requests and vLLM's internal format.

**Key Functions:**
- `parse_data()`: Passes the full request dict through as an `RnaPrompt` (including any `pooling_method` field).
- `pre_process()`: Converts to vLLM format. If `pooling_method` is present in the prompt, it is forwarded into `multi_modal_data["rna"]` so that `RnaProcessorItems` can encode it as a `pooling_method_id` tensor.
- `post_process()`: Extracts embeddings from model output.
- `merge_pooling_params()`: Sets task type to `"embed"`. No `extra_kwargs` — pooling method travels via mm_kwargs.

**Runtime Pooling Override:**
Include `pooling_method` in the `data` field of the request to override the model default for that request:
```json
{ "model": "...", "data": { "gene_ids": [...], "expr_values": [...], "pooling_method": "mean_pooling" } }
```

**Data Flow:**
```
HTTP JSON → parse_data() → pre_process() → vLLM batching →
Model.forward() (pools here) → BiomedRnaPooler.forward() (pass-through) → post_process() → HTTP JSON response
```

## Why pooling_method travels via mm_kwargs

vLLM's pooling pipeline has a hard split between `forward()` and the pooler:

```
model.forward(input_ids, positions, **mm_kwargs)  →  hidden_states
model.pooler(hidden_states, pooling_metadata)      →  embeddings
```

`PoolingParams.extra_kwargs` is only available inside the pooler — it is not passed to `forward()`. This means the pooler cannot call `get_embeddings_from_outputs()` with different methods per request without access to the raw BMFM `ModelOutput` and `attention_mask`, both of which only exist during `forward()`. Storing them on the model as instance state between the two calls is fragile and not thread-safe.

The solution is to encode `pooling_method` as a scalar integer tensor (`pooling_method_id`) and include it in the per-request multimodal data dict alongside `gene_ids`, `expr_values`, and `attention_mask`. vLLM's `MultiModalBatchedField` stacks all per-request scalar tensors into `Tensor[batch, 1]`, which arrives in `forward(**kwargs)` just like any other mm field. Pooling is then performed inside `forward()`, and `BiomedRnaPooler` becomes a simple pass-through that unwraps the already-pooled `[batch, 1, hidden_size]` tensor.

Integer encoding (see `constants.py`):

| ID | Method |
|----|--------|
| -2 | `first_token` (CLS at position 0) |
| -3 | `mean_pooling` (masked mean of non-CLS tokens) |
| -4 | `pooling_layer` (trained pooler head output) |
| n ≥ 0 | literal token at position n |

Named methods use negative codes so they can never collide with token positions (always ≥ 0).

## Testing

```bash
pytest tests/

# Specific tests
pytest tests/test_biomed_rna.py          # batching correctness + pooling override
pytest tests/test_io_processor.py        # IO processor + pooling_method plumbing
pytest tests/test_regression.py          # vLLM vs direct bmfm-targets comparison
```

## References
- [HuggingFace Model](https://huggingface.co/ibm-research/biomed.rna.llama.47m.wced.multitask.v1)
- [Paper: arXiv:2506.14861](https://arxiv.org/abs/2506.14861)
