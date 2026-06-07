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
- `BiomedRnaForSequenceEmbedding`: Main model class wrapping BMFM's `LlamaForMultiTaskModel`
- `preprocess.py`: Converts gene expression data to vLLM format

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
- `parse_data()`: Validates incoming JSON with gene_ids, expr_values, attention_mask
- `pre_process()`: Converts to vLLM format with dummy token and multi-modal data
- `post_process()`: Extracts embeddings from model output
- `merge_pooling_params()`: Sets task type to "embed" for pooling endpoint

**Data Flow:**
```
HTTP JSON → parse_data() → pre_process() → vLLM batching →
Model.forward() → post_process() → HTTP JSON response
```

See `docs/io_processor_comparison.md` for implementation details.

## Testing

```bash
pytest tests/

# Specific tests
pytest tests/test_biomed_rna.py      # unit tests
pytest tests/test_regression.py      # vLLM vs direct bmfm-targets comparison
pytest tests/test_plugin_integration.py  # Plugin registration
pytest tests/test_preprocess.py      # Data preprocessing
```

## References
- [HuggingFace Model](https://huggingface.co/ibm-research/biomed.rna.llama.47m.wced.multitask.v1)
- [Paper: arXiv:2506.14861](https://arxiv.org/abs/2506.14861)
