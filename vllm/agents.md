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
- Uses BMFM's native forward pass and pooling methods

## Installation

```bash
pip install -e .
pip install bmfm-targets
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

## Usage

See `examples/` directory:
- `biomed_rna_example.py` - Basic inference
- `biomed_rna_streaming_example.py` - Streaming h5ad files

## Testing

```bash
# Run all tests
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
