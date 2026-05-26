# vLLM BiomedRNA Model Plugins

Running BiomedRNA models with vLLM through the plugin system.

## Installation

Install biomed-multi-omic and vllm plugin:

```
cd $HOME/git/biomed-multi-omic
pip install -e ..
cd vllm
pip install -e .
```
`

## Prerequisites

### Model Weights Conversion (Required)

The HuggingFace model repository currently lacks `model.safetensors` files. We convert the checkpoint to SafeTensors format and config.json before using this plugin.


```bash
python scripts/convert_ckpt_to_safetensors.py
```

**Optional arguments:**
- `--output-dir`: Output directory (default: `/dccstor/bmfm-targets1/users/sivanra/models`)
- `--model-id`: HuggingFace model ID (default: `ibm-research/biomed.rna.llama.47m.wced.multitask.v1`)



## Usage

To analyze h5ad file and generate embeddings run  [`examples/biomed_rna_example.py`](examples/biomed_rna_example.py)

```bash
# Edit paths in the example file, then run on gpu:
python examples/biomed_rna_example.py
```

## Testing

```bash
python -m pytest tests/ -v -s
```
