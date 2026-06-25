"""
Online example to get embeddings from BiomedRNA model via vLLM server.

This example uses the custom IO processor plugin to handle multi-modal RNA data
through vLLM's /pooling endpoint.

Start the server first:

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

Note: Do NOT use --skip-tokenizer-init. While the biomed_rna IO processor doesn't
use a tokenizer, vLLM's pooling endpoint initializes a default scoring processor
that requires one. The tokenizer will be created from the vocab_size in config.json
but won't be used for RNA processing.

Monitor server start up using:

    tail -f vllm_server.log

Then run this script:

    python examples/online_biomed_rna_example.py

The IO processor plugin handles serialization/deserialization between
HTTP JSON and vLLM's internal multi-modal format.
"""

import os
from pathlib import Path

import anndata
import numpy as np
import requests
from vllm_biomed_rna_plugin.preprocess import preprocess_anndata
from vllm_biomed_rna_plugin.utils import WCED_MULTITASK_MODEL, load_tokenizer

# Path to example h5ad file
ZHENG_SMALL_H5AD_PATH = Path(__file__).parent / "resources" / "zheng68k.h5ad"

# Model to use - change to MLM_MULTITASK_MODEL if desired
MODEL_REPO = WCED_MULTITASK_MODEL


def main():
    """
    Demonstrate online embedding generation using vLLM server with IO processor plugin.

    Uses the /pooling endpoint with custom RNA data format handled by the
    biomed_rna IO processor plugin.
    """
    # Server configuration
    server_host = os.environ.get("VLLM_SERVER_HOST", "localhost")
    server_port = os.environ.get("VLLM_SERVER_PORT", "8000")
    server_url = f"http://{server_host}:{server_port}/pooling"
    model_name = os.environ.get(
        "VLLM_MODEL_NAME",
        MODEL_REPO,
    )

    # Load and preprocess h5ad data
    print("=" * 80)
    print("BiomedRNA Online Embedding Generation (IO Processor Plugin)")
    print("=" * 80)
    print(f"Loading data from: {ZHENG_SMALL_H5AD_PATH}")

    # Load a small subset of cells for demonstration
    num_cells = 10
    adata = anndata.read_h5ad(ZHENG_SMALL_H5AD_PATH)[:num_cells]
    print(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

    # Load tokenizer and preprocess
    print("Preprocessing cells...")
    tokenizer = load_tokenizer(MODEL_REPO)
    cell_data = preprocess_anndata(
        adata,
        tokenizer,
        max_length=1024,
    )

    print(f"Preprocessed {len(cell_data)} cells")
    print(
        f"Sequence length: {len(cell_data[0]['multi_modal_data']['rna']['gene_ids'])}"
    )
    print(f"Server endpoint: {server_url}")
    print(f"Model: {model_name}")
    print()

    # Process each cell and get embeddings
    print("Generating embeddings from server...")
    embeddings = []

    for i, cell in enumerate(cell_data):
        rna_data = cell["multi_modal_data"]["rna"]

        payload = {
            "model": model_name,
            "data": {
                "gene_ids": rna_data["gene_ids"].tolist(),
                "expr_values": rna_data["expr_values"].tolist(),
                "attention_mask": rna_data["attention_mask"].tolist(),
            },
        }
        # send to vllm
        try:
            response = requests.post(server_url, json=payload, timeout=60)

            if response.status_code != 200:
                print(
                    f"\n❌ Server returned error {response.status_code} for cell {i+1}"
                )
                print(f"Response: {response.text}")
                response.raise_for_status()

            result = response.json()

            # Extract embedding from IO processor output
            if "data" in result:
                embedding = np.array(result["data"]["embedding"])
            elif "embedding" in result:
                embedding = np.array(result["embedding"])
            else:
                raise ValueError(f"Unexpected response format: {result.keys()}")
            embeddings.append(embedding)

            print(f"Cell {i+1}/{len(cell_data)}: embedding shape {embedding.shape}")

        except requests.exceptions.RequestException as e:
            print(f"\n❌ Error communicating with server for cell {i+1}: {e}")
            raise

    # Display results
    print()
    print("=" * 80)
    print("Embedding Statistics")
    print("=" * 80)

    # Convert to numpy array for analysis
    embeddings_array = np.array(embeddings)  # [num_cells, hidden_size]

    print(f"Total embeddings shape: {embeddings_array.shape}")
    print(
        f"Embedding value range: [{embeddings_array.min():.4f}, {embeddings_array.max():.4f}]"
    )

    print()
    print("=" * 80)
    print("✓ Successfully generated embeddings from vLLM server!")
    print("=" * 80)


if __name__ == "__main__":
    main()
