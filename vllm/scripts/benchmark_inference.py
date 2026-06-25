"""
Benchmark BiomedRNA inference: Standard batched vs vLLM concurrent.
Note its best to run this on a exclusive gpu node.

Usage:
    # Run standard batched inference sweep
    python vllm/scripts/benchmark_inference.py standard --batch-sizes 8 16 32 64 128 256 512

    # Run vLLM concurrent inference sweep (requires vLLM server running)
    python vllm/scripts/benchmark_inference.py vllm --concurrent-requests 8 16 32 64 128 256 512 1024

    # Generate comparison plot
    python vllm/scripts/benchmark_inference.py plot

Start vLLM server before running vLLM benchmarks:
    vllm serve ibm-research/biomed.rna.llama.47m.wced.multitask.v1.vllm \
        --runner pooling --trust-remote-code --enforce-eager \
        --no-enable-prefix-caching --dtype float32 \
        --gpu-memory-utilization 0.5 --io-processor-plugin biomed_rna \
        --enable-mm-embeds > vllm_server.log 2>&1 &

Output: vllm/output/[standard|vllm]_*.[json|npy] and performance_comparison.png
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Optional imports for specific subcommands
try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import anndata
    from vllm_biomed_rna_plugin.preprocess import preprocess_anndata
    from vllm_biomed_rna_plugin.utils import load_tokenizer

    import bmfm_targets as bmfm

    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False


def load_dataset(num_cells: int = -1, num_genes: int = -1) -> anndata.AnnData:
    """Load and optionally subset the dataset."""
    data_dir = os.environ.get("BMFM_TARGETS_ZHENG68K_DATA")
    if not data_dir:
        raise ValueError("BMFM_TARGETS_ZHENG68K_DATA environment variable not set")

    h5ad_path = Path(data_dir) / "Zheng68k_preprocess.h5ad"
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Dataset not found: {h5ad_path}")

    print(f"Loading dataset from: {h5ad_path}")
    adata = anndata.read_h5ad(h5ad_path)

    if num_cells > 0:
        adata = adata[:num_cells]
    if num_genes > 0:
        adata = adata[:, :num_genes]

    print(f"Dataset: {adata.n_obs} cells × {adata.n_vars} genes")
    return adata


# ============================================================================
# Standard Inference
# ============================================================================


def run_standard_inference(
    adata: anndata.AnnData,
    batch_size: int,
    checkpoint: str,
    max_length: int,
) -> tuple[dict, np.ndarray]:
    """Run standard inference and return timing results."""
    print(f"\nRunning standard inference with batch_size={batch_size}...")

    start_time = time.time()
    bmfm.inference(
        adata,
        checkpoint=checkpoint,
        batch_size=batch_size,
        max_length=max_length,
        embedding_key="X_bmfm",
        device="auto",
    )
    total_time = time.time() - start_time

    embeddings = np.array(adata.obsm["X_bmfm"])

    results = {
        "method": "standard",
        "batch_size": batch_size,
        "num_cells": adata.n_obs,
        "num_genes": adata.n_vars,
        "total_time_seconds": total_time,
        "time_per_cell_ms": (total_time / adata.n_obs) * 1000,
        "embedding_shape": list(embeddings.shape),
        "checkpoint": checkpoint,
        "max_length": max_length,
    }

    print(f"Completed in {total_time:.2f}s ({results['time_per_cell_ms']:.2f} ms/cell)")
    return results, embeddings


def cmd_standard(args):
    """Run standard inference parameter sweep."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Standard Inference Parameter Sweep")
    print("=" * 80)
    adata = load_dataset(args.num_cells, args.num_genes)

    for batch_size in args.batch_sizes:
        cells_str = f"{adata.n_obs}" if args.num_cells > 0 else "all"
        genes_str = f"{adata.n_vars}" if args.num_genes > 0 else "all"
        base_name = f"standard_batch{batch_size}_cells{cells_str}_genes{genes_str}"
        json_path = output_dir / f"{base_name}.json"
        npy_path = output_dir / f"{base_name}.npy"

        if json_path.exists():
            print(f"\n⏭️  Skipping batch_size={batch_size} (already exists)")
            continue

        try:
            adata_copy = adata.copy()
            results, embeddings = run_standard_inference(
                adata_copy,
                batch_size=batch_size,
                checkpoint=args.checkpoint,
                max_length=args.max_length,
            )

            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"✓ Saved results to {json_path}")

            if args.save_embeddings:
                np.save(npy_path, embeddings)
                print(f"✓ Saved embeddings to {npy_path}")

        except Exception as e:
            print(f"❌ Error with batch_size={batch_size}: {e}")
            continue

    print("\n" + "=" * 80)
    print(f"Standard inference sweep complete! Results: {output_dir}")
    print("=" * 80)


# ============================================================================
# vLLM Inference
# ============================================================================


async def fetch_embedding_async(
    session: aiohttp.ClientSession,
    server_url: str,
    model_name: str,
    cell_data: dict,
    cell_idx: int,
    semaphore: asyncio.Semaphore,
) -> tuple[int, np.ndarray]:
    """Fetch embedding for a single cell with concurrency control."""
    async with semaphore:
        rna_data = cell_data["multi_modal_data"]["rna"]
        payload = {
            "model": model_name,
            "data": {
                "gene_ids": rna_data["gene_ids"].tolist(),
                "expr_values": rna_data["expr_values"].tolist(),
                "attention_mask": rna_data["attention_mask"].tolist(),
            },
        }

        async with session.post(
            server_url, json=payload, timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(f"Server error {response.status}: {text}")

            result = await response.json()
            embedding = np.array(
                result.get("data", {}).get("embedding") or result.get("embedding")
            )
            return cell_idx, embedding


async def _run_vllm_inference_async(
    cell_data: list,
    server_url: str,
    model_name: str,
    concurrent_requests: int,
) -> np.ndarray:
    """Run vLLM inference with concurrent requests."""
    semaphore = asyncio.Semaphore(concurrent_requests)
    embeddings = [None] * len(cell_data)

    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_embedding_async(session, server_url, model_name, cell, i, semaphore)
            for i, cell in enumerate(cell_data)
        ]

        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            cell_idx, embedding = await task
            embeddings[cell_idx] = embedding

            if i % 100 == 0 or i == len(cell_data):
                print(f"  Processed {i}/{len(cell_data)} cells", end="\r")

    print()
    return np.array(embeddings)


def run_vllm_inference(
    adata: anndata.AnnData,
    concurrent_requests: int,
    model_name: str,
    server_host: str,
    server_port: str,
    max_length: int,
) -> tuple[dict, np.ndarray]:
    """Run vLLM inference and return timing results."""
    print(f"\nRunning vLLM inference with concurrent_requests={concurrent_requests}...")

    print("Preprocessing cells...")
    preprocess_start = time.time()
    tokenizer = load_tokenizer(model_name.replace(".vllm", ""))
    cell_data = preprocess_anndata(adata, tokenizer, max_length=max_length)
    preprocess_time = time.time() - preprocess_start
    print(f"Preprocessed {len(cell_data)} cells in {preprocess_time:.2f}s")

    print(f"Sending {len(cell_data)} requests with {concurrent_requests} concurrent...")
    server_url = f"http://{server_host}:{server_port}/pooling"
    inference_start = time.time()

    embeddings = asyncio.run(
        _run_vllm_inference_async(
            cell_data, server_url, model_name, concurrent_requests
        )
    )

    inference_time = time.time() - inference_start
    total_time = preprocess_time + inference_time

    results = {
        "method": "vllm",
        "concurrent_requests": concurrent_requests,
        "num_cells": adata.n_obs,
        "num_genes": adata.n_vars,
        "preprocess_time_seconds": preprocess_time,
        "inference_time_seconds": inference_time,
        "total_time_seconds": total_time,
        "time_per_cell_ms": (total_time / adata.n_obs) * 1000,
        "embedding_shape": list(embeddings.shape),
        "model_name": model_name,
        "max_length": max_length,
        "server_host": server_host,
        "server_port": server_port,
    }

    print(f"Completed in {total_time:.2f}s ({results['time_per_cell_ms']:.2f} ms/cell)")
    return results, embeddings


def cmd_vllm(args):
    """Run vLLM inference parameter sweep."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("vLLM Inference Parameter Sweep")
    print("=" * 80)
    adata = load_dataset(args.num_cells, args.num_genes)

    for concurrent_requests in args.concurrent_requests:
        cells_str = f"{adata.n_obs}" if args.num_cells > 0 else "all"
        genes_str = f"{adata.n_vars}" if args.num_genes > 0 else "all"
        base_name = (
            f"vllm_concurrent{concurrent_requests}_cells{cells_str}_genes{genes_str}"
        )
        json_path = output_dir / f"{base_name}.json"
        npy_path = output_dir / f"{base_name}.npy"

        if json_path.exists():
            print(
                f"\n⏭️  Skipping concurrent_requests={concurrent_requests} (already exists)"
            )
            continue

        try:
            results, embeddings = run_vllm_inference(
                adata,
                concurrent_requests=concurrent_requests,
                model_name=args.model_name,
                server_host=args.server_host,
                server_port=args.server_port,
                max_length=args.max_length,
            )

            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"✓ Saved results to {json_path}")

            if args.save_embeddings:
                np.save(npy_path, embeddings)
                print(f"✓ Saved embeddings to {npy_path}")

        except Exception as e:
            print(f"❌ Error with concurrent_requests={concurrent_requests}: {e}")
            continue

    print("\n" + "=" * 80)
    print(f"vLLM inference sweep complete! Results: {output_dir}")
    print("=" * 80)


# ============================================================================
# Plot Comparison
# ============================================================================


def load_experiment_results(experiment_dir: Path) -> tuple[dict, dict]:
    """Load all experiment results from JSON files."""
    standard_results = {}
    vllm_results = {}

    for json_file in experiment_dir.glob("*.json"):
        with open(json_file) as f:
            result = json.load(f)

        method = result.get("method")
        if method == "standard":
            batch_size = result["batch_size"]
            standard_results[batch_size] = result
        elif method == "vllm":
            concurrent_requests = result["concurrent_requests"]
            vllm_results[concurrent_requests] = result

    return standard_results, vllm_results


def plot_comparison(
    standard_results: dict,
    vllm_results: dict,
    output_path: Path,
):
    """Create comparison plot of time per cell vs parameter."""
    standard_params = sorted(standard_results.keys())
    standard_times = [standard_results[p]["time_per_cell_ms"] for p in standard_params]

    vllm_params = sorted(vllm_results.keys())
    vllm_times = [vllm_results[p]["time_per_cell_ms"] for p in vllm_params]

    # Calculate best speedup for title
    best_speedup = 1.0
    best_standard_time = 0
    best_vllm_time = 0
    if standard_results and vllm_results:
        best_standard_time = min(
            r["time_per_cell_ms"] for r in standard_results.values()
        )
        best_vllm_time = min(r["time_per_cell_ms"] for r in vllm_results.values())
        best_speedup = best_standard_time / best_vllm_time

    plt.figure(figsize=(12, 7))

    # Plot both methods
    plt.plot(
        standard_params,
        standard_times,
        marker="o",
        linewidth=2,
        markersize=8,
        label="Standard Inference",
        color="#2E86AB",
    )
    plt.plot(
        vllm_params,
        vllm_times,
        marker="s",
        linewidth=2,
        markersize=8,
        label="vLLM Inference",
        color="#A23B72",
    )

    # Add OOM marker at batch size 512
    oom_y_position = max(standard_times) * 1.05
    plt.plot(
        512,
        oom_y_position,
        marker="x",
        markersize=15,
        markeredgewidth=3,
        color="#2E86AB",
        linestyle="None",
    )
    plt.annotate(
        "Out of Memory",
        xy=(512, oom_y_position),
        xytext=(10, 5),
        textcoords="offset points",
        fontsize=9,
        color="#2E86AB",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
    )

    # Add batch size labels
    for param, time_val in zip(standard_params, standard_times):
        plt.annotate(
            f"batch={param}",
            xy=(param, time_val),
            xytext=(0, -15),
            textcoords="offset points",
            fontsize=8,
            color="#2E86AB",
            ha="center",
        )

    # Add concurrent request labels
    for param, time_val in zip(vllm_params, vllm_times):
        plt.annotate(
            f"concurrent={param}",
            xy=(param, time_val),
            xytext=(0, 10),
            textcoords="offset points",
            fontsize=8,
            color="#A23B72",
            ha="center",
        )

    # Formatting
    plt.xlabel("Batch Size / Concurrent Requests", fontsize=12, fontweight="bold")
    plt.ylabel("Time per Cell (ms)", fontsize=12, fontweight="bold")

    # Set x-axis ticks
    all_params = sorted(set(standard_params + vllm_params + [512]))
    plt.xticks(all_params, [str(p) for p in all_params], rotation=45)

    # Title with best speedup
    title = "BiomedRNA - Standard Batched Inference vs Concurrent vLLM Inference\n"
    title += f"Standard batched best: {best_standard_time:.1f}ms, Concurrent best: {best_vllm_time:.1f}ms ({best_speedup:.2f}x speedup)"
    plt.title(title, fontsize=12, fontweight="bold", pad=20)

    plt.legend(fontsize=11, loc="best")
    plt.grid(True, alpha=0.3, linestyle="--")

    # Use log scale for x-axis if range is large
    if (
        max(max(standard_params), max(vllm_params))
        / min(min(standard_params), min(vllm_params))
        > 10
    ):
        plt.xscale("log")
        plt.xlabel(
            "Batch Size / Concurrent Requests (log scale)",
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved plot to {output_path}")

    # Display summary
    print("\n" + "=" * 80)
    print("Performance Summary")
    print("=" * 80)

    if standard_results:
        print("\nStandard (PyTorch) Inference:")
        print(f"  Batch sizes tested: {sorted(standard_results.keys())}")
        best_standard = min(
            standard_results.items(), key=lambda x: x[1]["time_per_cell_ms"]
        )
        print(
            f"  Best: batch_size={best_standard[0]} → {best_standard[1]['time_per_cell_ms']:.2f} ms/cell"
        )

    if vllm_results:
        print("\nvLLM Inference:")
        print(f"  Concurrent requests tested: {sorted(vllm_results.keys())}")
        best_vllm = min(vllm_results.items(), key=lambda x: x[1]["time_per_cell_ms"])
        print(
            f"  Best: concurrent={best_vllm[0]} → {best_vllm[1]['time_per_cell_ms']:.2f} ms/cell"
        )

    if standard_results and vllm_results:
        print(f"\nOverall best speedup: {best_speedup:.2f}x (vLLM vs Standard)")

    print("=" * 80)


def cmd_plot(args):
    """Generate comparison plot from experiment results."""
    experiment_dir = Path(args.output_dir)
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    output_path = experiment_dir / "performance_comparison.png"

    print("=" * 80)
    print("Inference Performance Comparison")
    print("=" * 80)
    print(f"Loading experiments from: {experiment_dir}")

    standard_results, vllm_results = load_experiment_results(experiment_dir)

    if not standard_results and not vllm_results:
        print("\n❌ No experiment results found!")
        print("Run experiments first:")
        print("  python vllm/scripts/benchmark_inference.py standard")
        print("  python vllm/scripts/benchmark_inference.py vllm")
        return

    print(f"Found {len(standard_results)} standard experiments")
    print(f"Found {len(vllm_results)} vLLM experiments")

    plot_comparison(standard_results, vllm_results, output_path)


# ============================================================================
# Main CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark BiomedRNA inference performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Standard inference subcommand
    standard_parser = subparsers.add_parser(
        "standard", help="Run standard inference sweep"
    )
    standard_parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 16, 32, 64],
        help="Batch sizes to test (default: 1 8 16 32 64)",
    )
    standard_parser.add_argument(
        "--num-cells",
        type=int,
        default=10000,
        help="Number of cells (default: 10000)",
    )
    standard_parser.add_argument(
        "--num-genes",
        type=int,
        default=1000,
        help="Number of genes (default: 1000)",
    )
    standard_parser.add_argument(
        "--checkpoint",
        type=str,
        default="ibm-research/biomed.rna.llama.47m.wced.multitask.v1",
        help="Model checkpoint",
    )
    standard_parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Max sequence length (default: 1024)",
    )
    standard_parser.add_argument(
        "--output-dir",
        type=str,
        default="vllm/output",
        help="Output directory (default: vllm/output)",
    )
    standard_parser.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Save embeddings to .npy files (default: False)",
    )

    # vLLM inference subcommand
    vllm_parser = subparsers.add_parser("vllm", help="Run vLLM inference sweep")
    vllm_parser.add_argument(
        "--concurrent-requests",
        type=int,
        nargs="+",
        default=[1, 8, 16, 32, 64, 128, 256],
        help="Concurrent requests to test (default: 1 8 16 32 64 128 256)",
    )
    vllm_parser.add_argument(
        "--num-cells",
        type=int,
        default=10000,
        help="Number of cells (default: 10000)",
    )
    vllm_parser.add_argument(
        "--num-genes",
        type=int,
        default=1000,
        help="Number of genes (default: 1000)",
    )
    vllm_parser.add_argument(
        "--model-name",
        type=str,
        default="ibm-research/biomed.rna.llama.47m.wced.multitask.v1.vllm",
        help="Model name for vLLM server",
    )
    vllm_parser.add_argument(
        "--server-host",
        type=str,
        default="localhost",
        help="vLLM server host (default: localhost)",
    )
    vllm_parser.add_argument(
        "--server-port",
        type=str,
        default="8000",
        help="vLLM server port (default: 8000)",
    )
    vllm_parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Max sequence length (default: 1024)",
    )
    vllm_parser.add_argument(
        "--output-dir",
        type=str,
        default="vllm/output",
        help="Output directory (default: vllm/output)",
    )
    vllm_parser.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Save embeddings to .npy files (default: False)",
    )

    # Plot subcommand
    plot_parser = subparsers.add_parser("plot", help="Generate comparison plot")
    plot_parser.add_argument(
        "--output-dir",
        type=str,
        default="vllm/output",
        help="Directory with experiment results (default: vllm/output)",
    )

    args = parser.parse_args()

    if args.command == "standard":
        cmd_standard(args)
    elif args.command == "vllm":
        cmd_vllm(args)
    elif args.command == "plot":
        cmd_plot(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

# Made with Bob
