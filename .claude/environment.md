# Environment — biomed-multi-omic (public fork)

## Key facts

- **Package name**: `bmfm-targets` (same as inner-source repo at `/u/dmichael/bmfm-targets`)
- **Entry point**: `bmfm-targets-run` → `bmfm_targets.tasks.scbert.scbert_main:main`
  - The inner-source repo uses `bmfm-targets-scbert` for the same entry point — **different binary name**
- **Python**: 3.13 works (env `biomedmultiomic`, below). The scipy/torchmetrics
  breakage is specific to `conda_targets`, not to 3.13 in general.
- **Configs live in `run/`** — top-level `run/` directory, not `bmfm_targets/tasks/scbert/`
  - Use `-cd run -cn <config>` (or just `-cn <config>` if Hydra's config path is set to `run/`)

## Name clash with inner-source

Both repos install as `bmfm-targets`, so only one version can own that import name
per env. **All three pre-existing envs map `bmfm-targets` to the inner-source
checkout at `/u/dmichael/bmfm-targets`** (editable install):

- `conda_targets` (py3.13, broken scipy/numpy) — inner-source
- `targets` (py3.11) — inner-source
- `targets3-13` (py3.13) — inner-source

**Do not install this repo into any of those.** For this repo use `biomedmultiomic`
(below), which is dedicated to `/u/dmichael/biomed-multi-omic`.

> Note: running `pytest` from this repo's root imports `bmfm_targets` from the repo
> dir via sys.path even in an inner-source env, which is why tests can appear to pass
> there — but the entry point (`bmfm-targets-run`) and any installed console scripts
> still resolve to the inner-source package. Use `biomedmultiomic` to avoid ambiguity.

## Working env for this repo: `biomedmultiomic`

Conda env, Python 3.13, at `/proj/bmfm/users/dmichael/envs/biomedmultiomic`.
Created dedicated to this repo (avoids the inner-source clash above).

```bash
conda activate /proj/bmfm/users/dmichael/envs/biomedmultiomic
```

Installed so far:
- `pre-commit` (4.6.0) — run hooks with `pre-commit run --files <paths>` or
  `pre-commit run --all-files`. Hooks (ruff, black, etc.) run in their own isolated
  envs, so this does **not** require the repo's deps to be installed.

Not yet installed: this repo itself. To make `bmfm-targets` / `bmfm-targets-run`
resolve to **this** repo (and enable full test runs), install it editable here:

```bash
cd /u/dmichael/biomed-multi-omic
/proj/bmfm/users/dmichael/envs/biomedmultiomic/bin/pip install -e ".[test]"
```

Verify:
```bash
bmfm-targets-run --help
python -c "import bmfm_targets, os; print(os.path.realpath(bmfm_targets.__file__))"
# should print a path under /u/dmichael/biomed-multi-omic
```

## h2 secondary home

```bash
alias h2='cd /proj/bmfm/users/dmichael'
```

Path: `/proj/bmfm/users/dmichael/`

Existing envs (see "Name clash" above — the first three are inner-source):
- `biomedmultiomic` — **this repo**, Python 3.13 (conda). Has `pre-commit`.
- `conda_targets` — inner-source bmfm-targets, Python 3.13, **broken** (scipy/numpy compat)
- `targets` — inner-source, Python 3.11
- `targets3-13` — inner-source, Python 3.13

Data in h2:
- `adata_cache/` — per-cell-type h5ad files from CellXGene
- `benchmarking/` — evaluation data

## scconcept branch

Branch: `origin/scconcept-from-main`

New files vs main:
- `bmfm_targets/training/data_module.py` — adds `ContrastiveDataModule`
- `bmfm_targets/datasets/paired_view_collator.py` — `_PairedViewCollator` (gene-panel splits)
- `bmfm_targets/datasets/samplers.py` — `ConditionHomogeneousBatchSampler`
- `bmfm_targets/training/losses/objectives.py` — `ContrastiveObjective` (InfoNCE/CLIP)
- `bmfm_targets/training/losses/sources.py` — `CellEmbeddingContrastiveSource`
- `bmfm_targets/training/callbacks.py` — `ContrastiveMetricsCallback`
- `bmfm_targets/models/predictive/layers.py` — `ContrastiveHead` in `SCBaseFieldDecoder`
- `bmfm_targets/tests/test_contrastive_loss.py` — unit tests
- `run/scconcept_pretrain.yaml` — training config
- `run/extract_embeddings.py` — extract CLS + contrastive embeddings, compare to PCA
- `run/evaluate_representations.py` — KNN accuracy comparison
- `docs/scconcept_results.md` — evidence the approach works (PBMC3k, Recall@1=38.5%)

## Training data

Primary test dataset: PBMC3k (2700 cells, 9 cell types, 13714 genes)
- Expected at `data/real/pbmc3k.h5ad` relative to repo root

The scconcept config (`run/scconcept_pretrain.yaml`) currently hardcodes
`/Users/dmichael/github/biomed-multi-omic/data/real/pbmc3k.h5ad` — override with:
```bash
bmfm-targets-run -cd run -cn scconcept_pretrain \
    data_module.dataset_kwargs.processed_data_source=<path>/pbmc3k.h5ad
```

## Key config knobs (scconcept_pretrain.yaml)

- `data_module.condition_columns: [celltype]` — column used for homogeneous batch sampling
- `data_module.n_cells_per_batch` / `n_cells_per_batch` — batch size (keep in sync)
- `data_module.max_length` — token budget per cell (64 in config, 256 used in results doc)
- `trainer.losses[0].both_batch: true` — 2N×2N combined InfoNCE (scConcept steady-state)
- `model.checkpoint` — backbone; defaults to `ibm-research/biomed.rna.bert.110m.wced.multitask.v1`

## Input data policy

All checkpoints expect **raw integer counts** (normalized internally to CPM10k+log1p).
PBMC3k from 10x Genomics is raw counts — no preprocessing needed.

## Ablation training yamls (in run/)

Three-way comparison, all 10 epochs on 1% CellXGene:

| File | Backbone | Loss | Question |
|---|---|---|---|
| `wced_1pct_10ep.yaml` | WCED gene-error ckpt | WCED only | baseline |
| `scconcept_1pct_10ep.yaml` | WCED multitask v1 | contrastive only | does scConcept help? |
| `scconcept_scratch_1pct_10ep.yaml` | random init | contrastive only | is WCED backbone needed? |

Run all three, then benchmark each with:
```bash
bash bmfm_targets/evaluation/benchmark_configs/benchmark_run.sh \
    --name=<task_name> --ckpt=<best .ckpt> --do_zs=true --do_ft=true
```

WCED baseline checkpoint:
`/proj/bmfm/users/dmichael/training_runs/llama_medium_wced_gene_error/epoch=10-step=90036-val_loss=0.14.ckpt`

## benchmark_run.sh

Uses `bmfm-targets-run` (corrected from `bmfm-targets-scbert` which is the inner-source binary).
Config dir: `bmfm_targets/evaluation/benchmark_configs/`
Currently runs `multiple_sclerosis` and `myeloid` datasets by default.

## Running tests

```bash
conda activate /proj/bmfm/users/dmichael/envs/biomedmultiomic  # after `pip install -e ".[test]"`
cd /u/dmichael/biomed-multi-omic
pytest bmfm_targets/tests/test_contrastive_loss.py -v
```
