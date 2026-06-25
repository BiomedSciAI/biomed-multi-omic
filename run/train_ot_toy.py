"""
Local toy-dataset training run to verify OT (and optionally WCED) loss converges.

Builds a synthetic h5ad, runs pytorch-lightning Trainer for a fixed number of
steps, and prints the loss at each step.  Mirrors what the YAML configs do but
without Hydra, so it can be run directly:

    # OT-only
    PYTHONPATH=/u/dmichael/scRNA2ChIP python run/train_ot_toy.py

    # OT + WCED reconstruction
    PYTHONPATH=/u/dmichael/scRNA2ChIP python run/train_ot_toy.py --wced

Exit code is non-zero if loss does not decrease by >=20% over the run.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy.sparse as sp
from pytorch_lightning.callbacks import Callback

GENES = [
    "DMBT1P1",
    "LINC01282",
    "LOC105377447",
    "AC104304.1",
    "ARMC4P1",
    "FRG2EP",
    "AL513188.1",
    "RP11-1134I14.4",
    "RP11-955H22.4",
    "AC005614.5",
    "BRCA1",
    "TP53",
    "EGFR",
    "MYC",
    "KRAS",
    "PTEN",
    "AKT1",
    "VEGFA",
    "CDH1",
    "VIM",
    "FN1",
    "ACTA2",
    "TGFB1",
    "IL6",
    "TNF",
    "IFNG",
    "CD3E",
    "CD4",
    "CD8A",
    "FOXP3",
    "CD19",
    "MS4A1",
    "CD14",
    "FCGR3A",
    "NKG7",
    "GNLY",
    "GZMB",
    "PRF1",
    "NCAM1",
    "KLRD1",
    "HBA1",
    "HBB",
    "GYPA",
    "ITGA2B",
    "GP1BA",
    "PECAM1",
    "VWF",
    "CDH5",
    "COL1A1",
    "COL3A1",
]
N_SCRNA_PER_TISSUE = 40
N_CHIP_PER_TISSUE = 10
TISSUES = ["tissue_A", "tissue_B"]
MAX_STEPS = 60
LEARNING_RATE = 5e-4


def make_h5ad(path: Path) -> Path:
    rng = np.random.default_rng(0)
    rows = []
    for tissue in TISSUES:
        for _ in range(N_SCRNA_PER_TISSUE):
            rows.append(
                {"data_type": "scRNA", "tissue_label": tissue, "split_random": "train"}
            )
        for _ in range(N_CHIP_PER_TISSUE):
            rows.append(
                {"data_type": "ChIP", "tissue_label": tissue, "split_random": "train"}
            )
    obs = pd.DataFrame(rows)
    obs.index = [f"cell_{i}" for i in range(len(obs))]
    X = sp.csr_matrix(rng.exponential(1.0, (len(obs), len(GENES))).astype(np.float32))
    var = pd.DataFrame(index=GENES)
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.write_h5ad(path)
    return path


class StepLossLogger(Callback):
    """Prints train_loss at every step and records for convergence check."""

    def __init__(self):
        self.losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = trainer.callback_metrics.get("train/loss")
        ot_loss = trainer.callback_metrics.get("train/ot_loss")
        if loss is not None:
            self.losses.append(loss.item())
            ot_str = f"  ot_loss={ot_loss.item():.4f}" if ot_loss is not None else ""
            print(f"  step {batch_idx:3d}  loss={loss.item():.4f}{ot_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wced", action="store_true", help="Enable WCED reconstruction loss"
    )
    args = parser.parse_args()

    from bmfm_targets.config import FieldInfo, SCBertConfig
    from bmfm_targets.config.training_config import TrainerConfig
    from bmfm_targets.datasets.scRNA2ChIP import ConcatDataModule
    from bmfm_targets.tokenization.load import load_tokenizer
    from bmfm_targets.training.modules.scrna_to_chip import ScrnaToChipTranslationModule

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        h5ad_path = make_h5ad(tmpdir / "data.h5ad")
        print(
            f"Wrote toy h5ad  shape=({(N_SCRNA_PER_TISSUE + N_CHIP_PER_TISSUE) * len(TISSUES)}, {len(GENES)})"
        )

        tokenizer = load_tokenizer("all_genes_v2")

        fields = [
            FieldInfo(field_name="genes"),
            FieldInfo(
                field_name="expressions",
                is_input=True,
                tokenization_strategy="continuous_value_encoder",
                encoder_kwargs={
                    "kind": "mlp_with_special_token_embedding",
                    "zero_as_special_token": True,
                },
            ),
            FieldInfo(
                field_name="label_expressions",
                is_input=False,
                tokenization_strategy="continuous_value_encoder",
                decode_modes={
                    "wced": {"vocab_field": "genes", "logit_outputs": ["mse"]}
                },
            ),
        ]
        for field in fields:
            field.update_vocab_size(tokenizer)

        model_config = SCBertConfig(
            fields=fields,
            num_hidden_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            intermediate_size=128,
        )

        losses = []
        sequence_label_extractor = None
        if args.wced:
            from bmfm_targets.training.masking import WCEDMasker

            sequence_label_extractor = WCEDMasker(
                tokenizer=tokenizer,
                lookup_field_name="genes",
                value_field_name="label_expressions",
            )
            losses = [
                {
                    "field_name": "label_expressions",
                    "name": "mse",
                    "wced_target": "all_genes",
                }
            ]

        trainer_config = TrainerConfig(
            losses=losses,
            learning_rate=LEARNING_RATE,
            enable_perturbation_metrics=False,
        )

        dm = ConcatDataModule(
            tokenizer=tokenizer,
            fields=fields,
            batch_size=8,
            num_workers=0,
            max_length=64,
            collation_strategy="sequence_labeling",
            dataset_kwargs={
                "processed_data_source": str(h5ad_path),
                "label_columns": ["tissue_label"],
                "new_field": "label_expressions",
            },
            transform_datasets=False,
            transform_kwargs={"split_column_name": "split_random"},
            limit_genes=None,
            use_ot_batching=True,
            celltype_column="tissue_label",
            sequence_label_extractor=sequence_label_extractor,
        )
        dm.setup("fit")

        module = ScrnaToChipTranslationModule(
            model_config,
            trainer_config,
            tokenizer=tokenizer,
            ot_weight=1.0,
            wced_weight=1.0 if args.wced else 0.0,
        )

        loss_logger = StepLossLogger()
        pl_trainer = pl.Trainer(
            max_steps=MAX_STEPS,
            accelerator="cpu",
            log_every_n_steps=1,
            enable_checkpointing=False,
            enable_model_summary=False,
            logger=False,
            callbacks=[loss_logger],
        )

        mode = "OT + WCED" if args.wced else "OT-only"
        print(f"\n--- {mode} training ({MAX_STEPS} steps, lr={LEARNING_RATE}) ---")
        pl_trainer.fit(module, datamodule=dm)

        losses_recorded = loss_logger.losses
        if len(losses_recorded) < 2:
            print("ERROR: fewer than 2 loss values recorded")
            sys.exit(1)

        first = np.mean(losses_recorded[:5])
        last = np.mean(losses_recorded[-5:])
        pct_drop = 100 * (first - last) / abs(first)
        print(
            f"\nfirst-5 avg: {first:.4f}  last-5 avg: {last:.4f}  drop: {pct_drop:.1f}%"
        )

        if pct_drop < 20:
            print(f"FAIL: loss did not drop >=20% (got {pct_drop:.1f}%)")
            sys.exit(1)

        print(
            f"PASS: {mode} loss converged ({pct_drop:.1f}% drop over {MAX_STEPS} steps)"
        )


if __name__ == "__main__":
    main()
