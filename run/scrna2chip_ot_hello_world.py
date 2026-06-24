"""Hello-world for scRNA→ChIP population-OT training.

Builds a tiny synthetic h5ad, runs a single training_step with the OT loss,
and prints the loss value.  No GPU needed.

Usage:
    python run/scrna2chip_ot_hello_world.py
"""

import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

GENES = ["DMBT1P1", "LINC01282", "LOC105377447", "AC104304.1", "ARMC4P1",
         "FRG2EP", "AL513188.1", "RP11-1134I14.4", "RP11-955H22.4", "AC005614.5",
         "BRCA1", "TP53", "EGFR", "MYC", "KRAS", "PTEN", "AKT1", "VEGFA",
         "CDH1", "VIM", "FN1", "ACTA2", "TGFB1", "IL6", "TNF", "IFNG",
         "CD3E", "CD4", "CD8A", "FOXP3", "CD19", "MS4A1", "CD14", "FCGR3A",
         "NKG7", "GNLY", "GZMB", "PRF1", "NCAM1", "KLRD1", "HBA1", "HBB",
         "GYPA", "ITGA2B", "GP1BA", "PECAM1", "VWF", "CDH5", "COL1A1", "COL3A1"]
N_SCRNA_PER_TISSUE = 20
N_CHIP_PER_TISSUE = 10
TISSUES = ["tissue_A", "tissue_B"]


def make_h5ad(path: Path) -> Path:
    rng = np.random.default_rng(0)
    n_genes = len(GENES)

    rows = []
    for tissue in TISSUES:
        for _ in range(N_SCRNA_PER_TISSUE):
            rows.append({"data_type": "scRNA", "tissue_label": tissue, "split_random": "train"})
        for _ in range(N_CHIP_PER_TISSUE):
            rows.append({"data_type": "ChIP", "tissue_label": tissue, "split_random": "train"})

    obs = pd.DataFrame(rows)
    obs.index = [f"cell_{i}" for i in range(len(obs))]
    X = sp.csr_matrix(rng.exponential(1.0, (len(obs), n_genes)).astype(np.float32))
    var = pd.DataFrame(index=GENES)
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.write_h5ad(path)
    print(f"Wrote {path}  shape={adata.shape}")
    return path


def main():
    from bmfm_targets.config import FieldInfo, SCBertConfig
    from bmfm_targets.config.training_config import TrainerConfig
    from bmfm_targets.datasets.scRNA2ChIP import ConcatDataModule, ConcatDataset
    from bmfm_targets.tokenization.load import load_tokenizer
    from bmfm_targets.training.modules.scrna_to_chip import ScrnaToChipTranslationModule

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        h5ad_path = make_h5ad(tmpdir / "data.h5ad")

        tokenizer = load_tokenizer("all_genes_v2")

        fields = [
            FieldInfo(field_name="genes"),
            FieldInfo(
                field_name="expressions",
                is_input=True,
                tokenization_strategy="continuous_value_encoder",
                encoder_kwargs={"kind": "mlp_with_special_token_embedding", "zero_as_special_token": True},
            ),
            FieldInfo(
                field_name="label_expressions",
                is_input=False,
                tokenization_strategy="continuous_value_encoder",
                decode_modes={"wced": {"vocab_field": "genes", "logit_outputs": ["mse"]}},
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

        trainer_config = TrainerConfig(
            losses=[],
            learning_rate=1e-4,
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
        )

        dm.setup("fit")

        loader = dm.train_dataloader()
        batch = next(iter(loader))

        assert "chip_population" in batch, "chip_population missing from batch"
        print(f"chip_population shape: {batch['chip_population'].shape}")
        print(f"input_ids shape:       {batch['input_ids'].shape}")

        module = ScrnaToChipTranslationModule(
            model_config,
            trainer_config,
            tokenizer=tokenizer,
            ot_weight=1.0,
            wced_weight=1.0,
        )

        loss = module.training_step(batch, 0)
        assert loss is not None, (
            "loss is None — no WCED logits produced. "
            "Check that label_expressions field has decode_modes: wced configured "
            "and that the model was built with the correct fields."
        )
        print(f"training_step loss: {loss.item():.4f}")
        print("Hello world OK")


if __name__ == "__main__":
    main()
