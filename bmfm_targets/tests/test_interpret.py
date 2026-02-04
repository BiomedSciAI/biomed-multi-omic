import os.path
import tempfile
from pathlib import Path

import pandas.testing
import pytest
import torch

from bmfm_targets import config
from bmfm_targets.datasets import zheng68k
from bmfm_targets.evaluation import interpret
from bmfm_targets.tasks import task_utils
from bmfm_targets.tests import helpers
from bmfm_targets.tokenization import load_tokenizer
from bmfm_targets.training.modules import SequenceClassificationTrainingModule


def test_interpret_run(pl_data_module_zheng68k_seq_cls):
    model_config, data_module = prepare_interpret_run_on_fresh_model(
        pl_data_module_zheng68k_seq_cls
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = config.InterpretTaskConfig(
            accelerator="cpu", max_epochs=1, default_root_dir=tmpdir
        )
        pl_trainer = task_utils.make_trainer_for_task(task_config)
        task_utils.interpret_run(pl_trainer, task_config, data_module, model_config)
        attributions = interpret.load_attributions(Path(tmpdir + "/attributions.json"))
        assert len(attributions) == 3
        assert all(a.name is not None for a in attributions)
        assert all(
            a.label_column_attributions[0].label_attributions[0].label_name is not None
            for a in attributions
        )

        attr_df = interpret.join_sample_attributions(attributions)
        mean_attr_df = interpret.get_mean_attributions(attr_df)
        assert mean_attr_df.shape[0] > 10


def test_interpret_run_regression(gene2vec_fields):
    from bmfm_targets.datasets.sciplex3 import SciPlex3DataModule

    from .helpers import SciPlex3Paths

    tokenizer = load_tokenizer("gene2vec")

    label_columns = [
        config.LabelColumnInfo(
            label_column_name="size_factor", is_regression_label=True
        )
    ]

    dm = SciPlex3DataModule(
        dataset_kwargs={
            "data_dir": SciPlex3Paths.root,
            "split_column": "split_random",
        },
        tokenizer=tokenizer,
        fields=gene2vec_fields,
        label_columns=label_columns,
        mlm=False,
        collation_strategy="sequence_classification",
        batch_size=1,
        max_length=32,
        limit_dataset_samples=3,
    )
    dm.prepare_data()
    dm.setup("predict")
    helpers.update_label_columns(dm.label_columns, dm.label_dict)
    model_config = config.SCBertConfig(
        fields=gene2vec_fields,
        label_columns=dm.label_columns,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        hidden_size=32,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = config.InterpretTaskConfig(
            accelerator="cpu", max_epochs=1, default_root_dir=tmpdir
        )
        pl_trainer = task_utils.make_trainer_for_task(task_config)
        task_utils.interpret_run(pl_trainer, task_config, dm, model_config)
        attributions = interpret.load_attributions(Path(tmpdir + "/attributions.json"))
        assert len(attributions) == 3
        attr_df = interpret.join_sample_attributions(attributions)
        mean_attr_df = interpret.get_mean_attributions(attr_df)
        assert mean_attr_df.shape[0] > 8


def test_interpret_run_with_filtering_attribute(pl_data_module_zheng68k_seq_cls):
    model_config, data_module = prepare_interpret_run_on_fresh_model(
        pl_data_module_zheng68k_seq_cls
    )

    filter = {"celltype": ["CD19+ B", "CD34+"]}
    # len(attributions[0].label_column_attributions[0].label_attributions)
    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = config.InterpretTaskConfig(
            accelerator="cpu",
            max_epochs=1,
            default_root_dir=tmpdir,
            attribute_filter=filter,
        )
        pl_trainer = task_utils.make_trainer_for_task(task_config)
        task_utils.interpret_run(pl_trainer, task_config, data_module, model_config)
        attributions = interpret.load_attributions(Path(tmpdir + "/attributions.json"))
        assert len(attributions[0].label_column_attributions[0].label_attributions) == 2


def test_interpret_run_with_bad_attribute_kwargs_raises_error(
    pl_data_module_zheng68k_seq_cls,
):
    model_config, data_module = prepare_interpret_run_on_fresh_model(
        pl_data_module_zheng68k_seq_cls
    )

    # this is the simplest way to test that the kwargs are actually propagated
    attr_kwargs = {"bad_arg_name": "bad_val"}
    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = config.InterpretTaskConfig(
            accelerator="cpu",
            max_epochs=1,
            default_root_dir=tmpdir,
            attribute_kwargs=attr_kwargs,
        )
        pl_trainer = task_utils.make_trainer_for_task(task_config)
        with pytest.raises(Exception, match="unexpected keyword argument"):
            task_utils.interpret_run(pl_trainer, task_config, data_module, model_config)


def prepare_interpret_run_on_fresh_model(pl_data_module_zheng68k_seq_cls):
    model_config = config.SCBertConfig(
        fields=pl_data_module_zheng68k_seq_cls.fields,
        label_columns=pl_data_module_zheng68k_seq_cls.label_columns,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        hidden_size=32,
    )
    tokenizer = load_tokenizer("gene2vec")
    data_module = zheng68k.Zheng68kDataModule(
        dataset_kwargs={
            "processed_data_source": pl_data_module_zheng68k_seq_cls.processed_data_file
        },
        transform_datasets=False,
        tokenizer=tokenizer,
        fields=pl_data_module_zheng68k_seq_cls.fields,
        label_columns=pl_data_module_zheng68k_seq_cls.label_columns,
        collation_strategy="sequence_classification",
        batch_size=1,
        max_length=32,
        limit_dataset_samples=3,
    )
    data_module.setup("predict")
    return model_config, data_module


@pytest.fixture()
def dummy_attributions():
    sample_attributions = [
        {
            "name": "cell1",
            "label_column_attributions": [
                {
                    "gt_label": "blue",
                    "pred_label": "green",
                    "label_column_name": "color",
                    "label_attributions": [
                        {
                            "label_name": "blue",
                            "attributions": [("ABC", 1.0), ("DEF", 4.5)],
                        },
                        {
                            "label_name": "green",
                            "attributions": [("ABC", 0.5), ("GHI", 3.5)],
                        },
                    ],
                }
            ],
        },
        {
            "name": "cell2",
            "label_column_attributions": [
                {
                    "gt_label": "green",
                    "pred_label": "green",
                    "label_column_name": "color",
                    "label_attributions": [
                        {
                            "label_column_name": "color",
                            "label_name": "blue",
                            "attributions": [("FHE", 1.4), ("GHI", 2.5)],
                        },
                        {
                            "label_column_name": "color",
                            "label_name": "green",
                            "attributions": [("ABD", 0.4), ("GHI", 1.1)],
                        },
                    ],
                }
            ],
        },
    ]
    return [interpret.SampleAttribution(**k) for k in sample_attributions]


def test_save_attributions_json(dummy_attributions):
    with tempfile.TemporaryDirectory() as tmpdir:
        ofname = Path(tmpdir) / "attrs.json"
        interpret.save_sample_attributions(dummy_attributions, ofname)
        loaded = interpret.load_attributions(ofname)
        assert loaded == dummy_attributions


def test_save_attributions_nonjson_not_implemented(dummy_attributions):
    with pytest.raises(NotImplementedError):
        interpret.save_sample_attributions(dummy_attributions, "attrs.h5ad")


def test_instantiate_module_from_seq_cls_ckpt(zheng_seq_cls_ckpt):
    tokenizer = load_tokenizer(os.path.dirname(zheng_seq_cls_ckpt))
    module = interpret.SequenceClassificationAttributionModule.load_from_checkpoint(
        zheng_seq_cls_ckpt,
        tokenizer=tokenizer,
        weights_only=False,
    )
    seq_cls_module = SequenceClassificationTrainingModule.load_from_checkpoint(
        zheng_seq_cls_ckpt, weights_only=False
    )
    attr_model = module.model
    seq_cls_model = seq_cls_module.model
    for t, l in zip(attr_model.named_parameters(), seq_cls_model.named_parameters()):
        torch.testing.assert_close(t[1], l[1])


def test_interpret_run_from_seq_cls_ckpt(
    pl_data_module_zheng68k_seq_cls, zheng_seq_cls_ckpt
):
    _, data_module = prepare_interpret_run_on_fresh_model(
        pl_data_module_zheng68k_seq_cls
    )
    model_config = None
    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = config.InterpretTaskConfig(
            accelerator="cpu",
            max_epochs=1,
            default_root_dir=tmpdir,
            checkpoint=zheng_seq_cls_ckpt,
        )
        pl_trainer = task_utils.make_trainer_for_task(task_config)
        task_utils.interpret_run(pl_trainer, task_config, data_module, model_config)
        attributions = interpret.load_attributions(Path(tmpdir + "/attributions.json"))
        assert len(attributions) == 3
        attr_df = interpret.join_sample_attributions(attributions)
        mean_attr_df = interpret.get_mean_attributions(attr_df)
        assert mean_attr_df.shape[0] > 10


def test_repeated_interpret_run_are_identical(
    pl_data_module_zheng68k_seq_cls, zheng_seq_cls_ckpt
):
    _, data_module = prepare_interpret_run_on_fresh_model(
        pl_data_module_zheng68k_seq_cls
    )
    model_config = None
    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = config.InterpretTaskConfig(
            accelerator="cpu",
            max_epochs=1,
            default_root_dir=tmpdir,
            checkpoint=zheng_seq_cls_ckpt,
        )
        pl_trainer = task_utils.make_trainer_for_task(task_config)
        task_utils.interpret_run(pl_trainer, task_config, data_module, model_config)
        attributions = interpret.load_attributions(Path(tmpdir + "/attributions.json"))
        attr_df = interpret.join_sample_attributions(attributions)
        mean_attr_df = interpret.get_mean_attributions(attr_df)

        pl_trainer = task_utils.make_trainer_for_task(task_config)
        task_utils.interpret_run(pl_trainer, task_config, data_module, model_config)
        attributions2 = interpret.load_attributions(Path(tmpdir + "/attributions.json"))
        attr_df2 = interpret.join_sample_attributions(attributions2)
        mean_attr_df2 = interpret.get_mean_attributions(attr_df2)

        pandas.testing.assert_frame_equal(mean_attr_df, mean_attr_df2)


def test_interpret_run_using_different_data_from_ckpt(
    pl_data_module_zheng68k_seq_cls, zheng_seq_cls_ckpt
):
    from bmfm_targets.datasets import sciplex3

    model_config = None
    label_columns = ["target", "cell_type"]
    dm = sciplex3.SciPlex3DataModule(
        dataset_kwargs={
            "data_dir": helpers.SciPlex3Paths.root,
            "split_column": "split_random",
            "label_columns": label_columns,
        },
        tokenizer=pl_data_module_zheng68k_seq_cls.tokenizer,
        fields=pl_data_module_zheng68k_seq_cls.fields,
        batch_size=3,
        max_length=8,
        pad_to_multiple_of=2,
        limit_dataset_samples=3,
        collation_strategy="sequence_classification",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = config.InterpretTaskConfig(
            accelerator="cpu",
            default_root_dir=tmpdir,
            checkpoint=zheng_seq_cls_ckpt,
            attribute_kwargs={"n_steps": 1},
        )
        dm.setup(task_config.setup_stage)
        pl_trainer = task_utils.make_trainer_for_task(task_config)
        task_utils.interpret_run(pl_trainer, task_config, dm, model_config)
        attributions = interpret.load_attributions(Path(tmpdir + "/attributions.json"))
        assert len(attributions) == 3
        attr_df = interpret.join_sample_attributions(attributions)
        mean_attr_df = interpret.get_mean_attributions(attr_df)
        assert mean_attr_df.shape[1] == 4
        gene_vocab = dm.tokenizer.get_field_vocab("genes")
        assert all(i in gene_vocab for i in mean_attr_df.index)


def test_interpret_run_from_seq_cls_multitask_ckpt(sciplex3_seqcls_mt_model_and_ckpt):
    from bmfm_targets.datasets import sciplex3

    trained_model, ckpt = sciplex3_seqcls_mt_model_and_ckpt
    model_config = None
    label_columns = ["target", "cell_type"]
    dm = sciplex3.SciPlex3DataModule(
        dataset_kwargs={
            "data_dir": helpers.SciPlex3Paths.root,
            "split_column": "split_random",
            "label_columns": label_columns,
        },
        tokenizer=load_tokenizer("gene2vec"),
        fields=trained_model.config.fields,
        batch_size=3,
        max_length=8,
        pad_to_multiple_of=2,
        limit_dataset_samples={"train": 2, "dev": 2, "predict": 3},
        collation_strategy="sequence_classification",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = config.InterpretTaskConfig(
            accelerator="cpu",
            default_root_dir=tmpdir,
            checkpoint=ckpt,
            attribute_kwargs={"n_steps": 1},
        )
        dm.setup(task_config.setup_stage)
        pl_trainer = task_utils.make_trainer_for_task(task_config)
        task_utils.interpret_run(pl_trainer, task_config, dm, model_config)
        attributions = interpret.load_attributions(Path(tmpdir + "/attributions.json"))
        assert len(attributions) == 3
        attr_df = interpret.join_sample_attributions(attributions)
        mean_attr_df = interpret.get_mean_attributions(attr_df)
        assert mean_attr_df.shape[1] == 4
        gene_vocab = dm.tokenizer.get_field_vocab("genes")
        assert all(i in gene_vocab for i in mean_attr_df.index)


def test_interpret_run_from_multitask_ckpt(sciplex3_mt_model_and_ckpt):
    from bmfm_targets.datasets import sciplex3

    trained_model, ckpt = sciplex3_mt_model_and_ckpt
    model_config = None
    label_columns = ["target", "cell_type"]
    dm = sciplex3.SciPlex3DataModule(
        dataset_kwargs={
            "data_dir": helpers.SciPlex3Paths.root,
            "split_column": "split_random",
            "label_columns": label_columns,
        },
        tokenizer=load_tokenizer("gene2vec"),
        fields=trained_model.config.fields,
        batch_size=3,
        max_length=8,
        pad_to_multiple_of=2,
        limit_dataset_samples={"train": 2, "dev": 2, "predict": 3},
        collation_strategy="multitask",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = config.InterpretTaskConfig(
            accelerator="cpu",
            default_root_dir=tmpdir,
            checkpoint=ckpt,
            attribute_kwargs={"n_steps": 1},
        )
        dm.setup(task_config.setup_stage)
        pl_trainer = task_utils.make_trainer_for_task(task_config)
        task_utils.interpret_run(pl_trainer, task_config, dm, model_config)
        attributions = interpret.load_attributions(Path(tmpdir + "/attributions.json"))
        assert len(attributions) == 3
        attr_df = interpret.join_sample_attributions(attributions)
        mean_attr_df = interpret.get_mean_attributions(attr_df)
        assert mean_attr_df.shape[1] == 4
        gene_vocab = dm.tokenizer.get_field_vocab("genes")
        assert all(i in gene_vocab for i in mean_attr_df.index)
