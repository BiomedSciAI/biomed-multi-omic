from bmfm_targets import config
from bmfm_targets.datasets.dnaseq import (
    DNASeqChromatinProfileDataModule,
    DNASeqCorePromoterDataModule,
    DNASeqCovidDataModule,
    DNASeqDrosophilaEnhancerDataModule,
    DNASeqEpigeneticMarksDataModule,
    DNASeqMPRADataModule,
    DNASeqPromoterDataModule,
    DNASeqSpliceSiteDataModule,
    DNASeqTranscriptionFactorDataModule,
    StreamingDNASeqChromatinProfileDataModule,
)
from bmfm_targets.tests import helpers
from bmfm_targets.tokenization import (
    load_tokenizer,
)


def test_core_promoter_datamodule(dnaseq_fields):
    dnaseq_dataset_kwargs = {
        "processed_data_source": helpers.DNASeqCorePromoterPaths.processed_data_source,
        "label_dict_path": helpers.DNASeqCorePromoterPaths.label_dict_path,
        "num_workers": 0,
    }
    label_columns = [
        config.LabelColumnInfo(label_column_name="label", n_unique_values=2)
    ]
    tokenizer = load_tokenizer("ref2vec")
    pl_data_module_dnaseq_core_promoter = DNASeqCorePromoterDataModule(
        tokenizer=tokenizer,
        batch_size=2,
        fields=dnaseq_fields,
        label_columns=label_columns,
        collation_strategy="sequence_classification",
        max_length=512,
        dataset_kwargs=dnaseq_dataset_kwargs,
    )
    pl_data_module_dnaseq_core_promoter.setup("fit")

    for batch in pl_data_module_dnaseq_core_promoter.train_dataloader():
        assert tuple(batch["attention_mask"].shape) == (2, 512)
        assert tuple(batch["input_ids"].shape) == (2, 1, 512)
        assert tuple(batch["labels"]["label"].shape) == (2,)
        assert int(batch["input_ids"][0][0][0]) == 3
        break


def test_promoter_datamodule(dnaseq_fields):
    dnaseq_dataset_kwargs = {
        "processed_data_source": helpers.DNASeqPromoterPaths.processed_data_source,
        "label_dict_path": helpers.DNASeqPromoterPaths.label_dict_path,
        "num_workers": 0,
    }
    label_columns = [
        config.LabelColumnInfo(label_column_name="promoter_presence", n_unique_values=2)
    ]
    tokenizer = load_tokenizer("ref2vec")
    pl_data_module_dnaseq_promoter = DNASeqPromoterDataModule(
        tokenizer=tokenizer,
        batch_size=2,
        fields=dnaseq_fields,
        label_columns=label_columns,
        collation_strategy="sequence_classification",
        max_length=512,
        dataset_kwargs=dnaseq_dataset_kwargs,
    )
    pl_data_module_dnaseq_promoter.setup("fit")

    for batch in pl_data_module_dnaseq_promoter.train_dataloader():
        assert tuple(batch["attention_mask"].shape) == (2, 512)
        assert tuple(batch["input_ids"].shape) == (2, 1, 512)
        assert tuple(batch["labels"]["promoter_presence"].shape) == (2,)
        assert int(batch["input_ids"][0][0][0]) == 3
        break


def test_streaming_chromatin_profile_datamodule(dnaseq_fields):
    dnaseq_dataset_kwargs = {
        "processed_data_source": helpers.StreamingDNASeqChromatinProfilePaths.processed_data_source,
        # "label_dict_path": helpers.DNASeqChromatinProfilePaths.label_dict_path,
        # "num_workers": 0,
    }

    # label_columns = [
    #     config.LabelColumnInfo(label_column_name="dnase_0", n_unique_values=2),
    #     config.LabelColumnInfo(label_column_name="dnase_1", n_unique_values=2),
    #     config.LabelColumnInfo(label_column_name="dnase_2", n_unique_values=2),
    # ]

    label_columns = [
        config.LabelColumnInfo(label_column_name="dnase_" + str(i), n_unique_values=2)
        for i in range(125)
    ]
    label_columns += [
        config.LabelColumnInfo(label_column_name="tf_" + str(i), n_unique_values=2)
        for i in range(125, 815)
    ]
    label_columns += [
        config.LabelColumnInfo(label_column_name="histone_" + str(i), n_unique_values=2)
        for i in range(815, 919)
    ]

    tokenizer = load_tokenizer("ref2vec")
    pl_data_module_dnaseq_promoter = StreamingDNASeqChromatinProfileDataModule(
        tokenizer=tokenizer,
        batch_size=2,
        fields=dnaseq_fields,
        label_columns=label_columns,
        collation_strategy="sequence_classification",
        max_length=512,
        dataset_kwargs=dnaseq_dataset_kwargs,
    )
    pl_data_module_dnaseq_promoter.setup("fit")

    for batch in pl_data_module_dnaseq_promoter.train_dataloader():
        assert tuple(batch["attention_mask"].shape) == (2, 512)
        assert tuple(batch["input_ids"].shape) == (2, 1, 512)
        assert tuple(batch["labels"]["dnase_0"].shape) == (2,)
        assert tuple(batch["labels"]["dnase_1"].shape) == (2,)
        assert tuple(batch["labels"]["dnase_2"].shape) == (2,)
        assert int(batch["input_ids"][0][0][0]) == 3
        break


def test_chromatin_profile_datamodule(dnaseq_fields):
    dnaseq_dataset_kwargs = {
        "processed_data_source": helpers.DNASeqChromatinProfilePaths.processed_data_source,
        "label_dict_path": helpers.DNASeqChromatinProfilePaths.label_dict_path,
        "num_workers": 0,
    }

    label_columns = [
        config.LabelColumnInfo(label_column_name="dnase_0", n_unique_values=2),
        config.LabelColumnInfo(label_column_name="dnase_1", n_unique_values=2),
        config.LabelColumnInfo(label_column_name="dnase_2", n_unique_values=2),
    ]
    tokenizer = load_tokenizer("ref2vec")
    pl_data_module_dnaseq_promoter = DNASeqChromatinProfileDataModule(
        tokenizer=tokenizer,
        batch_size=2,
        fields=dnaseq_fields,
        label_columns=label_columns,
        collation_strategy="sequence_classification",
        max_length=512,
        dataset_kwargs=dnaseq_dataset_kwargs,
    )
    pl_data_module_dnaseq_promoter.setup("fit")

    for batch in pl_data_module_dnaseq_promoter.train_dataloader():
        assert tuple(batch["attention_mask"].shape) == (2, 512)
        assert tuple(batch["input_ids"].shape) == (2, 1, 512)
        assert tuple(batch["labels"]["dnase_0"].shape) == (2,)
        assert tuple(batch["labels"]["dnase_1"].shape) == (2,)
        assert tuple(batch["labels"]["dnase_2"].shape) == (2,)
        assert int(batch["input_ids"][0][0][0]) == 3
        break


def test_covid_datamodule(dnaseq_fields):
    dnaseq_dataset_kwargs = {
        "processed_data_source": helpers.DNASeqCovidPaths.processed_data_source,
        "label_dict_path": helpers.DNASeqCovidPaths.label_dict_path,
        "num_workers": 0,
    }
    label_columns = [
        config.LabelColumnInfo(
            label_column_name="label", n_unique_values=1, is_regression_label=True
        )
    ]
    tokenizer = load_tokenizer("ref2vec")
    pl_data_module_covid = DNASeqCovidDataModule(
        tokenizer=tokenizer,
        batch_size=2,
        fields=dnaseq_fields,
        label_columns=label_columns,
        collation_strategy="sequence_classification",
        max_length=512,
        dataset_kwargs=dnaseq_dataset_kwargs,
    )
    pl_data_module_covid.setup("fit")

    for batch in pl_data_module_covid.train_dataloader():
        assert tuple(batch["attention_mask"].shape) == (2, 512)
        assert tuple(batch["input_ids"].shape) == (2, 1, 512)
        assert tuple(batch["labels"]["label"].shape) == (2,)
        assert int(batch["input_ids"][0][0][0]) == 3
        break


def test_drosophila_enhancer_module(dnaseq_fields):
    dnaseq_dataset_kwargs = {
        "processed_data_source": helpers.DNASeqDrosophilaEnhancerPaths.processed_data_source,
        "label_dict_path": helpers.DNASeqDrosophilaEnhancerPaths.label_dict_path,
        "num_workers": 0,
    }
    label_columns = [
        config.LabelColumnInfo(
            label_column_name="Dev_log2_enrichment",
            n_unique_values=1,
            is_regression_label=True,
        )
    ]
    tokenizer = load_tokenizer("ref2vec")
    pl_data_module_drosophila = DNASeqDrosophilaEnhancerDataModule(
        tokenizer=tokenizer,
        batch_size=2,
        fields=dnaseq_fields,
        label_columns=label_columns,
        collation_strategy="sequence_classification",
        max_length=512,
        dataset_kwargs=dnaseq_dataset_kwargs,
    )
    pl_data_module_drosophila.setup("fit")

    for batch in pl_data_module_drosophila.train_dataloader():
        assert tuple(batch["attention_mask"].shape) == (2, 512)
        assert tuple(batch["input_ids"].shape) == (2, 1, 512)
        assert tuple(batch["labels"]["Dev_log2_enrichment"].shape) == (2,)
        assert int(batch["input_ids"][0][0][0]) == 3
        break


def test_epigenetic_marks_module(dnaseq_fields):
    dnaseq_dataset_kwargs = {
        "processed_data_source": helpers.DNASeqEpigeneticMarksPaths.processed_data_source,
        "label_dict_path": helpers.DNASeqEpigeneticMarksPaths.label_dict_path,
        "num_workers": 0,
    }
    label_columns = [
        config.LabelColumnInfo(label_column_name="label", n_unique_values=2)
    ]
    tokenizer = load_tokenizer("ref2vec")
    pl_data_module_em = DNASeqEpigeneticMarksDataModule(
        tokenizer=tokenizer,
        batch_size=2,
        fields=dnaseq_fields,
        label_columns=label_columns,
        collation_strategy="sequence_classification",
        max_length=512,
        dataset_kwargs=dnaseq_dataset_kwargs,
    )
    pl_data_module_em.setup("fit")

    for batch in pl_data_module_em.train_dataloader():
        assert tuple(batch["attention_mask"].shape) == (2, 512)
        assert tuple(batch["input_ids"].shape) == (2, 1, 512)
        assert tuple(batch["labels"]["label"].shape) == (2,)
        assert int(batch["input_ids"][0][0][0]) == 3
        break


def test_splice_site_module(dnaseq_fields):
    dnaseq_dataset_kwargs = {
        "processed_data_source": helpers.DNASeqSpliceSitePaths.processed_data_source,
        "label_dict_path": helpers.DNASeqSpliceSitePaths.label_dict_path,
        "num_workers": 0,
    }
    label_columns = [
        config.LabelColumnInfo(label_column_name="label", n_unique_values=3)
    ]
    tokenizer = load_tokenizer("ref2vec")
    pl_data_module_splice_site = DNASeqSpliceSiteDataModule(
        tokenizer=tokenizer,
        batch_size=2,
        fields=dnaseq_fields,
        label_columns=label_columns,
        collation_strategy="sequence_classification",
        max_length=512,
        dataset_kwargs=dnaseq_dataset_kwargs,
    )
    pl_data_module_splice_site.setup("fit")

    for batch in pl_data_module_splice_site.train_dataloader():
        assert tuple(batch["attention_mask"].shape) == (2, 512)
        assert tuple(batch["input_ids"].shape) == (2, 1, 512)
        assert tuple(batch["labels"]["label"].shape) == (2,)
        assert int(batch["input_ids"][0][0][0]) == 3
        break


def test_mpra_module(dnaseq_fields):
    dnaseq_dataset_kwargs = {
        "processed_data_source": helpers.DNASeqMPRAPaths.processed_data_source,
        "label_dict_path": helpers.DNASeqMPRAPaths.label_dict_path,
        "num_workers": 0,
    }
    label_columns = [
        config.LabelColumnInfo(
            label_column_name="mean_value", n_unique_values=1, is_regression_label=True
        )
    ]
    tokenizer = load_tokenizer("ref2vec")
    pl_data_module_mpra = DNASeqMPRADataModule(
        tokenizer=tokenizer,
        batch_size=2,
        fields=dnaseq_fields,
        label_columns=label_columns,
        collation_strategy="sequence_classification",
        max_length=512,
        dataset_kwargs=dnaseq_dataset_kwargs,
    )
    pl_data_module_mpra.setup("fit")

    for batch in pl_data_module_mpra.train_dataloader():
        assert tuple(batch["attention_mask"].shape) == (2, 512)
        assert tuple(batch["input_ids"].shape) == (2, 1, 512)
        assert tuple(batch["labels"]["mean_value"].shape) == (2,)
        assert int(batch["input_ids"][0][0][0]) == 3
        break


def test_transcription_factor_module(dnaseq_fields):
    dnaseq_dataset_kwargs = {
        "processed_data_source": helpers.DNASeqTranscriptionFactorPaths.processed_data_source,
        "label_dict_path": helpers.DNASeqTranscriptionFactorPaths.label_dict_path,
        "num_workers": 0,
    }
    label_columns = [
        config.LabelColumnInfo(label_column_name="label", n_unique_values=2)
    ]
    tokenizer = load_tokenizer("ref2vec")
    pl_data_module_tf = DNASeqTranscriptionFactorDataModule(
        tokenizer=tokenizer,
        batch_size=2,
        fields=dnaseq_fields,
        label_columns=label_columns,
        collation_strategy="sequence_classification",
        max_length=512,
        dataset_kwargs=dnaseq_dataset_kwargs,
    )
    pl_data_module_tf.setup("fit")

    for batch in pl_data_module_tf.train_dataloader():
        assert tuple(batch["attention_mask"].shape) == (2, 512)
        assert tuple(batch["input_ids"].shape) == (2, 1, 512)
        assert tuple(batch["labels"]["label"].shape) == (2,)
        assert int(batch["input_ids"][0][0][0]) == 3
        break
