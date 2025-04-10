from typing import Any

from bmfm_targets.datasets.dnaseq import (
    DNASeqChromatinProfileDataset,
    DNASeqCorePromoterDataset,
    DNASeqCovidDataset,
    DNASeqDrosophilaEnhancerDataset,
    DNASeqEpigeneticMarksDataset,
    DNASeqMPRADataset,
    DNASeqPromoterDataset,
    DNASeqSpliceSiteDataset,
    DNASeqTranscriptionFactorDataset,
    StreamingDNASeqChromatinProfileDataset,
)
from bmfm_targets.tests import helpers


def test_core_promoter_dataset():
    dataset_kwargs: dict[str, Any] = {
        "processed_data_source": helpers.DNASeqCorePromoterPaths.processed_data_source,
        "label_dict_path": helpers.DNASeqCorePromoterPaths.label_dict_path,
        "label_columns": ["label"],
    }

    ds = DNASeqCorePromoterDataset(**dataset_kwargs, split="train")
    assert len(ds[0].metadata) == 1
    assert ds[0].metadata["label"] == "1"


def test_promoter_dataset():
    dataset_kwargs: dict[str, Any] = {
        "processed_data_source": helpers.DNASeqPromoterPaths.processed_data_source,
        "label_dict_path": helpers.DNASeqPromoterPaths.label_dict_path,
        "label_columns": ["promoter_presence"],
    }

    ds = DNASeqPromoterDataset(**dataset_kwargs, split="train")
    assert len(ds[0].metadata) == 1
    assert ds[0].metadata["promoter_presence"] == "0"


def test_splice_site_dataset():
    dataset_kwargs: dict[str, Any] = {
        "processed_data_source": helpers.DNASeqSpliceSitePaths.processed_data_source,
        "label_dict_path": helpers.DNASeqSpliceSitePaths.label_dict_path,
        "label_columns": ["label"],
    }

    ds = DNASeqSpliceSiteDataset(**dataset_kwargs, split="train")
    assert len(ds[0].metadata) == 1
    assert ds[0].metadata["label"] == "0"


def test_covid_dataset():
    dataset_kwargs: dict[str, Any] = {
        "processed_data_source": helpers.DNASeqCovidPaths.processed_data_source,
        "label_dict_path": helpers.DNASeqCovidPaths.label_dict_path,
        "regression_label_columns": ["label"],
    }

    ds = DNASeqCovidDataset(**dataset_kwargs, split="train")
    assert len(ds[0].metadata) == 1
    assert ds[0].metadata["label"] == 7.0


def test_chromatin_profile_dataset():
    dataset_kwargs: dict[str, Any] = {
        "processed_data_source": helpers.DNASeqChromatinProfilePaths.processed_data_source,
        "label_dict_path": helpers.DNASeqChromatinProfilePaths.label_dict_path,
        "label_columns": ["dnase_0", "dnase_1", "dnase_2"],
    }

    ds = DNASeqChromatinProfileDataset(**dataset_kwargs, split="train")
    assert len(ds[0].metadata) == 3
    assert ds[0].metadata["dnase_0"] == "0"
    assert ds[0].metadata["dnase_1"] == "0"
    assert ds[0].metadata["dnase_2"] == "1"


def test_drosophila_enhancer_dataset():
    dataset_kwargs: dict[str, Any] = {
        "processed_data_source": helpers.DNASeqDrosophilaEnhancerPaths.processed_data_source,
        "label_dict_path": helpers.DNASeqDrosophilaEnhancerPaths.label_dict_path,
        "regression_label_columns": ["Dev_log2_enrichment"],
    }

    ds = DNASeqDrosophilaEnhancerDataset(**dataset_kwargs, split="train")
    assert len(ds[0].metadata) == 1
    assert ds[0].metadata["Dev_log2_enrichment"] == 5.71


def test_epigenetic_marks_dataset():
    dataset_kwargs: dict[str, Any] = {
        "processed_data_source": helpers.DNASeqEpigeneticMarksPaths.processed_data_source,
        "label_dict_path": helpers.DNASeqEpigeneticMarksPaths.label_dict_path,
        "label_columns": ["label"],
    }

    ds = DNASeqEpigeneticMarksDataset(**dataset_kwargs, split="train")
    assert len(ds[0].metadata) == 1
    assert ds[0].metadata["label"] == "1"


def test_mpra_dataset():
    dataset_kwargs: dict[str, Any] = {
        "processed_data_source": helpers.DNASeqMPRAPaths.processed_data_source,
        "label_dict_path": helpers.DNASeqMPRAPaths.label_dict_path,
        "regression_label_columns": ["mean_value"],
    }

    ds = DNASeqMPRADataset(**dataset_kwargs, split="train")
    assert len(ds[0].metadata) == 1
    assert ds[0].metadata["mean_value"] == -0.804


def test_transcription_factor_dataset():
    dataset_kwargs: dict[str, Any] = {
        "processed_data_source": helpers.DNASeqTranscriptionFactorPaths.processed_data_source,
        "label_dict_path": helpers.DNASeqTranscriptionFactorPaths.label_dict_path,
        "label_columns": ["label"],
    }

    ds = DNASeqTranscriptionFactorDataset(**dataset_kwargs, split="train")
    assert len(ds[0].metadata) == 1
    assert ds[0].metadata["label"] == "0"


def test_streaming_DNASeq_chromatin_profile_dataset():
    dataset_kwargs: dict[str, Any] = {
        "processed_data_source": helpers.StreamingDNASeqChromatinProfilePaths.processed_data_source,
        # "label_dict_path": helpers.StreamingDNASeqChromatinProfilePaths.label_dict_path,
    }

    ds = StreamingDNASeqChromatinProfileDataset(**dataset_kwargs, split="dev")
    assert len(ds[0].metadata) == 919
    assert ds[0].metadata["dnase_1"] == "1"
