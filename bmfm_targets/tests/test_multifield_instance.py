import random
from collections import Counter
from itertools import chain

import numpy as np
import pandas as pd
import pytest
from numpy.random import MT19937, RandomState, SeedSequence

from bmfm_targets.tokenization import MultiFieldInstance
from bmfm_targets.tokenization.resources import reference_data
from bmfm_targets.training import sample_transforms


def test_can_instantiate():
    mfi = MultiFieldInstance(
        metadata={"cell_name": "test"},
        data={"genes": "token1", "expressions": "token2"},
    )
    assert mfi["genes"] == "token1"
    assert mfi["expressions"] == "token2"


def test_can_instantiate_from_str_lists():
    mfi = MultiFieldInstance(
        metadata={"cell_name": "test"},
        data={"genes": ["token1", "token2"], "expressions": ["token3", "token4"]},
    )
    assert mfi["genes"] == ["token1", "token2"]
    assert mfi["expressions"] == ["token3", "token4"]


def test_shuffle_sort_instance():
    mfi_0 = MultiFieldInstance(
        metadata={"cell_name": "test"},
        data={
            "genes": ["token1", "token2", "token3", "token4"],
            "expressions": ["4", "1", "5", "6"],
        },
    )
    mfi = sample_transforms.sort_by_field(mfi_0, "expressions")
    assert mfi["genes"] == ["token4", "token3", "token1", "token2"]
    assert mfi["expressions"] == ["6", "5", "4", "1"]
    mfi = sample_transforms.randomize(mfi_0, seed=123)
    assert mfi["genes"] != ["token4", "token3", "token1", "token2"]
    assert mfi["expressions"] != ["6", "5", "4", "1"]


def test_remap_feild(tmp_path):
    import pandas as pd

    mapping_data = {
        "species_a_gene_name": ["TOKEN1", "TOKEN2", "TOKEN3", "TOKEN4"],
        "species_b_gene_name": ["token1", "token2", "token3", "token4"],
    }
    tmp_mapping_file = tmp_path / "ortholog_mapping.tsv"
    pd.DataFrame(mapping_data).to_csv(tmp_mapping_file, sep="\t")

    mfi_0 = MultiFieldInstance(
        metadata={"cell_name": "test"},
        data={
            "genes": ["TOKEN1", "TOKEN2", "TOKEN3", "TOKEN4"],
            "expressions": ["4", "1", "5", "6"],
        },
    )

    mapping = reference_data.get_ortholog_genes(
        from_species="species_a",
        to_species="species_b",
        return_mapping=True,
        mapping_file=tmp_mapping_file,
    )
    mfi = sample_transforms.field_remap(mfi_0, mapping, "genes")

    assert mfi["genes"] == ["token1", "token2", "token3", "token4"]
    assert mfi["expressions"] == ["4", "1", "5", "6"]


def test_dropout_chunk():
    mfi = MultiFieldInstance(
        metadata={"cell_name": "test"},
        data={
            "genes": ["token" + str(x) for x in range(1, 9)],
            "expressions": [str(x) for x in range(1, 9)],
        },
    )

    mfi1 = sample_transforms.dropout_chunk_in_range(mfi, 9, (0, 10))
    assert len(mfi1["genes"]) == 8

    mfi1 = sample_transforms.dropout_chunk_in_range(mfi, 2, (0, 1))
    assert len(mfi1["genes"]) == 6


def test_dropout_random():
    mfi = MultiFieldInstance(
        metadata={"cell_name": "test"},
        data={
            "genes": ["token" + str(x) for x in range(1, 100)],
            "expressions": [str(x) for x in range(1, 100)],
        },
    )
    mfi1 = sample_transforms.dropout_random(mfi, 0.5)
    assert len(mfi1["genes"]) > 30
    assert len(mfi1["genes"]) < 70


def test_rda_downsampling():
    mfi = MultiFieldInstance(
        metadata={"cell_name": "test"},
        data={
            "genes": ["token" + str(x) for x in range(1, 100)],
            "expressions": [str(x) for x in range(1, 100)],
        },
    )
    mfi1 = sample_transforms.rda_downsample(mfi, 10, 10000, 100)

    assert mfi1["genes"][:2] == ["[S]", "[T]"]
    assert float(mfi1["expressions"][0]) == float(mfi1["expressions"][1])
    assert float(mfi1["label_expressions"][0]) == float(mfi1["label_expressions"][1])
    assert mfi1["expressions"][0] == np.log1p(36)


def test_poisson_downsample():
    mfi = MultiFieldInstance(
        metadata={"cell_name": "test"},
        data={
            "genes": ["token" + str(x) for x in range(1, 100)],
            "expressions": [str(x) for x in range(1, 100)],
        },
    )

    mfi1 = sample_transforms.poisson_downsample(mfi, renoise=0.6)
    n_genes = len(mfi["expressions"])
    expected_T = np.log1p(sum([float(x) for x in mfi["expressions"]]))

    assert mfi1["genes"][:2] == ["[S]", "[T]"]
    assert float(mfi1["expressions"][1]) == expected_T
    assert len(mfi1["label_expressions"]) == n_genes + 2
    assert (np.count_nonzero(np.array(mfi1["expressions"]) == 0)) > 0


def test_pad_zero_expressed_genes_batchwise():
    mfis = []
    max_length = 70
    expressed_lengths = [40, 50, 60, 70, 80]
    for i in range(5):
        mfi = MultiFieldInstance(
            metadata={"cell_name": "test"},
            data={
                "genes": ["token" + str(x) for x in range(1, 101)],
                "expressions": list(range(1, expressed_lengths[i] + 1))
                + [0] * (100 - expressed_lengths[i]),
            },
        )

        mfis.append(mfi)
    updated_mfis = []

    expressed_genes_in_batch = {
        gene
        for mfi in mfis
        for gene, expression in zip(mfi.data["genes"], mfi.data["expressions"])
        if expression != 0.0
    }
    for mfi in mfis:
        updated_mfi = sample_transforms.pad_zero_expressed_genes(
            mfi,
            pad_zero_expression_strategy={"strategy": "batch_wise"},
            expressed_genes_in_batch=expressed_genes_in_batch,
            max_length=max_length,
        )
        updated_mfis.append(updated_mfi)
    assert len(updated_mfis[0]["genes"]) == 70
    assert updated_mfis[0]["expressions"][expressed_lengths[0] :] == [0.0] * 30

    assert len(updated_mfis[4]["genes"]) == 70
    assert 0.0 not in updated_mfis[4]["expressions"]


def test_pad_zero_expressed_genes_random():
    max_length = 90
    expressed_lengths = [40, 50, 60, 70, 80]
    mfis = []
    for i in range(5):
        mfi = MultiFieldInstance(
            metadata={"cell_name": "test"},
            data={
                "genes": ["token" + str(x) for x in range(1, 101)],
                "expressions": list(range(1, expressed_lengths[i] + 1))
                + [0] * (100 - expressed_lengths[i]),
            },
        )

        mfis.append(mfi)
    updated_mfis = []
    for mfi in mfis:
        updated_mfi = sample_transforms.pad_zero_expressed_genes(
            mfi,
            pad_zero_expression_strategy={"strategy": "random"},
            max_length=max_length,
        )
        updated_mfis.append(updated_mfi)

    assert len(updated_mfis[0]["genes"]) == 90
    assert updated_mfis[0]["expressions"][expressed_lengths[0] :] == [0.0] * 50
    assert updated_mfis[0]["genes"][-10:] != updated_mfis[1]["genes"][-10:]
    assert updated_mfis[0]["expressions"][-10:] == updated_mfis[1]["expressions"][-10:]


def test_rda_downsampling_always_less_than_threshold():
    for i in range(100):
        mfi = MultiFieldInstance(
            metadata={"cell_name": "test"},
            data={
                "genes": ["token" + str(x) for x in range(1, 10)],
                "expressions": [str(x) for x in random.sample(range(0, 100), 10)],
            },
        )
        mfi1 = sample_transforms.rda_downsample(mfi, 10, 10000, 1000)
        assert mfi1["genes"][:2] == ["[S]", "[T]"]
        assert float(mfi1["expressions"][0]) == float(mfi1["expressions"][1])
        assert float(mfi1["label_expressions"][0]) == float(
            mfi1["label_expressions"][1]
        )
        assert mfi1["expressions"][0] == np.log1p(
            np.sum([int(exp) for exp in mfi["expressions"][0:8]])
        )


def test_rda_downsampling_always_greater_than_threshold():
    gamma_sampled = 0
    gamma_zero = 0
    for i in range(100):
        mfi = MultiFieldInstance(
            metadata={"cell_name": "test"},
            data={
                "genes": ["token" + str(x) for x in range(1, 10)],
                "expressions": [str(x) for x in random.sample(range(0, 100), 10)],
            },
        )
        mfi1 = sample_transforms.rda_downsample(mfi, 10, 10000, 10)
        assert mfi1["genes"][:2] == ["[S]", "[T]"]
        assert float(mfi1["expressions"][0]) <= float(mfi1["expressions"][1])
        assert float(mfi1["label_expressions"][0]) <= float(
            mfi1["label_expressions"][1]
        )
        assert mfi1["expressions"][1] == np.log1p(
            np.sum([int(exp) for exp in mfi["expressions"][0:8]])
        )
        assert mfi1["expressions"][0] <= np.log1p(
            np.sum([int(exp) for exp in mfi["expressions"][0:8]])
        )

        if float(mfi1["expressions"][0]) == float(mfi1["expressions"][1]):
            gamma_zero = gamma_zero + 1
        else:
            gamma_sampled = gamma_sampled + 1

    assert gamma_zero > 30
    assert gamma_sampled > 30
    assert gamma_zero + gamma_sampled == 100


def test_rda_align():
    target_reads = 1000
    for _ in range(100):
        mfi = MultiFieldInstance(
            metadata={"cell_name": "test"},
            data={
                "genes": ["token" + str(x) for x in range(1, 10)],
                "expressions": list(random.sample(range(0, 100), 10)),
            },
        )
        mfi1 = sample_transforms.rda_align(
            mfi,
            target_read_resolution=target_reads,
            normalized_sum=10000,
            max_length=10,
        )
        assert mfi1["genes"][:2] == ["[S]", "[T]"]
        assert float(mfi1["expressions"][0]) <= float(mfi1["expressions"][1])

        assert mfi1["expressions"][0] == np.log1p(
            np.sum([int(exp) for exp in mfi["expressions"][0:8]])
        )
        assert mfi1["expressions"][1] <= np.log1p(target_reads)


def test_float_to_int():
    def _normed_negative_binomial(size=10):
        rs = RandomState(MT19937(SeedSequence(123456789)))
        s = rs.negative_binomial(1, 0.01, size)
        return np.log1p(10000 * s / sum(s)).tolist()

    for _ in range(100):
        mfi = MultiFieldInstance(
            metadata={"cell_name": "test"},
            data={
                "genes": ["token" + str(x) for x in range(1, 10)],
                "expressions": _normed_negative_binomial(),
            },
        )
        mfi_round = sample_transforms.downcast_numeric_fields(
            mfi, fields_to_downcast=["expressions"]
        )
        assert not all(isinstance(i, int) for i in mfi["expressions"])
        assert all(isinstance(i, int) for i in mfi_round["expressions"])
        mfi_floor = sample_transforms.downcast_numeric_fields(
            mfi, fields_to_downcast=["expressions"], casting_strategy="floor"
        )
        mfi_ceil = sample_transforms.downcast_numeric_fields(
            mfi, fields_to_downcast=["expressions"], casting_strategy="ceil"
        )
        assert (
            sum(mfi_ceil["expressions"])
            > sum(mfi_round["expressions"])
            > sum(mfi_floor["expressions"])
        )


def test_uce_input():
    """Verify number of genes depends on expressions."""
    mfi = MultiFieldInstance(
        data={"genes": ["MFSD14A", "TRMT13", "TSPY9"], "expressions": [10, 1, 1]}
    )
    chrom_df = pd.DataFrame(
        data={
            "gene_symbol": ["MFSD14A", "TRMT13", "TSPY9"],
            "chromosome": ["1", "1", "Y"],
            "start": [100038095, 100133215, 9487267],
            "species": ["human", "human", "human"],
        }
    ).set_index("gene_symbol")
    transformed_mfi = sample_transforms.encode_expression_as_repeats(
        mfi, chrom_df, seed=42
    )
    gene_counts = Counter(transformed_mfi["genes"])
    assert gene_counts["MFSD14A"] >= gene_counts["TRMT13"], gene_counts
    assert gene_counts["MFSD14A"] >= gene_counts["TSPY9"], gene_counts


def test_uce_sorting_is_correctly_random():
    """Verify gene sorting in the same chromosome."""
    mfi = MultiFieldInstance(
        data={"genes": ["MFSD14A", "TRMT13", "TSPY9"], "expressions": [1, 1, 1]}
    )

    chrom_df = pd.DataFrame(
        data={
            "gene_symbol": ["MFSD14A", "TRMT13", "TSPY9"],
            "chromosome": ["1", "1", "Y"],
            "start": [10003, 100133215, 9487267],
            "species": ["human", "human", "human"],
        }
    ).set_index("gene_symbol")

    examples = [
        sample_transforms.encode_expression_as_repeats(mfi, chrom_df)
        for _ in range(100)
    ]
    chromsome_examples = [
        example for example in examples if "MFSD14A" in example["genes"]
    ]
    same_chromsome_examples = [
        example for example in chromsome_examples if "TRMT13" in example["genes"]
    ]
    diff_chromsome_examples = [
        example for example in chromsome_examples if "TSPY9" in example["genes"]
    ]

    same_chromosome_order = lambda mfi: mfi["genes"].index("MFSD14A") < mfi[
        "genes"
    ].index("TRMT13")
    diff_chromsome_order = lambda mfi: mfi["genes"].index("MFSD14A") < mfi[
        "genes"
    ].index("TSPY9")
    assert all(same_chromosome_order(mfi) for mfi in same_chromsome_examples)
    assert any(diff_chromsome_order(mfi) for mfi in diff_chromsome_examples)
    assert not all(diff_chromsome_order(mfi) for mfi in diff_chromsome_examples)


def test_uce_sorting_is_correctly_sort_by_chromosome():
    """Verify no chromosome group appears more than once."""
    mfi = MultiFieldInstance(
        data={
            "genes": ["GENE1_chr1", "GENE2_chr1", "GENE3_chr6", "GENE4_chr7"],
            "expressions": [2, 1, 1, 1],
        }
    )
    chrom_df = pd.DataFrame(
        data={
            "gene_symbol": ["GENE1_chr1", "GENE2_chr1", "GENE3_chr6", "GENE4_chr7"],
            "chromosome": ["chr1", "chr1", "chr6", "chr7"],
            "start": [1, 10, 9999, 999],
            "species": ["human", "human", "human", "human"],
        }
    ).set_index("gene_symbol")
    examples = [
        sample_transforms.encode_expression_as_repeats(mfi, chrom_df) for _ in range(20)
    ]
    # extract chroms
    chrom_examples = [
        [genes.split("chr")[1] for genes in example.data["genes"]]
        for example in examples
    ]
    for i, chroms in enumerate(chrom_examples):
        prev_list = []
        prev = None
        for chrom in chroms:
            assert chrom not in prev_list
            if (chrom != prev) and (i > 0):
                # save previous elements
                prev_list.append(prev)
            prev = chrom


def test_uce_undefined_chromosome_input():
    """Verify genes which are undefined for chromosome are removed."""
    mfi = MultiFieldInstance(
        data={"genes": ["GENE1", "GENE2", "GENE3"], "expressions": [10, 1, 3]}
    )
    chrom_df = pd.DataFrame(
        data={
            "gene_symbol": ["GENE1", "GENE2"],
            "chromosome": ["1", "A"],
            "start": [100, 9999],
            "species": ["human", "human"],
        }
    ).set_index("gene_symbol")
    transformed_mfi = sample_transforms.encode_expression_as_repeats(mfi, chrom_df)
    gene_counts = Counter(transformed_mfi["genes"])
    assert gene_counts["GENE3"] == 0


def test_limit_input_tokens():
    """Only use the subset of genes/tokens given (mainly used for WCED)."""
    mfi = MultiFieldInstance(
        data={
            "genes": [f"GENE{i}" for i in range(100)],
            "expressions": np.random.randint(1, 100, 100).tolist(),
        }
    )

    transformed_mfi = sample_transforms.limit_fields_to_token_list(
        mfi=mfi, token_list=["GENE0", "GENE99"], field_name="genes"
    )

    assert len(transformed_mfi["genes"]) == 2


@pytest.mark.parametrize("dropout_fraction", [0.1, 0.3, 0.5, 0.7])
def test_selective_dropout(dropout_fraction):
    """Dropout certain genes with higher probability than others."""
    mfi = MultiFieldInstance(
        data={
            "genes": [f"GENE{i}" for i in range(100)],
            "expressions": np.random.randint(1, 100, 100).tolist(),
        }
    )
    dropout_weights = pd.Series(
        index=mfi["genes"], data=[4] * 25 + [3] * 25 + [2] * 25 + [1] * 25
    )

    transformed_mfis = [
        sample_transforms.selective_dropout(
            mfi, dropout_weights=dropout_weights, dropout_fraction=dropout_fraction
        )
        for _ in range(3000)
    ]

    transformed_seq_len = np.mean([tm.seq_length for tm in transformed_mfis])

    np.testing.assert_almost_equal(
        transformed_seq_len / mfi.seq_length, 1 - dropout_fraction, decimal=1
    )

    gene_freq = sorted(
        Counter(chain.from_iterable(tm["genes"] for tm in transformed_mfis)).items(),
        key=lambda x: int(x[0].lstrip("GENE")),
    )
    assert np.mean([i[1] for i in gene_freq[:25]]) < np.mean(
        [i[1] for i in gene_freq[25:50]]
    )
    assert np.mean([i[1] for i in gene_freq[25:50]]) < np.mean(
        [i[1] for i in gene_freq[50:75]]
    )
    assert np.mean([i[1] for i in gene_freq[50:75]]) < np.mean(
        [i[1] for i in gene_freq[75:]]
    )
