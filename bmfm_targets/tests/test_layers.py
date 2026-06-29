import pytest
import torch

from bmfm_targets.config import FieldInfo, SCBertConfig
from bmfm_targets.models.predictive import layers


@pytest.mark.parametrize("n_sin_basis", [5, 10, 20])
@pytest.mark.parametrize(
    ("sigmoid_centers", "sigmoid_orientations"),
    [
        (None, None),
        ([1, 3, 5], [1, -1, 1]),
    ],
)
@pytest.mark.parametrize("basis_scale", [0.1, 1])
@pytest.mark.parametrize("shift", [0, 1])
@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize("zero_as_special_token", [True, False])
def test_scale_adapt(
    n_sin_basis,
    sigmoid_centers,
    sigmoid_orientations,
    basis_scale,
    shift,
    trainable,
    zero_as_special_token,
):
    field = FieldInfo("expressions", num_special_tokens=5)
    config = SCBertConfig(fields=[field])
    generator = torch.Generator().manual_seed(42)
    sae = layers.ScaleAdaptEncoder(
        config,
        field,
        n_sin_basis=n_sin_basis,
        sigmoid_centers=sigmoid_centers,
        sigmoid_orientations=sigmoid_orientations,
        basis_scale=basis_scale,
        shift=shift,
        trainable=trainable,
        zero_as_special_token=zero_as_special_token,
        generator=generator,
    )

    input = torch.tensor([[0.0001, 0.001, 1.1, 12, 0, -1]])
    encoding = sae(input)
    encoding = encoding[0]  # remove batch dimension

    # no nans
    assert not encoding.isnan().any()

    delta_1e4_1e3 = torch.norm(encoding[0] - encoding[1])
    delta_1e4_1_1 = torch.norm(encoding[0] - encoding[2])
    delta_1e4_0 = torch.norm(encoding[0] - encoding[4])

    # nearby numbers closer
    assert delta_1e4_1e3 < delta_1e4_1_1
    if zero_as_special_token:
        # jumping to 1 is less than jumping to 0
        assert delta_1e4_1_1 < delta_1e4_0
    else:
        # jumping to 0 is less than jumping to 1
        assert delta_1e4_0 < delta_1e4_1_1


def test_sc_base_field_decoder_init_with_wced_decoder():
    fields = [
        FieldInfo("genes", vocab_size=1234),
        FieldInfo(
            "expressions",
            decode_modes={
                "wced": {
                    "vocab_field": "genes",
                    "logit_outputs": ["mse", "is_zero_bce"],
                }
            },
        ),
    ]

    model_config = SCBertConfig(fields=fields)

    decoder = layers.SCBaseFieldDecoder(model_config)
    assert sorted(decoder.field_decoders) == ["expressions_wced"]
    wced_decoder = decoder.field_decoders["expressions_wced"]
    assert wced_decoder.num_outputs_per_target == 2
    assert wced_decoder.output_size == 1234


def test_wced_decoder_only_projects_the_decode_token():
    """
    WCED is whole-cell: it must project only the decode token (index 0).

    Running the vocab-wide projection over the full sequence would materialize a
    [batch, seq_len, vocab] activation (and gradient) seq_len x larger than needed,
    which is the dominant memory cost for large gene vocabularies. Only token 0 is
    ever consumed downstream (WCEDFieldSource.extract_logits), so the head must
    return a length-1 sequence dim regardless of the input sequence length.
    """
    vocab_size = 1234
    fields = [
        FieldInfo("genes", vocab_size=vocab_size),
        FieldInfo(
            "expressions",
            decode_modes={"wced": {"vocab_field": "genes", "logit_outputs": ["mse"]}},
        ),
    ]
    model_config = SCBertConfig(fields=fields, hidden_size=32)
    decoder = layers.SCBaseFieldDecoder(model_config)

    batch, seq_len = 2, 17
    hidden_states = torch.randn(batch, seq_len, model_config.hidden_size)
    field_logits = decoder(hidden_states)

    wced_logits = field_logits["expressions_wced"]
    # single logit output -> shape is [batch, 1, vocab], not [batch, seq_len, vocab]
    assert wced_logits.shape == (batch, 1, vocab_size)
    # the downstream extraction (result[:, 0, :]) must still work
    assert wced_logits[:, 0, :].shape == (batch, vocab_size)


@pytest.mark.parametrize("decode_from", [0, 1, 2, 3])
def test_label_decoder_can_decode_from_custom_sequence_position(decode_from):
    from bmfm_targets.config import LabelColumnInfo

    fields = [
        FieldInfo("genes", vocab_size=1234),
        FieldInfo("expressions", tokenization_strategy="continuous_value_encoder"),
    ]
    label_column = LabelColumnInfo(
        "cell_type", decode_from=decode_from, n_unique_values=5
    )
    model_config = SCBertConfig(fields=fields, label_columns=[label_column])

    mth = layers.SCMultiTaskHead(model_config)

    batch_size = 8
    seq_len = 128
    sequence_output = torch.rand((batch_size, seq_len, model_config.hidden_size))
    pooled_output = torch.rand((batch_size, model_config.hidden_size))

    sequence_output[:, decode_from, :] = 0

    predictions = mth(
        sequence_output,
        pooled_output,
    )

    assert torch.all(predictions["cell_type"] == 0)
