import logging
import math

logger = logging.getLogger(__name__)
from functools import partial

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig

from bmfm_targets.config import FieldInfo, SCModelConfigBase
from bmfm_targets.training.metrics import masked_mean


class SCEmbeddingsLayer(nn.Module):
    """
    Construct the embeddings from field specific embeddings.

    Attributes
    ----------
        config:  class instance with the configuration to build a new model.
    """

    def __init__(self, config: SCModelConfigBase):
        """
        Init embeddings layer.

        Args:
        ----
            config (SCBertConfig): A SCBertConfig class instance with the configuration to build a new model.
        """
        super().__init__()
        self.config = config
        assert self.config.fields is not None
        for field in self.config.fields:
            if field.is_input:
                self.create_field_embedding_layer(config, field)

        self.position_embedding_type = config.position_embedding_type
        if self.position_embedding_type is not None:
            self.create_position_embedding_layer(config)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def create_position_embedding_layer(self, config):
        if self.position_embedding_type == "absolute":
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size
            )
            self.register_buffer(
                "position_ids",
                torch.arange(config.max_position_embeddings).expand((1, -1)),
                persistent=False,
            )

        elif self.position_embedding_type == "sinusoidal":
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size
            )
            self.sinusoidal_position_embeddings(
                config.max_position_embeddings,
                config.hidden_size,
                self.position_embeddings.weight,
            )
            self.register_buffer(
                "position_ids",
                torch.arange(config.max_position_embeddings).expand((1, -1)),
                persistent=False,
            )
        else:
            raise ValueError(
                "Unknown position embedding type. Should be either 'absolute' or 'sinusoidal'"
            )

    def create_field_embedding_layer(self, config, field: FieldInfo):
        if field.tokenization_strategy == "continuous_value_encoder":
            embedding_layer = self.make_continuous_value_encoder_layer(config, field)
            setattr(self, field.field_name + "_embeddings", embedding_layer)

        elif field.pretrained_embedding is not None:
            self.create_field_embedding_layer_with_pretrained_embeddings(config, field)
        else:
            embedding_layer = nn.Embedding(
                field.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
            )
            setattr(self, field.field_name + "_embeddings", embedding_layer)

    def make_continuous_value_encoder_layer(self, config, field: FieldInfo):
        default_kind = "mlp_with_special_token_embedding"
        if field.continuous_value_encoder_kwargs is not None:
            kwargs = {**field.continuous_value_encoder_kwargs}
            kind = kwargs.pop("kind", default_kind)
        else:
            kind = default_kind
            kwargs = {}

        if kind == "mlp":
            return ContinuousValueEncoder(config, **kwargs)
        if kind == "mlp_with_special_token_embedding":
            return ContinuousValueEncoderWithSpecialTokenEmbeddings(
                config, field, **kwargs
            )
        if kind == "scale_adapt":
            return ScaleAdaptEncoder(config, field, **kwargs)
        raise ValueError(
            'Continuous value "kind" should be in '
            f"[mlp, scale_adapt, mlp_with_special_token_embedding], not {kwargs['kind']}"
        )

    def create_field_embedding_layer_with_pretrained_embeddings(self, config, field):
        pretrained_embeddings = torch.from_numpy(
            field.pretrained_embedding.load_pretrained_embeddings().values
        )
        logger.info(f"loading pre computed embeddings for field {field}")
        logger.info(f"loading pre computed from {field.pretrained_embedding.filename}")

        frozen_indices_embedding = (
            field.pretrained_embedding.embedding_indices_to_freeze
        )
        indices_pre_computed = field.pretrained_embedding.pre_trained_indices_to_use
        logger.info(f"number of frozen indices {len(frozen_indices_embedding)}")
        embedding_layer = nn.Embedding(
            field.vocab_size,
            pretrained_embeddings.shape[1],
            padding_idx=config.pad_token_id,
        )
        logger.info(f"size of embedding layer {embedding_layer.weight.shape}")
        embedding_layer.weight.data[frozen_indices_embedding] = pretrained_embeddings[
            indices_pre_computed
        ].type(embedding_layer.weight.dtype)
        embedding_layer.weight.register_hook(
            partial(set_grad_to_zero, indices=frozen_indices_embedding)
        )
        setattr(
            self,
            field.field_name + "_embeddings",
            embedding_layer,
        )
        setattr(
            self,
            field.field_name + "_embeddings_projection",
            nn.Linear(pretrained_embeddings.shape[1], config.hidden_size),
        )

    def sinusoidal_position_embeddings(
        self, max_position_embeddings, hidden_size, embeddings
    ):
        """
        Create sinusoidal embeddings.

        Args:
        ----
            max_position_embeddings (int): The maximum number of positions.
            hidden_size (int): The hidden size.
            embeddings (torch.Tensor): A torch tensor with shape [max_position_embeddings, hidden_size] containing the embeddings.

        Returns:
        -------
            torch.Tensor: A torch tensor with shape [max_position_embeddings, hidden_size] containing the sinusoidal embeddings.
        """
        position_ids = torch.arange(max_position_embeddings).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size)
        )
        embeddings.requires_grad = False
        embeddings[:, 0::2] = torch.sin(position_ids * div_term)
        embeddings[:, 1::2] = torch.cos(position_ids * div_term)
        embeddings.detach_()

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of the embeddings layer.

        Args:
        ----
            input_ids (torch.Tensor): A torch tensor with shape [batch_size, num_fields, seq_length] containing the input ids.
            position_ids (torch.LongTensor | None): position ids. Default, None
            inputs_embeds: (torch.Tensor | None) precalculated embeddings. If supplied,
                bypasses logic of this module. Default, None.

        Returns:
        -------
            torch.Tensor: A torch tensor with shape [batch_size, _, hidden_size] containing the embeddings.
        """
        if inputs_embeds is not None:
            return inputs_embeds
        embeddings = []
        assert self.config.fields is not None
        # works as is, but should be refactored a little bit, its a bit messy
        input_fields = [field for field in self.config.fields if field.is_input]
        for i, field in enumerate(input_fields):
            embeds = self.calculate_field_embedding(input_ids, i, field)
            embeddings.append(embeds)
        seq_length = input_ids.size(2)

        updated_embeddings = torch.sum(torch.stack(embeddings, dim=2), dim=2)
        if self.position_embedding_type is not None:
            if position_ids is None:
                position_ids = self.position_ids[:, :seq_length]
            else:
                assert (
                    position_ids.size(1) == seq_length
                ), "Position IDs passed to SCEmbeddingsLayer have shape {} and should have shape {}".format(
                    position_ids.size(), seq_length
                )
            position_embeddings = self.position_embeddings(position_ids)
            updated_embeddings += position_embeddings

        updated_embeddings = self.LayerNorm(updated_embeddings)
        updated_embeddings = self.dropout(updated_embeddings)
        return updated_embeddings

    def calculate_field_embedding(
        self, input_ids: torch.Tensor, i: int, field: FieldInfo
    ):
        if field.pretrained_embedding is not None:
            embeds = getattr(self, field.field_name + "_embeddings")(
                input_ids[:, i, :].int()
            )
            return getattr(self, field.field_name + "_embeddings_projection")(embeds)
        if field.tokenization_strategy == "continuous_value_encoder":
            return getattr(self, field.field_name + "_embeddings")(
                input_ids[:, i, :].float()
            )
        return getattr(self, field.field_name + "_embeddings")(input_ids[:, i, :].int())


class SCIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class SCOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def set_grad_to_zero(grads, indices):
    """
    Set gradients to zero for given indices.
    This ability is created for freezing only part of the embedding layer during training
    in instances when there are only partial pre computed embeddings available and the rest should be learned.

    Args:
    ----
        grads: the gradients accessed with register_hook
        indices: the indices to set to zero.

    """
    grads.data[indices] *= 0


def get_embeddings_from_outputs(outputs, attention_mask, pooling_method):
    if pooling_method == "pooling_layer":
        return outputs.embeddings
    if pooling_method == "first_token":
        last_hidden_states = outputs.hidden_states[-1]
        # use CLS token only
        return last_hidden_states[:, 0, :]
    if pooling_method == "mean_pooling":
        last_hidden_states = outputs.hidden_states[-1]
        # use everything except CLS token
        return masked_mean(last_hidden_states[:, 1:, :], attention_mask[:, 1:])
    raise ValueError(f"Unsupported pooler type: {pooling_method}")


class SCSelfOutput(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SCPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def make_field_decoder(
    config, field_name: str, decode_mode: str, vocab_size: int = 1
) -> nn.Module:
    if decode_mode in ("regression", "is_zero"):
        return FieldDecoder(config)
    elif decode_mode == "token_scores":
        if vocab_size == 1:
            raise ValueError(
                f"Cannot use `token_scores` mode with vocab_size=1 for: {field_name}"
            )
        if vocab_size is None:
            raise ValueError(
                f"Cannot use `token_scores` mode without setting vocab size for field: {field_name}"
            )
        return FieldDecoder(config, vocab_size)
    else:
        raise ValueError(f"Unsupported decode mode: {decode_mode}")


class FieldDecoder(nn.Linear):
    def __init__(
        self,
        config: SCModelConfigBase,
        output_size: int = 1,
    ):
        super().__init__(config.hidden_size, output_size)
        self.bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return super().forward(hidden_states)


class LabelDecoder(nn.Linear):
    def __init__(
        self,
        config: SCModelConfigBase,
        output_size: int = 1,
    ):
        super().__init__(config.hidden_size, output_size)
        self.bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return super().forward(hidden_states)


class SCBaseFieldDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.field_decoders = nn.ModuleDict()

        for field in self.fields_to_decode():
            for decode_mode in field.decode_modes:
                output_dim = field.vocab_size if decode_mode == "token_scores" else 1
                field_decoder_name = f"{field.field_name}_{decode_mode}"
                field_decoder = make_field_decoder(
                    config, field.field_name, decode_mode, output_dim
                )
                self.field_decoders[field_decoder_name] = field_decoder

    def forward(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        field_logits = {}
        for field_decoder_name, field_decoder in self.field_decoders.items():
            field_logits[field_decoder_name] = field_decoder(hidden_states)
        return field_logits


class SCClassificationDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.label_decoders = nn.ModuleDict()

        for label in self.config.label_columns:
            output_dim = 1 if label.is_regression_label else label.output_size
            label_decoder = LabelDecoder(config, output_dim)
            self.label_decoders[label.label_column_name] = label_decoder

    def forward(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        label_logits = {}
        for label_decoder_name, label_decoder in self.label_decoders.items():
            label_logits[label_decoder_name] = label_decoder(hidden_states)
        return label_logits


class SCLMFieldDecoder(SCBaseFieldDecoder):
    def fields_to_decode(self):
        return filter(lambda f: f.is_masked, self.config.fields)


class SCLabelFieldDecoder(SCBaseFieldDecoder):
    def fields_to_decode(self):
        return filter(lambda f: not f.is_input, self.config.fields)


class SCPredictionHeadTransform(nn.Module):
    def __init__(self, config: SCModelConfigBase):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class SCLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = SCPredictionHeadTransform(config)
        self.decoder = SCLMFieldDecoder(config)

    def forward(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden_states = self.transform(hidden_states)
        return self.decoder(hidden_states)


class SCSequenceLabelPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = SCPredictionHeadTransform(config)
        self.decoder = SCLabelFieldDecoder(config)

    def forward(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden_states = self.transform(hidden_states)
        return self.decoder(hidden_states)


class SCOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = SCLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class SCMultiTaskClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = SCClassificationDecoder(config)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(pooled_output)
        return prediction_scores


class SCSequenceLabelingHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = SCSequenceLabelPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class SCClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config: SCModelConfigBase, output_size: int):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, output_size)

        self.config = config

    def forward(self, pooled_output, **kwargs):
        x = self.dense(pooled_output)
        x = ACT2FN[self.config.hidden_act](x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class SCMultiTaskHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = SCOnlyMLMHead(config)
        self.label_predictions = SCMultiTaskClassificationHead(config)

    def forward(self, sequence_output, pooled_output):
        predictions = self.predictions(sequence_output)
        predictions.update(self.label_predictions(pooled_output))
        return predictions


class ContinuousValueEncoder(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.dense_1 = nn.Linear(1, 128)
        self.activation = nn.LeakyReLU()
        self.dense_2 = nn.Linear(128, config.hidden_size)

    def forward(self, input_value: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = input_value.size(0), input_value.size(1)
        input_value = input_value.reshape(-1, 1)
        input_value = self.activation(self.dense_1(input_value))
        input_value = self.activation(self.dense_2(input_value))
        input_value = input_value.reshape(batch_size, seq_length, -1)
        return input_value


class ContinuousValueEncoderWithSpecialTokenEmbeddings(nn.Module):
    def __init__(
        self,
        config: SCModelConfigBase,
        field: FieldInfo,
        zero_as_special_token: bool = False,
    ):
        super().__init__()
        self.config = config
        self.hidden_dim = 128
        self.dense_1 = nn.Linear(1, self.hidden_dim)
        self.dense_2 = nn.Linear(self.hidden_dim, config.hidden_size)
        self.activation = nn.LeakyReLU()
        self.zero_as_special_token = zero_as_special_token
        if zero_as_special_token is True:
            num_special_tokens = field.num_special_tokens + 1
        else:
            num_special_tokens = field.num_special_tokens
        self.special_token_embeddings = nn.Embedding(
            num_embeddings=num_special_tokens,
            embedding_dim=self.hidden_dim,
            padding_idx=2,
        )

    def forward(self, input_value: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = input_value.size(0), input_value.size(1)
        input_value_flat = input_value.contiguous().view(-1)

        special_token_mask = input_value_flat < 0

        if self.zero_as_special_token:
            zero_mask = input_value_flat == 0.0
            special_token_mask |= zero_mask

        special_token_indices = torch.nonzero(special_token_mask).squeeze(1)
        mapped_special_token_values = -(
            input_value_flat[special_token_indices].long() + 1
        )

        if self.zero_as_special_token:
            mapped_special_token_values[
                input_value_flat[special_token_indices] == 0
            ] = (self.special_token_embeddings.num_embeddings - 1)

        special_token_embeds = self.special_token_embeddings(
            mapped_special_token_values
        )

        continuous_values = input_value_flat[~special_token_mask].view(-1, 1)
        continuous_value_embeds = self.activation(self.dense_1(continuous_values))
        dtype = input_value.dtype
        output_tensor = torch.zeros(
            input_value_flat.size(0),
            self.hidden_dim,
            device=input_value.device,
            dtype=dtype,
        )

        output_tensor[~special_token_mask] = continuous_value_embeds.to(dtype)
        output_tensor[special_token_mask] = special_token_embeds.to(dtype)

        output_tensor = self.dense_2(self.activation(output_tensor))
        output_tensor = output_tensor.view(batch_size, seq_length, -1)

        return output_tensor


class ScaleAdaptEncoder(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        field,
        n_sin_basis: int,
        sigmoid_centers: list[float] | None = None,
        sigmoid_orientations: list[float] | None = None,
        basis_scale: float = 1.0,
        shift: float = 1.0,
        trainable: bool = True,
    ):
        super().__init__()

        self.special_token_embeddings = nn.Embedding(
            num_embeddings=field.num_special_tokens,
            embedding_dim=config.hidden_size,
            padding_idx=2,
        )

        self.has_sigmoids = self.has_sin_basis = False
        n_sigmoid_centers = len(sigmoid_centers) if sigmoid_centers else 0

        if n_sin_basis > 0:
            basis = [math.pi * 2.0 * basis_scale]
            for _ in range(n_sin_basis - 1):
                basis.append(basis[-1] * 2.0)
            if trainable:
                self.register_parameter(
                    "basis", param=torch.nn.Parameter(torch.tensor(basis))
                )
            else:
                self.register_buffer("basis", torch.tensor(basis))

            self.has_sin_basis = True

        if sigmoid_centers:
            self.register_buffer(
                "sigmoid_orientations", torch.tensor(sigmoid_orientations)
            )
            self.register_buffer("sigmoid_centers", torch.tensor(sigmoid_centers))
            self.has_sigmoids = True

        self.dense = nn.Linear(
            n_sin_basis * 2 + n_sigmoid_centers, config.hidden_size, bias=False
        )
        self.shift = shift
        self.output_tensor = None

    def forward(self, x):
        batch_size, seq_length = x.size(0), x.size(1)
        x = x.reshape(-1)
        special_token_mask = x < 0
        special_token_indices = torch.nonzero(special_token_mask).squeeze(1)
        mapped_special_token_values = -(x[special_token_indices].long() + 1)
        special_token_embeds = self.special_token_embeddings(
            mapped_special_token_values
        ).squeeze(1)

        continuous_values = self.encode_cont_values(x)
        continuous_values[special_token_mask] = special_token_embeds.to(
            continuous_values.dtype
        )
        continuous_values = continuous_values.reshape(batch_size, seq_length, -1)
        return continuous_values

    def encode_cont_values(self, x):
        x = x - self.shift
        x_s = x[..., None]
        x = x_s * self.basis
        features = [torch.sin(x), torch.cos(x)] if self.has_sin_basis else []
        if self.has_sigmoids:
            features.append(
                torch.sigmoid(x_s * self.sigmoid_orientations - self.sigmoid_centers)
            )
        x = torch.cat(features, dim=-1)
        x = self.dense(x)

        return x
