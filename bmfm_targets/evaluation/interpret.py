import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from captum.attr import LayerIntegratedGradients
from pydantic import BaseModel
from scipy.stats import ttest_1samp
from transformers import PreTrainedModel

from bmfm_targets.config.model_config import SCModelConfigBase
from bmfm_targets.models import get_model_from_config, instantiate_classification_model
from bmfm_targets.tokenization import MultiFieldTokenizer
from bmfm_targets.training.metrics import (
    get_loss_tasks,
)


class LabelAttribution(BaseModel):
    label_name: None | str
    attributions: list[tuple[str, float]]

    def to_df(self, drop_duplicates=True):
        s = pd.DataFrame(self.attributions).set_index(0).squeeze()
        if drop_duplicates:
            s = s[~s.index.duplicated()]
        return s


class LabelColumnAttribution(BaseModel):
    label_column_name: str
    pred_label: float | str
    label_attributions: list[LabelAttribution]


class SampleAttribution(BaseModel):
    name: str
    label_column_attributions: list[LabelColumnAttribution]


class SequenceClassificationAttributionModule(pl.LightningModule):
    def __init__(
        self,
        model_config: SCModelConfigBase,
        tokenizer: MultiFieldTokenizer,
        label_dict: dict[str, dict[str, int]],
        attribute_kwargs: dict[str, Any] | None = None,
        attribute_filter: dict[str, Any] | None = None,
        modeling_strategy: Literal["multitask"]
        | Literal["sequence_classification"] = "sequence_classification",
        **kwargs,
    ):
        """
        Pytorch Lightning module for training a masked language model.

        Args:
        ----
            model_config_dict (dict): Dictionary containing the model configuration.
            trainer_config (TrainerConfig): Training configuration.
            label_dict (dict[dict[str, int]]): a nested label dictionary

        """
        super().__init__()

        self.model_config = model_config
        self.tokenizer = tokenizer
        self.label_dict = label_dict
        self.label_output_size_dict = {i: len(k) for i, k in self.label_dict.items()}
        self.modeling_strategy = modeling_strategy
        if "trainer_config" in kwargs:
            losses = kwargs["trainer_config"].losses
        else:
            losses = []

        self.loss_tasks = get_loss_tasks(
            losses,
            label_columns=self.model_config.label_columns,
        )
        if self.modeling_strategy == "sequence_classification":
            self.model = instantiate_classification_model(
                self.model_config, self.loss_tasks
            )
        elif self.modeling_strategy == "multitask":
            self.model = get_model_from_config(
                self.model_config, modeling_strategy=self.modeling_strategy
            )
        else:
            raise ValueError(f"Interpret not supported for {modeling_strategy}")
        self.train_labels = [*self.label_dict.keys()]
        self.attribute_kwargs = attribute_kwargs if attribute_kwargs is not None else {}
        self.attribute_filter = attribute_filter if attribute_filter is not None else {}

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> SampleAttribution:
        """
        Run attribution algorithm for batch of single sample.

        Args:
        ----
            batch (dict[str,torch.tensor]): batch from DataLoader of a single sample

        Returns:
        -------
            SampleAttribution: attribution for sample
        """
        with torch.enable_grad():
            single_attribution = get_sample_attribution(
                self.model,
                self.tokenizer,
                self.train_labels,
                self.label_dict,
                batch,
                self.attribute_filter,
                **self.attribute_kwargs,
            )

        return single_attribution


def save_sample_attributions(
    sample_attributions: list[SampleAttribution], ofname: str | Path
):
    """
    Save sample attributions to json.

    Args:
    ----
        sample_attributions (list[SampleAttribution]): list of SampleAttribution objs
        ofname (str | Path): output filename  (must end in .json)

    Raises:
    ------
        NotImplementedError: extensions other than .json raise an error
    """
    ofname = Path(ofname)
    if ofname.suffix != ".json":
        raise NotImplementedError("Only save to json is implemented")
    with open(ofname, "w") as f:
        json.dump([sample.model_dump() for sample in sample_attributions], f)


def summarize_attributions(attributions: torch.Tensor) -> torch.Tensor:
    """
    Summarize contributions from different dimensions of attribution tensors.

    Args:
    ----
        attributions (torch.Tensor): attributions tensors

    Returns:
    -------
        torch.Tensor: averaged attribution tensor
    """
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


def replace_non_special_tokens_with_pad(
    input_ids_tensor: torch.Tensor,
    cls_token_id: int,
    sep_token_id: int,
    pad_token_id: int,
) -> torch.Tensor:
    # Create a mask for non-special tokens
    non_special_mask = (
        (input_ids_tensor != cls_token_id)
        & (input_ids_tensor != sep_token_id)
        & (input_ids_tensor != pad_token_id)
    )

    result_tensor = input_ids_tensor.clone()

    # Replace non-special tokens with PAD token ID in the copy
    result_tensor[non_special_mask] = pad_token_id

    return result_tensor


def get_sample_attribution(
    model: PreTrainedModel,
    tokenizer: MultiFieldTokenizer,
    label_columns: list[str],
    label_dict: dict[str, dict[str, int]],
    batch: dict,
    attribute_filter: dict,
    **attribute_kwargs,
) -> SampleAttribution:
    """
    Get attributions for all outputs for a given sample.

    Args:
    ----
        model (PreTrainedModel): sequence classification model
        tokenizer (MultiFieldTokenizer): tokenizer for reconstructing gene names
        label_columns (list[str]): label_column_name to examine
        label_dict (dict[str,dict[str, int]]): label dict for all valid label_columns
        batch (dict): single sample batch dict

    Returns:
    -------
        SampleAttribution: all attributions for sample
    """
    reverse_label_dict = {
        l: {v: k for k, v in d.items()} for l, d in label_dict.items()
    }
    input_ids = batch["input_ids"].to(model.device)
    attention_mask = batch["attention_mask"].to(model.device)
    cell_name = batch["cell_name"][0]
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    def prediction(label_column_name, logits):
        if len(reverse_label_dict[label_column_name]) > 1:
            index_pred = torch.argmax(logits, dim=-1).squeeze().item()
            return reverse_label_dict[label_column_name][index_pred]
        else:
            return logits.squeeze().item()

    if isinstance(logits, dict):
        pred_label = {l: prediction(l, logits[l]) for l in label_columns}
    else:
        pred_label = {label_columns[0]: prediction(label_columns[0], logits)}

    attribute_filter_final = {n: list(label_dict[n].keys()) for n in label_columns}
    attribute_filter_final.update(attribute_filter)

    label_column_attributions = [
        {
            "label_column_name": this_label_column_name,
            "pred_label": pred_label[this_label_column_name],
            "label_attributions": [
                {
                    "label_name": label_name
                    if len(label_dict[this_label_column_name]) > 1
                    else None,
                    "attributions": get_single_label_attributions_for_sample(
                        model,
                        input_ids,
                        attention_mask,
                        label_id=label_id,
                        tokenizer=tokenizer,
                        label_column_name=this_label_column_name,
                        **attribute_kwargs,
                    ),
                }
                for label_name, label_id in label_dict[this_label_column_name].items()
                if label_name in attribute_filter_final[this_label_column_name]
            ],
        }
        for this_label_column_name in label_columns
    ]

    sample_attribution = SampleAttribution(
        name=cell_name, label_column_attributions=label_column_attributions
    )

    return sample_attribution


def get_single_label_attributions_for_sample(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    label_id: int,
    tokenizer: MultiFieldTokenizer,
    label_column_name: str | None = None,
    **attribute_kwargs,
) -> list[tuple[str, float]]:
    """
    Generate attributions for single label_id for sample.

    This method uses `captum`'s LinearIntegratedGradient to get attributions
    for genes and expressions for a given predicted label_id.

    Args:
    ----
        model (PreTrainedModel): the pretrained model, must be a sequence classification model
        input_ids (torch.Tensor): the sample's already tokenized input_ids, must be a single sample only
        attention_mask (torch.Tensor): the attention mask for the sample, must be a single sample only
        label_id (int): the id of the output to check attribution for
        tokenizer (MultiFieldTokenizer): the tokenizer to translate the ids back to gene names
        label_column_name: label_column_name - necessary for multitask models where logits
          are returned as a dict

    Returns:
    -------
        list[tuple[str, float]]: list of gene name, attribution tuples for sample
    """

    def predict(inputs, attention_mask=None):
        output = model(
            input_ids=inputs,
            attention_mask=attention_mask,
        )
        if isinstance(output.logits, dict):
            return output.logits[label_column_name]
        return output.logits

    ref_input_ids = replace_non_special_tokens_with_pad(
        input_ids,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
    )
    model_embeddings = model.__getattr__(model.base_model_prefix).embeddings
    lig_genes_embeddings = LayerIntegratedGradients(
        predict, model_embeddings.genes_embeddings
    )
    lig_expressions_embeddings = LayerIntegratedGradients(
        predict, model_embeddings.expressions_embeddings
    )
    attributions_genes = lig_genes_embeddings.attribute(
        input_ids,
        baselines=ref_input_ids,
        target=label_id,
        additional_forward_args=(attention_mask,),
        **attribute_kwargs,
    )
    attributions_expressions = lig_expressions_embeddings.attribute(
        input_ids,
        baselines=ref_input_ids,
        target=label_id,
        additional_forward_args=(attention_mask,),
        **attribute_kwargs,
    )
    attributions_summary = (
        summarize_attributions(attributions_genes + attributions_expressions)
        .detach()
        .cpu()
        .numpy()
    )
    gene_attributions = list(
        zip(
            tokenizer.convert_field_ids_to_tokens(input_ids[0][0], field="genes"),
            attributions_summary,
        )
    )

    return gene_attributions


def load_attributions(json_file: str | Path) -> list[SampleAttribution]:
    """Load SampleAttribution objects from json file."""
    return [SampleAttribution(**x) for x in json.loads(Path(json_file).read_text())]


def join_sample_attributions(
    all_samples: list[SampleAttribution], label_idx=0
) -> pd.DataFrame:
    """
    Join list of SampleAttributions to single DataFrame for a given label_idx.

    For a population of samples across a given label_idx, this joins the attributions
    into a single DataFrame. This is somewhat non-trivial because the list of genes
    is different from sample to sample, and there are duplicate gene names due to the
    presence of [PAD] and [UNK]

    Args:
    ----
        all_samples (list[SampleAttribution]): a list of SampleAttribution objects
        label_idx (int, optional): the label index to use. The attributions will be
          different for each label_idx.

    Returns:
    -------
        pd.DataFrame: DataFrame indexed by gene name with columns named `s0`, `s1` etc
          and values equal to the attributions measured or NaN if not present in the
          sample
    """
    if all_samples[0].name is None:
        keys = [f"s{i}" for i in range(len(all_samples))]
    else:
        keys = [s.name for s in all_samples]
    joined_samples = pd.concat(
        [
            pd.DataFrame(
                {
                    ls.label_column_name: ls.label_attributions[label_idx].to_df(
                        drop_duplicates=True
                    )
                    for ls in s.label_column_attributions
                }
            )
            for s in all_samples
        ],
        axis=1,
        keys=keys,
    )
    return joined_samples


def get_mean_attributions(
    attribution_df: pd.DataFrame, alpha=0.05, significance_attr=0.1
) -> pd.DataFrame:
    """
    Calculate mean attributions for every gene across samples including significance testing.

    Args:
    ----
        attribution_df (pd.DataFrame): DataFrame generated by join_sample_attributions or similar
        alpha (float, optional): p-value significance threshold - will be divided by N for
            Bonferroni correction. Defaults to 0.05.
        significance_attr (float, optional): Minimum absolute value of attribution to be considered
            "significant". Defaults to 0.1.

    Returns:
    -------
        pd.DataFrame: a DataFrame with mean attributions per gene, p-value and a column
           named "highlight" for p-values that are lower than the Bonferroni corrected
           threshold and have absolute valued average attribution larger than the
           significance threshold
    """
    p_values = attribution_df.T.apply(lambda x: ttest_1samp(x.dropna(), 0)).T[1]
    mean_attr = attribution_df.mean(axis=1).rename("attribution")
    mean_attr = mean_attr.to_frame().assign(p_value=p_values).dropna()
    mean_attr["-log2p"] = -np.log2(mean_attr.p_value)
    mean_attr = mean_attr.assign(
        highlight=(mean_attr.p_value < alpha / len(mean_attr))
        & (mean_attr.attribution.abs() > significance_attr)
    )
    return mean_attr
