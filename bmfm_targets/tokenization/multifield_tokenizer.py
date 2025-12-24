import atexit
import math
import os
import tempfile
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.models.bert import BertTokenizerFast
from transformers.tokenization_utils_base import (
    BatchEncoding,
    PaddingStrategy,
    TensorType,
    TruncationStrategy,
)
from transformers.utils import logging

from ..config.tokenization_config import FieldInfo
from . import MultiFieldVocabulary
from .multifield_instance import MultiFieldInstance

logger = logging.get_logger(__name__)


def _stringify(this_input, is_split_into_words=True):
    if isinstance(this_input, list):
        if len(this_input) > 0 and isinstance(this_input[0], str):
            return this_input
        return list(map(str, this_input))
    elif isinstance(this_input, str):
        if is_split_into_words:
            return [this_input]
        else:
            return this_input
    else:
        raise ValueError(f"Unsupported tokenization input type {type(this_input)}")


class MultiFieldTokenizer:
    """
    A tokenizer that can handle multiple fields of input.
    Class Attributes:
        tokenizers (dict[PreTrainedTokenizerFast|PreTrainedTokenizer]): The sub tokenizers.
        load_relative=True : load subtokenizers from a "tokenizers" sub directory next to the top tokenizer.
    """

    SUB_TOKENIZER_CLASS = BertTokenizerFast

    def __init__(
        self,
        name_or_path=None,
        field_to_tokenizer_map: dict[str, str | Path] = None,
        load_relative=True,
        default_field=None,
        **kwargs,
    ):
        if field_to_tokenizer_map:
            field_to_tokenizer_map = {
                key: str(value) for key, value in field_to_tokenizer_map.items()
            }
        elif not name_or_path:
            raise ValueError(
                "Multifield tokenizer init requires either a name_or_path or field_to_tokenizer_map"
            )
        else:
            load_relative = True

        self.base_dir = Path(name_or_path) / "tokenizers" if load_relative else None
        self.name_or_path = name_or_path
        self.field_to_tokenizer_map = field_to_tokenizer_map
        self.tokenizers = {}
        self.kwargs = kwargs
        self.default_field = default_field
        self.load_all_subtokenizers()

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        save_subtokenizers: bool = True,
        filename_prefix: str = None,
        **kwargs,
    ):
        if filename_prefix is not None:
            Warning(
                "file prefix no longer supported for multiFieldTokenizer and will be ignored"
            )
        save_directory = Path(save_directory)
        if save_subtokenizers:
            (save_directory / "tokenizers").mkdir(exist_ok=True, parents=True)
            for field_name in self.tokenizers.keys():
                self.get_field_tokenizer(field=field_name).save_pretrained(
                    save_directory=save_directory / "tokenizers" / field_name
                )

    @classmethod
    def from_pretrained(cls, name_or_path, **kwargs):
        kwargs.update({"local_files_only": True})

        return MultiFieldTokenizer(name_or_path=name_or_path, **kwargs)

    @classmethod
    def from_old_multifield_tokenizer(
        cls, name_or_path, save_converted_tokenizer_back=False, **kwargs
    ):
        """
        convert an old multifield tokenizer into the new format.

        Args:
        ----
            name_or_path (str): path to tokenizer dir
            save_converted_tokenizer_back (bool, optional): If True, saves the converted tokenizer in the directory of the input tokenizer. Defaults to False.

        Returns:
        -------
            MultiFieldTokenizer: the converted tokenizer

        """
        if not (Path(name_or_path) / "multifield_vocab.json").exists():
            raise ValueError(f"Can not find a multifield tokenizer in {name_or_path}")
        vocab = MultiFieldVocabulary.load_from_json(
            str(Path(name_or_path) / "multifield_vocab.json")
        )
        fields = vocab.get_fields()
        field_to_tokenizer_map = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir).mkdir(exist_ok=True)
            empty_vocab_file_path = Path(tmpdir) / "empty_vocab.txt"
            empty_vocab_file_path.touch(exist_ok=True)

            for field_name in fields:
                id_to_token = {
                    v[1]: k for k, v in vocab.token_to_ids[field_name].items()
                }
                fast_tokenizer = cls.SUB_TOKENIZER_CLASS(
                    vocab_file=empty_vocab_file_path
                )
                fast_tokenizer.add_tokens(
                    [id_to_token[i] for i in range(len(id_to_token))]
                )

                updated_vocab = fast_tokenizer.get_vocab()
                sorted_vocab = [
                    key
                    for key, idx in sorted(
                        updated_vocab.items(), key=lambda item: item[1]
                    )
                ]
                with open(f"{tmpdir}/created_vocab.txt", "w") as vocab_file:
                    vocab_file.write("\n".join(sorted_vocab) + "\n")

                updated_tokenizer = BertTokenizerFast(
                    vocab_file=f"{tmpdir}/created_vocab.txt"
                )
                field_to_tokenizer_map[field_name] = field_name

                updated_tokenizer.save_pretrained(
                    save_directory=Path(tmpdir) / "tokenizers" / field_name
                )
            full_tokenizer = cls.from_pretrained(name_or_path=tmpdir, **kwargs)
            if save_converted_tokenizer_back:
                full_tokenizer.save_pretrained(save_directory=name_or_path)
                full_tokenizer = cls.from_pretrained(name_or_path)
        return full_tokenizer

    @classmethod
    def convert_tokenizer_to_bert(
        cls, name_or_path, save_converted_tokenizer_back=False, **kwargs
    ):
        loaded_tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        if isinstance(loaded_tokenizer, cls.SUB_TOKENIZER_CLASS):  # nothing to do
            return loaded_tokenizer

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir).mkdir(exist_ok=True)
            empty_vocab_file_path = Path(tmpdir) / "empty_vocab.txt"
            empty_vocab_file_path.touch(exist_ok=True)

            vocab = loaded_tokenizer.get_vocab()
            id_to_token = {v: k for k, v in vocab.items()}
            fast_tokenizer = cls.SUB_TOKENIZER_CLASS(
                vocab_file=empty_vocab_file_path,
                **kwargs,
            )
            fast_tokenizer.add_tokens([id_to_token[i] for i in range(len(id_to_token))])

            updated_vocab = fast_tokenizer.get_vocab()
            sorted_vocab = [
                key
                for key, idx in sorted(updated_vocab.items(), key=lambda item: item[1])
            ]
            vocab_file_path = f"{tmpdir}/created_vocab.txt"
            with open(vocab_file_path, "w") as vocab_file:
                vocab_file.write("\n".join(sorted_vocab) + "\n")

            updated_tokenizer = cls.SUB_TOKENIZER_CLASS(vocab_file=vocab_file_path)
            save_directory = Path(tmpdir) / "updated_tokenizer"
            updated_tokenizer.save_pretrained(
                save_directory=save_directory,
            )
            full_tokenizer = cls.SUB_TOKENIZER_CLASS.from_pretrained(
                pretrained_model_name_or_path=save_directory, **kwargs
            )
            if save_converted_tokenizer_back:
                full_tokenizer.save_pretrained(save_directory=name_or_path)
                full_tokenizer = cls.SUB_TOKENIZER_CLASS.from_pretrained(name_or_path)
        return full_tokenizer

    @classmethod
    def from_sub_tokenizers(cls, field_to_tokenizer_map: dict = None, **kwargs):
        """
        note: this method should only be used for new configurations.  Otherwise, use
            multi_field_tokenizer = MultiFieldTokenizer.create_new_multi_field_tokenizer_from_sub_tokenizers(field_dictionary)
            multi_field_tokenizer.save_pretrained(path_for_new_tokenizer).

        just once, and from now on you can use
            multi_field_tokenizer = AutoTokenizer.from_pretraind(path_for_new_tokenizer)

        We meed a dictionary of field_name:field_tokenizer_path

        Args:
        ----
            mf_paths_dict (dict[field_name:str -> path_to_saved_tokenizer:str]): field to tokenizer dictionary.

        Returns:
        -------
            MultiFieldTokenizer: created tokenizer

        """
        if not isinstance(field_to_tokenizer_map, dict):
            raise ValueError(
                "multifield tokenizer must be created from a dictionary of tokenizers, got %s"
                % str(field_to_tokenizer_map),
            )
        if field_to_tokenizer_map:
            field_to_tokenizer_map = {
                key: str(value) for key, value in field_to_tokenizer_map.items()
            }
        new_multi_field_tokenizer = MultiFieldTokenizer(
            field_to_tokenizer_map=field_to_tokenizer_map,
            **kwargs,
            load_relative=False,
        )
        new_multi_field_tokenizer.load_all_subtokenizers()

        # to make sure that the tokenizer will save and then load properly, we save and load it and use the loaded version
        # this should take care of any potentioal issues due to this bootstrup process of creating a tokenizer
        tmpdir = tempfile.TemporaryDirectory()
        # the temprorary directory has to live untill the program exits, as we may need to re-read the file (for vocab reset)
        atexit.register(tmpdir.cleanup)
        new_multi_field_tokenizer.save_pretrained(tmpdir.name)

        multi_field_tokenizer: MultiFieldTokenizer = (
            MultiFieldTokenizer.from_pretrained(
                tmpdir.name,
                field_to_tokenizer_map=field_to_tokenizer_map,
                **kwargs,
                load_relative=True,
            )
        )
        return multi_field_tokenizer

    def load_subtokenizer(self, field_name: str, field_to_tokenizer_map=None):
        """
        load a sub tokenizer from disk.

        field_name:
            name (str): name of tokenizer to load
        """
        if field_to_tokenizer_map is None:
            field_to_tokenizer_map = self.field_to_tokenizer_map
            path = field_to_tokenizer_map.get(field_name)
            if self.base_dir:
                path = f"{self.base_dir}/{field_name}"
        else:
            path = field_to_tokenizer_map.get(field_name)
        if path is None:
            raise ValueError(
                "Failed to load tokenizer for field %s as no path is present in the subtokenizer dictionary",
                field_name,
            )
        try:
            loaded_tok = self.SUB_TOKENIZER_CLASS.from_pretrained(
                path,
                do_lower_case=False,
                tokenize_chinese_chars=False,
                clean_text=False,
                strip_accents=None,
            )
        except Exception as e:
            logger.error(
                f"failed to load {path} as a {self.SUB_TOKENIZER_CLASS.__name__}"
            )
            raise e

        self.tokenizers[field_name] = loaded_tok

    def load_all_subtokenizers(self):
        if self.base_dir:
            path = Path(self.base_dir)
            self.field_to_tokenizer_map = {}
            for file in path.iterdir():
                if file.is_dir():
                    name = file.name
                    self.field_to_tokenizer_map[name] = name
        for key in self.field_to_tokenizer_map:
            self.load_subtokenizer(key)
        self.sanitize_special_tokens()

    #### tokenizer functionality duplicated - only things that actually get used

    @property
    def _default_tokenizer(self):
        if self.default_field is not None:
            default_field_name = self.default_field
        else:
            default_field_name = sorted(self.tokenizers.keys())[0]
        return self.get_field_tokenizer(default_field_name)

    @property
    def mask_token(self):
        return self._default_tokenizer.mask_token

    @property
    def mask_token_id(self):
        return self._default_tokenizer.mask_token_id

    @property
    def cls_token_id(self):
        return self._default_tokenizer.cls_token_id

    @property
    def sep_token_id(self):
        return self._default_tokenizer.sep_token_id

    @property
    def unk_token_id(self):
        return self._default_tokenizer.unk_token_id

    @property
    def unk_token(self):
        return self._default_tokenizer.unk_token

    @property
    def pad_token_id(self):
        return self._default_tokenizer.pad_token_id

    @property
    def padding_side(self):
        return self._default_tokenizer.padding_side

    @property
    def pad_token(self):
        return self._default_tokenizer.pad_token

    @property
    def all_special_ids(self):
        return self._default_tokenizer.all_special_ids

    @property
    def all_special_tokens(self):
        return self._default_tokenizer.all_special_tokens

    @property
    def is_fast(self):
        return True

    def __call__(
        self,
        mfi: MultiFieldInstance | list[MultiFieldInstance],
        mfi_pair: MultiFieldInstance | list[MultiFieldInstance] | None = None,
        fields: list[FieldInfo] | None = None,
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool | str | TruncationStrategy = True,
        max_length: int | None = None,
        stride: int = 0,
        is_split_into_words: bool = True,  # for multifield type input
        pad_to_multiple_of: int | None = None,
        return_tensors: str | TensorType | None = None,
        return_token_type_ids: bool | None = None,
        return_attention_mask: bool | None = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = True,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        """
        Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
        sequences.

        Args:
        ----
            mfi (`MultiFieldInstance`, `List[MultiFieldInstance]`, `List[List[MultiFieldInstance]]`, *optional*):
                The sequence or batch of sequences to be encoded.
            mfi_pair (`MultiFieldInstance`, `List[MultiFieldInstance]`, `List[List[MultiFieldInstance]]`, *optional*):
                The sequence or batch of sequences to be encoded.
        """
        # To avoid duplicating
        all_kwargs = {
            "add_special_tokens": add_special_tokens,
            "padding": padding,
            "truncation": truncation,
            "max_length": max_length,
            "stride": stride,
            "is_split_into_words": is_split_into_words,
            "pad_to_multiple_of": pad_to_multiple_of,
            "return_tensors": return_tensors,
            "return_token_type_ids": return_token_type_ids,
            "return_attention_mask": return_attention_mask,
            "return_overflowing_tokens": return_overflowing_tokens,
            "return_special_tokens_mask": return_special_tokens_mask,
            "return_offsets_mapping": return_offsets_mapping,
            "return_length": return_length,
            "verbose": verbose,
        }
        all_kwargs.update(kwargs)
        if isinstance(mfi, MultiFieldInstance):
            mfi = [mfi]
        encodings = {}

        present_field_names = mfi[0].keys()
        active_fields = [f for f in fields if f.field_name in present_field_names]
        inactive_fields = [f for f in fields if f.field_name not in present_field_names]
        # only non input fields are permitted to be missing
        # if a field is expected as an input, it must be present
        assert not any(f.is_input for f in inactive_fields)

        for field in active_fields:
            if field.tokenization_strategy == "tokenize":
                field_tokenizer = self.get_field_tokenizer(field.field_name)
            else:
                field_tokenizer = self._default_tokenizer
            field_inputs = [
                [_stringify(x[field.field_name], is_split_into_words) for x in mfi]
            ]
            if mfi_pair is not None:
                field_inputs.append(
                    [
                        _stringify(x[field.field_name], is_split_into_words)
                        for x in mfi_pair
                    ]
                )
            field_encodings = field_tokenizer(*field_inputs, **all_kwargs)

            if field.tokenization_strategy == "continuous_value_encoder":
                field_encodings[
                    "input_ids"
                ] = self._replace_tokenized_ids_with_continuous_values(
                    mfi,
                    mfi_pair,
                    field=field,
                    input_ids=field_encodings["input_ids"].float(),
                    added_specials=field_encodings["special_tokens_mask"],
                    token_type_ids=field_encodings["token_type_ids"],
                )

            encodings[field.field_name] = field_encodings
        return BatchEncoding(data=encodings)

    def _replace_tokenized_ids_with_continuous_values(
        self,
        mfi: list[MultiFieldInstance],
        mfi_pair: list[MultiFieldInstance] | None,
        field: FieldInfo,
        input_ids: torch.Tensor,
        added_specials: torch.Tensor,
        token_type_ids: torch.Tensor,
    ):
        for this_mfi, token_type_id in [(mfi, 0), (mfi_pair, 1)]:
            if this_mfi is None:
                continue
            update_idx = ((~added_specials) & (token_type_ids == token_type_id)).bool()
            for sample_idx, x in enumerate(this_mfi):
                sample_update_idx = update_idx[sample_idx]
                trunc_sample_len = sample_update_idx.sum()
                arr = torch.tensor(
                    x[field.field_name][:trunc_sample_len], dtype=input_ids.dtype
                )
                input_ids[sample_idx][sample_update_idx] = arr

        return input_ids

    def add_tokens_for_field(self, field_name, token_to_add):
        self.get_field_tokenizer(field_name).add_tokens(token_to_add)

    def get_field_vocab(self, field_name: str) -> dict[str, int]:
        """
        Gets the tokens for a given field.

        Args:
        ----
            field (str): The key for the field.

        Returns:
        -------
            dict[str, int]: The tokens for the field, with their ids

        """
        tokenizer: Tokenizer = self.tokenizers[field_name]
        return tokenizer.get_vocab()

    def field_vocab_size(self, name):
        return len(self.get_field_tokenizer(name))

    def get_field_tokenizer(
        self, field: str
    ) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        """
        get a specific sub tokenizer.

        Args:
        ----
            field (str): name of field

        Returns:
        -------
            Tokenizer: tokenizer for this field

        """
        return self.tokenizers[field]

    def reset_tokenizer_vocab(self, field_name: str):
        """
        reload a tokenizer from disk.

        Args:
        ----
            field_name (str): name of field for which the subtokenizer will be reset.

        """
        self.load_subtokenizer(field_name)

    def get_special_token_binary_mask(self, field: str) -> list[bool]:
        """
        Returns the special token mask for a given field.
        Used (only) in test_metrics.  TODO: is this needed?.

        Args:
        ----
            field (str): The field to get the special token mask for.

        Returns:
        -------
            List[bool]: The special token mask for the given field.

        """
        if field not in self.tokenizers.keys():
            raise ValueError(
                f"field {field} not found in the vocabulary. Available fields: {self.tokenizers.keys()}"
            )
        special_tokens = self.all_special_ids
        return [
            token_id in special_tokens
            for token_id in sorted(self.get_field_vocab(field).values())
        ]

    def convert_field_ids_to_tokens(
        self,
        ids: int | list[int],
        skip_special_tokens: bool = False,
        field: str = "expressions",
    ) -> str | list[str]:
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
        added tokens.

        Args:
        ----
            ids (`int` or `List[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
        -------
            `str` or `List[str]`: The decoded token(s).

        """
        return self.get_field_tokenizer(field).convert_ids_to_tokens(
            ids, skip_special_tokens
        )

    def convert_field_tokens_to_ids(self, field: str, tokens: list[str]) -> list[int]:
        """
        Converts a list of tokens to their corresponding IDs.

        Args:
        ----
            tokens (Union[str, List[str]]): The tokens to convert.
            field (str): The field the tokens belong to.

        Returns:
        -------
            int or List[int]: The IDs of the tokens.

        """
        return self.get_field_tokenizer(field).convert_tokens_to_ids(tokens=tokens)

    def get_token_values(self, field: str) -> list[float] | None:
        """
        Get float values for field, if available.

        Note: list must be all floats so it can be copied to cuda device

        Args:
        ----
            field (str): tokenizer field

        Returns:
        -------
            list[float] | None: list of float values with NaN for special tokens and
                numbers for other tokens or None if ANY of the non-special tokens are
                not castable to numbers

        """
        if field not in self.tokenizers:
            return None
        field_tokens = self.get_field_vocab(field)

        def num(t):
            try:
                return float(int(t))
            except:
                return float("nan")

        reversed_tokens = {v: k for k, v in field_tokens.items()}
        token_values = [num(reversed_tokens[t]) for t in sorted(reversed_tokens.keys())]
        if sum(map(math.isnan, token_values)) > len(self.all_special_tokens):
            logger.warn("Field %s cannot be converted to float values.", field)
            return None
        return token_values

    def sanitize_special_tokens(self, all_special_tokens_must_match=False):
        """
        Sanitizes the special tokens.  Mostly just verifies that all the special token ids match.

        Raises
        ------
            ValueError: If the special tokens are not the same for all fields.

        """
        if all_special_tokens_must_match:
            special_token_ids = [
                self.get_field_tokenizer(field_name).all_special_ids
                for field_name in self.tokenizers.keys()
            ]
            special_token_ids.append(self.all_special_ids)
            from itertools import product

            for i, j in product(special_token_ids, repeat=2):
                assert i == j


def set_custom_cls_post_processor(
    subtok: PreTrainedTokenizerFast, cls_tokens: list[str]
):
    """
    Set a post-processor that prepends custom CLS tokens to every sequence.

    Modifies tokenizer in-place.

    Args:
    ----
      tokenizer: the subtokenizer (not multifield tokenizer) must be "Fast"
      cls_tokens: the list of CLS-like tokens to include must already be present in the vocab

    """
    sep = subtok.sep_token or "[SEP]"
    cls_section = " ".join(cls_tokens)

    # single sequence: CLS_1 CLS_2 ... $A SEP
    single_template = f"{cls_section} $A {sep}"
    # pair sequence: CLS_1 CLS_2 ... $A SEP $B SEP
    pair_template = f"{cls_section} $A {sep} $B {sep}"
    specials = list(zip(subtok.all_special_tokens, subtok.all_special_ids))
    subtok.backend_tokenizer.post_processor = TemplateProcessing(
        single=single_template, pair=pair_template, special_tokens=specials
    )  # pyright: ignore[reportAttributeAccessIssue]
