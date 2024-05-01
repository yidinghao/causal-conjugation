import json
from typing import Dict, List, Optional, Tuple, Union

import torch
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, \
    BatchEncoding

from conditions import TrainCondition

# Type annotation for tokenizers
Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

# Cached datasets
_rc_dataset: Dict[Tuple[str, str], Dataset] = dict()
_agreement_dataset: Dict[str, Dataset] = dict()

rc_types = ["ORC", "ORRC", "SRC", "PRC", "PRRC"]


def _tokenize_rc(tokenizer: Tokenizer, example) -> BatchEncoding:
    prefix_2 = "" if example["Prefix2"] == " " else " " + example["Prefix2"]
    sentence = \
        "{} {}{} {} {}".format(example["Prefix1"], example["Subject"],
                               prefix_2, example["RC"], example["Suffix"])
    tokenized = tokenizer(sentence, return_tensors="pt")

    # Mark the positions of the input. 1 means it's inside the RC, 3
    # means it's a singular subject, 4 means it's a plural subject, and
    # all other tokens are 2.
    tokenized["label"] = \
        [2 for _ in tokenizer.tokenize(example["Prefix1"])] + \
        [4 if example["SubjIsPlur"] else 3
         for _ in tokenizer.tokenize(example["Subject"])] + \
        [2 for _ in tokenizer.tokenize(prefix_2)] + \
        [2 if example["RCType"] in ("O_Control", "S_Control") else 1
         for _ in tokenizer.tokenize(example["RC"])] + \
        [2 for _ in tokenizer.tokenize(example["Suffix"])]

    return tokenized


def _tokenize_agreement(tokenizer: Tokenizer,
                        example) -> BatchEncoding:
    """
    Tokenizes a single example from Marvin and Linzen (2018) by adding a
    [MASK] token between the Prefix and Suffix.

    :param tokenizer: The tokenizer used for tokenization
    :param example: The example to be tokenized
    :return: A prepared Huggingface batch
    """
    rc = "" if example["RC"] == " " else " " + example["RC"]
    sentence = \
        "{} {}{} {} {}".format(example["Prefix1"], example["Subject"],
                               rc, tokenizer.mask_token, example["Suffix"])

    tokenized = tokenizer(sentence, return_tensors="pt")
    tokenized["label"] = \
        [2 for _ in tokenizer.tokenize(example["Prefix1"])] + \
        [4 if example["SubjIsPlur"] else 3
         for _ in tokenizer.tokenize(example["Subject"])] + \
        [1 for _ in tokenizer.tokenize(example["RC"])] + \
        [0] + \
        [2 for _ in tokenizer.tokenize(example["Suffix"])]
    return tokenized


def _collate(batch_raw: List[Dict[str, torch.Tensor]]) -> \
        Dict[str, torch.Tensor]:
    keys = batch_raw[0].keys()
    return {k: pad_sequence([b[k].squeeze().unsqueeze(-1) for b in batch_raw],
                            batch_first=True).squeeze(-1) for k in keys}


def load_train_data(
        tokenizer: Tokenizer, batch_size: int = 32,
        rc_type: Optional[str] = None,
        train_condition: TrainCondition = TrainCondition.SUBJ) -> DataLoader:
    """

    :param tokenizer:
    :param batch_size:
    :param rc_type:
    :param train_condition:
    :return:
    """
    # Try looking up a cached dataset first...
    global _rc_dataset
    tokenizer_name = tokenizer.name_or_path
    if (tokenizer_name, train_condition) in _rc_dataset:
        dataset = _rc_dataset[(tokenizer_name, train_condition)]
    else:
        # ...otherwise just load from file
        with open("paths.json", "r") as f:
            paths_file = json.load(f)
        filename = paths_file["train_data_paths"][train_condition.name]
        dataset = load_dataset("csv", data_files=filename, skiprows=1,
                               column_names=["Prefix1", "Subject", "Prefix2",
                                             "RC", "Suffix", "RCType", "By?",
                                             "That?", "SubjIsPlur"])["train"]
        dataset = dataset.map(lambda e: _tokenize_rc(tokenizer, e))
        _rc_dataset[(tokenizer_name, train_condition)] = dataset

    # Filter by relative clause type
    if rc_type is not None:
        control = "O_Control" if rc_type in ("ORC", "ORRC") else "S_Control"
        dataset = dataset.filter(lambda e: e["RCType"] in (rc_type, control))

    # Convert to PyTorch DataLoader
    dataset.set_format(type="torch",
                       columns=["input_ids", "token_type_ids",
                                "attention_mask", "label"])
    return DataLoader(dataset, batch_size=batch_size, collate_fn=_collate)


def load_test_data(tokenizer: Tokenizer, batch_size: int = 32,
                   rc_type: Optional[str] = None) -> DataLoader:
    """
    Loads the subject-verb agreement test set.

    :param tokenizer:
    :param batch_size:
    :param rc_type:
    :return:
    """
    # Try looking up a cached dataset first...
    global _agreement_dataset
    tokenizer_name = tokenizer.name_or_path
    if tokenizer_name in _agreement_dataset:
        dataset = _agreement_dataset[tokenizer_name]
    else:
        # ...otherwise just load from file
        dataset = load_dataset("csv", skiprows=1,
                               data_files="data/test.csv",
                               column_names=["Prefix1", "Subject", "RC",
                                             "Suffix", "RCType", "SubjIsPlur",
                                             "AttrIsPlur"])["train"]
        dataset = dataset.map(lambda e: _tokenize_agreement(tokenizer, e))
        _agreement_dataset[tokenizer_name] = dataset

    # Filter by relative clause type
    if rc_type is not None:
        dataset = dataset.filter(lambda e: e["RCType"] == rc_type)

    # Convert to PyTorch DataLoader
    dataset.set_format(type="torch",
                       columns=["input_ids", "token_type_ids",
                                "attention_mask", "label"])
    return DataLoader(dataset, batch_size=batch_size, collate_fn=_collate)
