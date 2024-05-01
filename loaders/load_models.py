"""
Loads Huggingface stuff and caches it to a global variable so that it
doesn't have to be loaded multiple times.
"""
from typing import Dict, Union

from transformers import BertTokenizerFast, BertForMaskedLM, BertConfig

from _utils import timer
from models.alterable_bert import AlterableBertForMaskedLM

Bert = Union[BertForMaskedLM, AlterableBertForMaskedLM]
_bert_models: Dict[str, Bert] = dict()
_bert_tokenizers: Dict[str, BertTokenizerFast] = dict()
_bert_n_layers: Dict[str, int] = dict()
_bert_hidden_size: Dict[str, int] = dict()


def reset_model_cache():
    global _bert_models
    _bert_models = dict()


def load_bert_model(model_name: str, alterable: bool = True) -> Bert:
    """
    Loads a Huggingface model and caches it to a global variable.

    :param model_name: The model's Huggingface name
    :param alterable: If True, an AlterableBertForMaskedLM will be
        loaded; otherwise, a BertForMaskedLM will be loaded
    :return: The specified model
    """
    global _bert_models
    model_key = ("alt_" if alterable else "") + model_name

    with timer("Loading BERT model {}...".format(model_name)):
        if model_key in _bert_models:
            return _bert_models[model_key]
        elif alterable:
            model = AlterableBertForMaskedLM.from_pretrained(model_name)
        else:
            model = BertForMaskedLM.from_pretrained(model_name)
        _bert_models[model_key] = model
        return model


def load_bert_tokenizer(model_name: str) -> BertTokenizerFast:
    """
    Loads a Huggingface tokenizer and caches it to a global variable.

    :param model_name: The tokenizer's Huggingface name
    :return: The specified tokenizer
    """
    global _bert_tokenizers
    with timer("Loading BERT tokenizer {}...".format(model_name)):
        if model_name in _bert_tokenizers:
            return _bert_tokenizers[model_name]
        else:
            tokenizer = BertTokenizerFast.from_pretrained(model_name)
            _bert_tokenizers[model_name] = tokenizer
            return tokenizer


def num_layers(model_name: str) -> int:
    """
    Retrieves the number of layers in a BERT model.

    :param model_name: The model's Huggingface name
    :return: The number of layers in the model
    """
    global _bert_models, _bert_n_layers

    # First check to see if n_layers is already cached
    if model_name in _bert_n_layers:
        return _bert_n_layers[model_name]

    # If not, extract it from a BERT model or BertConfig
    if model_name in _bert_models:
        n_layers = _bert_models[model_name].config.num_hidden_layers + 1
    else:
        n_layers = BertConfig.from_pretrained(model_name).num_hidden_layers + 1

    _bert_n_layers[model_name] = n_layers
    return n_layers


def hidden_size(model_name: str) -> int:
    """
    Retrieves the hidden size of a BERT model.

    :param model_name: The model's Huggingface name
    :return: The model's hidden size
    """
    global _bert_models, _bert_hidden_size

    # First check to see if n_layers is already cached
    if model_name in _bert_hidden_size:
        return _bert_hidden_size[model_name]

    # If not, extract it from a BERT model or BertConfig
    if model_name in _bert_models:
        hidden_size_ = _bert_models[model_name].config.hidden_size
    else:
        hidden_size_ = BertConfig.from_pretrained(model_name).hidden_size

    _bert_hidden_size[model_name] = hidden_size_
    return hidden_size_
