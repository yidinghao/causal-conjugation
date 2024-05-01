"""
Script for testing subject-verb agreement with an alterable BERT.
"""
import math
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertTokenizer, BertTokenizerFast, BertForMaskedLM

from _utils import print_delay, timer
from conditions import TestCondition
from loaders.load_data import load_test_data
from models.alterable_bert import AlterableBertForMaskedLM

Bert = Union[BertForMaskedLM, AlterableBertForMaskedLM]
Tokenizer = Union[BertTokenizer, BertTokenizerFast]

verb_idx = {2001, 2015, 2020, 2066, 2293, 5223, 7459, 7777, 16424, 19837}
nouns_and_verbs = {103, 2001, 2015, 2020, 2051, 2066, 2111, 2208, 2245, 2265,
                   2293, 2299, 2336, 2338, 2399, 2704, 2774, 2808, 2961,
                   3008, 3065, 3110, 3117, 3166, 3185, 3208, 3237, 3353,
                   3457, 3738, 3836, 3861, 4062, 4169, 4405, 4620, 4932,
                   4944, 5089, 5205, 5223, 5265, 5691, 5878, 5961, 6002,
                   6048, 6304, 6569, 6617, 6687, 6853, 7459, 7500, 7767,
                   7777, 8013, 8033, 8160, 8221, 8930, 9431, 9760, 10026,
                   10095, 10153, 10487, 10489, 12706, 13448, 15893, 16424,
                   16804, 16838, 18815, 19837, 22283, 24789, 25375, 27828}


def test(model: Bert, tokenizer: Tokenizer, rc_type: str,
         alpha: float = 4., alter_layer: Optional[int] = None,
         test_condition: TestCondition = TestCondition.GLOBAL,
         alter_dimensions: Optional[torch.Tensor] = None) -> Tuple[float, ...]:
    """
    Tests subject-verb agreement for one RC type and one altered layer.
    No dependencies on other code.

    :param model:
    :param tokenizer:
    :param rc_type:
    :param alpha:
    :param alter_layer:
    :param test_condition:
    :param alter_dimensions:
    :return: The subject-verb agreement accuracy
    """
    is_index = tokenizer.vocab["is"]
    are_index = tokenizer.vocab["are"]
    mask_index = tokenizer.mask_token_id
    pad_index = tokenizer.pad_token_id
    loss_function = nn.CrossEntropyLoss(ignore_index=pad_index,
                                        reduction="sum")

    with timer("Loading test data..."):
        test_data = load_test_data(tokenizer, rc_type=rc_type)

    num_correct = 0
    num_total = 0
    num_tokens = 0
    total_loss = 0.
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
    model.eval()
    print_delay("Starting test.")
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_data), total=len(test_data)):
            # Put everything on Cuda
            if torch.cuda.is_available():
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to("cuda")

            # Extract gold labels and alter mask
            labels = (batch["label"] > 3).max(dim=-1).values

            # Calculate perplexity mask
            perplexity_mask = \
                1 - sum(batch["input_ids"] == w for w in nouns_and_verbs)

            # Calculate alter mask
            if test_condition == TestCondition.LOCAL:
                alter_mask = F.pad((batch["label"] >= 3).int(), (1, 1),
                                   value=0)
            elif test_condition == TestCondition.VERB:
                subj_mask = F.pad((batch["label"] >= 3).int(), (1, 1), value=0)
                verb_mask = sum(batch["input_ids"] == v for v in verb_idx)
                alter_mask = subj_mask + verb_mask
            elif test_condition == TestCondition.MASK:
                alter_mask = (batch["input_ids"] == mask_index).int(),
            elif test_condition == TestCondition.CONTROL:
                alter_mask = perplexity_mask
            else:
                alter_mask = None

            del batch["label"]

            # Extract model output
            mask_idx = (batch["input_ids"] == mask_index).nonzero(
                as_tuple=True)
            output = model(**batch, alter_layer=alter_layer, alpha=alpha,
                           alter_mask=alter_mask,
                           remove_features=alter_dimensions)

            # Calculate perplexity
            vocab_size = output.logits.shape[-1]
            logits = output.logits.view(-1, vocab_size)
            mlm_targets = (batch["input_ids"] * perplexity_mask).view(-1)
            total_loss += float(loss_function(logits, mlm_targets))
            num_tokens += int(perplexity_mask.sum())

            # Calculate accuracy
            output = output.logits[:, :, (is_index, are_index)][mask_idx]
            num_correct += int((labels == output.argmax(axis=-1)).sum())
            num_total += len(labels)

    accuracy = num_correct / num_total
    perplexity = math.exp(total_loss / num_tokens)
    return accuracy, perplexity
