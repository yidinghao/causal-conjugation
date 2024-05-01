"""
A helper function that runs a BERT model and extracts vectors in
specified positions. The code might be a bit convoluted since it was
written at an earlier stage of the project.
"""
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from _utils import print_delay
from conditions import TrainCondition
from loaders.load_data import load_train_data
from loaders.load_models import load_bert_model, load_bert_tokenizer, \
    num_layers
from vector_estimator.vocab import Vocabulary


def _get_bert_vectors(
        model_name: str, rc_type: str, conditions: Dict[int, str],
        train_condition: TrainCondition = TrainCondition.SUBJ) \
        -> Tuple[Vocabulary, ...]:
    """
    Extracts contextualized representations of words from layers of BERT
    or some other Huggingface model. The script works by reading the RC
    datasets and extracting representations for tokens indicated by the
    labels provided in the dataset. Which label values to extract is
    given by conditions (see below).

    :param model_name: The name of the Huggingface model to extract
        representations from
    :param rc_type: Which type of relative clause to use data from
    :param conditions: Each token of each input sentence is assigned a
        numerical label from 0 to 4 as follows:
            0 represents padding symbols
            1 represents tokens inside a relative clause
            2 represents all other tokens except 1, 3, and 4
            3 represents a singular matrix subject
            4 represents a plural matrix subject.
        This parameter allows the user to specify which types of tokens
        should have their representations extracted, indicated by the
        keys of the dict. Each type of token can additionally be
        assigned a name, indicated by the values of the dict. The output
        Vocabulary will only contain representations for token types
        indicated by the keys of this dict, and each token's name will
        reflect the names given by the values of this dict
    :param train_condition: If True, the verb number feature will be
        estimated from the subject token(s); otherwise, it will be
        estimated from the [MASK] token

    :return: Vocabularys containing extracted representations from each
        layer, starting from the embedding + positional layer
    """
    tokenizer = load_bert_tokenizer(model_name)
    model = load_bert_model(model_name)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    n_layers = num_layers(model_name)

    all_words: List[str] = []
    all_vectors: Tuple[List[torch.Tensor]] = tuple([] for _ in range(n_layers))
    print_delay("Extracting vectors for model {}...".format(model_name))

    rc_data = load_train_data(tokenizer, rc_type=rc_type, batch_size=5,
                              train_condition=train_condition)

    with torch.no_grad():
        for batch in tqdm(rc_data):
            # Put everything on gpu
            if torch.cuda.is_available():
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to("cuda")

            # Extract RC position
            labels = batch["label"]
            del batch["label"]

            # Extract model representations
            output = model(**batch, output_hidden_states=True)
            vectors = [output.hidden_states[l_].detach() for l_ in
                       range(n_layers)]
            del output

            # Get words (flattened)
            mask_idx = (labels != 0).nonzero(as_tuple=True)
            idx = batch["input_ids"][:, 1:-1][mask_idx]

            vectors: List[torch.Tensor] = \
                [v[:, 1:-1][mask_idx].detach() for v in vectors]
            words: List[str] = tokenizer.convert_ids_to_tokens(idx)
            labels: List[int] = [int(l_) for l_ in labels[mask_idx]]

            # Separate out sentences
            sentences: List[List[str]] = []
            sentence_labels: List[List[int]] = []
            sentence_vectors: List[Tuple[torch.Tensor]] = []

            j = 0
            for i, w in enumerate(words):
                if w == ".":
                    sentences.append(words[j:i + 1])
                    sentence_labels.append(labels[j:i + 1])
                    sentence_vectors.append(tuple(v[j:i + 1] for v in vectors))
                    j = i + 1

            # Find the relevant words and their vectors
            for s, s_labels, s_vecs in \
                    zip(sentences, sentence_labels, sentence_vectors):
                s_idx = []
                for i, (w, l_) in enumerate(zip(s, s_labels)):
                    if l_ in conditions:
                        context = [s[j] if j != i else "_" for j in
                                   range(len(s))]
                        word_name = \
                            "{}_{}_{}".format(conditions[l_], w, context)

                        s_idx.append(i)
                        all_words.append(word_name)

                for vs, v in zip(all_vectors, s_vecs):
                    vs.append(v[s_idx])

    return tuple(Vocabulary(all_words, torch.cat(vs, dim=0).cpu().numpy())
                 for vs in all_vectors)
