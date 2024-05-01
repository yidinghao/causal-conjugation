"""
Code for estimating feature representations using INLP.
"""
import os
import pickle
import random
from pathlib import Path
from typing import Optional, Union

import torch
from scipy.linalg import orth

from _get_vectors import _get_bert_vectors
from _utils import timer
from conditions import TrainCondition
from loaders.load_models import hidden_size
from vector_estimator.svm import SVMEstimator

Dir = Union[str, Path]


def _get_pickle_filename(dir_: Dir, model_name: str, rc_type: str, layer: int,
                         train_condition: TrainCondition) -> Path:
    """
    Generates a filename for a Pickled Vocabulary containing vectors
    extracted from BERT.
    """
    if isinstance(dir_, str):
        dir_ = Path(dir_)
    return dir_ / "{}_{}_vectors_{}_layer_{}.p".format(
        model_name.replace("/", "_"), train_condition.name, rc_type, layer)


def get_vectors(model_name: str, rc_type: str, layer: int,
                output_directory: Optional[Dir] = None,
                train_condition: TrainCondition = TrainCondition.SUBJ):
    """
    Extracts vectors from a BERT model using a training dataset.
    """
    # Loads pickled vectors if available
    if output_directory is not None:
        output_filename = _get_pickle_filename(
            output_directory, model_name, rc_type, layer, train_condition)

        if os.path.isfile(output_filename):
            with timer("Loading vectors from {}...".format(output_filename)):
                with open(output_filename, "rb") as f:
                    return pickle.load(f)

    # Run BERT and extract the vectors
    subj_vectors = _get_bert_vectors(
        model_name, rc_type, {3: "singular", 4: "plural"},
        train_condition=train_condition)

    # Save vectors from all layers to Pickle
    if output_directory is not None:
        for l_, vocab in enumerate(subj_vectors):
            output_filename = _get_pickle_filename(
                output_directory, model_name, rc_type, l_, train_condition)

            with timer("Saving Layer {} vectors to {}..."
                       "".format(l_, output_filename)):
                with open(output_filename, "wb") as o:
                    pickle.dump(vocab, o)

    return subj_vectors[layer]


def train(directory: Union[str, Path], model_name: str, rc_type: str,
          layer: int, estimator_type: type = SVMEstimator,
          train_condition: TrainCondition = TrainCondition.SUBJ,
          num_vecs: int = 8, num_words: Optional[int] = 2000) -> torch.Tensor:
    """
    Estimates a representation subspace for grammatical number.

    :param directory: The directory where pickle files for layer
        representations are stored
    :param model_name: The Huggingface name of the model that computed
        the layer representations
    :param rc_type: The relative clause type used to compute the layer
        representations
    :param layer: The layer from which the representations were
        extracted (0 is the embedding layer)

    :param estimator_type: The method used to estimate the subspace
    :param train_condition: If True, the verb number feature will
        be estimated from the subject token(s); otherwise, it will be
        estimated from the [MASK] token
    :param num_vecs: The dimensionality of the subspace to be computed
    :param num_words: If supplied, this many singular and plural words
        will be sampled in order to compute the subspace, as opposed to
        using all the words

    :return: An orthonormal subspace for the estimated representation
        subspace, of shape [num_vecs, hidden_size]
    """
    if train_condition == TrainCondition.RANDOM:
        return torch.randn(num_vecs, hidden_size(model_name),
                           device="cuda" if torch.cuda.is_available() else
                           "cpu")

    vocab = get_vectors(model_name, rc_type, layer,
                        train_condition=train_condition,
                        output_directory=directory)

    sg_words = [w for w in vocab.words if w.startswith("singular")]
    pl_words = [w for w in vocab.words if w.startswith("plural")]

    if num_words is not None:
        random.shuffle(sg_words)
        sg_words = sg_words[:num_words]

        random.shuffle(pl_words)
        pl_words = pl_words[:num_words]

    estimator = estimator_type(sg_words, pl_words)
    vecs = torch.Tensor(orth(estimator.get_vectors(vocab, n=num_vecs).T).T)
    if torch.cuda.is_available():
        return vecs.to("cuda")
    return vecs
