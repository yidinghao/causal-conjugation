"""
Computing a gender vector using SVMs based on Ravfogel et al. (2020).
"""
import random
from typing import List, Optional, Tuple

import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC

from _utils import timer, random_seed
from vector_estimator.base import VectorEstimator
from vector_estimator.vocab import Vocabulary

SVMData = List[Tuple[str, int]]


def prepare_dataset(m_words: List[str], f_words: List[str],
                    embeddings: Vocabulary, seed: Optional[int] = None) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a dataset for training an SVM to classify word embeddings as
    "masculine" or "feminine."

    :param m_words: The masculine words
    :param f_words: The feminine words
    :param embeddings: The word embeddings
    :param seed: A random seed to control the shuffling of the data
    :return: The embeddings for m_words and f_words and their labels
    """
    all_words = [(w, 0) for w in m_words] + [(w, 1) for w in f_words]
    with random_seed(seed):
        random.shuffle(all_words)
    words, classes = zip(*all_words)
    return embeddings[words], np.array(classes)


def train_svm(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
              test_y: np.ndarray) -> Tuple[LinearSVC, float]:
    """
    Trains a linear SVM to classify word embeddings as "masculine" or
    "feminine."

    :param train_x: The training inputs
    :param train_y: The training labels
    :param test_x: The testing inputs
    :param test_y: The testing labels
    :return: The trained SVM and its testing accuracy
    """
    svm = LinearSVC(fit_intercept=False, class_weight=None, dual=False,
                    random_state=0)
    svm.fit(train_x, train_y)
    return svm, svm.score(test_x, test_y)


class SVMEstimator(VectorEstimator):
    """
    Calculates a gender vector using an SVM based on Ravfogel et al.'s
    (2020) algorithm.
    """

    def __init__(self, m_words: List[str], f_words: List[str],
                 seed: Optional[int] = None):
        """
        Generates training and testing data for the SVM from a set of
        embeddings based on the he - she vector.

        :param m_words: The male words
        :param f_words: The female words
        :param seed: A random seed to use when shuffling the data
        """
        self.m_words = m_words
        self.f_words = f_words
        self.labels: Optional[np.ndarray] = None
        self.train_test_split: float = .7
        self.seed = seed

    def _get_reference_vectors(self, embeddings: Vocabulary) -> np.ndarray:
        """
        Retrieves the embeddings for classification from a given set of
        embeddings.

        :param embeddings: The embeddings that the SVM will classify as
            "masculine" or "feminine." They do not have to be the same
            as the embeddings used to generate the dataset.
        :return: The embeddings for the words appearing in the dataset
        """
        reference, labels = prepare_dataset(self.m_words, self.f_words,
                                            embeddings, seed=self.seed)
        self.labels = labels
        return reference

    def _get_single_vector(self, reference_vectors: np.ndarray) -> np.ndarray:
        """
        Trains an SVM to classify embeddings as "masculine" or
        "feminine" and then take the 1-dimensional weight matrix to be
        the gender vector.

        :param reference_vectors: The embeddings used for training and
            testing the SVM
        :return: The weight vector of the trained SVM
        """
        split = int(self.train_test_split * len(reference_vectors))

        train_x = reference_vectors[:split]
        test_x = reference_vectors[split:]
        train_y = self.labels[:split]
        test_y = self.labels[split:]

        with timer("Training SVM..."):
            svm, accuracy = train_svm(train_x, train_y, test_x, test_y)
        print("Accuracy: {:.1f}%".format(accuracy * 100))

        return preprocessing.normalize(svm.coef_)
