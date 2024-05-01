"""
An interface for algorithms that estimate a direction in a word
embedding space.
"""
from abc import ABC, abstractmethod

import numpy as np

from vector_estimator.vocab import Vocabulary, remove_single_direction


class VectorEstimator(ABC):
    """
    Base class for vector estimation algorithms.
    """

    @abstractmethod
    def _get_single_vector(self, reference_vectors: np.ndarray) -> np.ndarray:
        """
        Computes a single direction based on a set of "reference
        vectors."

        :param reference_vectors: A set of vectors from which to compute
            a direction
        :return: The computed direction. Shape: (embedding_size,)
        """
        pass

    @abstractmethod
    def _get_reference_vectors(self, embeddings: Vocabulary) -> np.ndarray:
        """
        Selects a set of "reference vectors" from a set of embeddings,
        which will be used to compute directions.

        :param embeddings: The embeddings to select the reference
            vectors from
        :return: The reference vectors. Shape: (num_reference_vectors,
            embedding_size)
        """
        pass

    def get_vectors(self, embeddings: Vocabulary, n: int = 1) -> np.ndarray:
        """
        By default, multiple directions are generated according to an
        iterative procedure. In each iteration, a single direction is
        calculated from a set of "reference vectors." This direction is
        then removed from the reference vectors before the next
        iteration.

        :param embeddings: The embeddings to compute a direction from
        :param n: The number of directions to compute
        :return: The computed directions. Shape: (n, embedding_size)
        """
        reference = self._get_reference_vectors(embeddings)
        vectors = []
        for _ in range(n):
            v = self._get_single_vector(reference)
            vectors.append(v)
            reference = remove_single_direction(reference, v.squeeze())
        return np.concatenate(vectors, axis=0)
