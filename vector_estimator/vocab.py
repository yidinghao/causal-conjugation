"""
Code for working with word embeddings and embedding spaces. This script
was copied from another project, so some of the code might be taken out
of context.
"""
from collections.abc import Iterable
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn import preprocessing

from _utils import cache_pickle, timer

Array = Union[np.ndarray, torch.Tensor]


@cache_pickle
def load_embeddings(filename: str) -> Tuple[List[str], np.ndarray]:
    """
    Loads word embeddings from a .txt file in word2vec or GloVe
    format.

    :param filename: The name of the file containing the embeddings
    :return: A vocabulary of words, and an array of shape (num_words,
        embedding_size) where each row is the word embedding for the
        corresponding word in the vocabulary
    """
    with open(filename, "r") as f:
        all_lines = [line.strip().split(" ", 1) for line in f]
        if all_lines[0][0].isnumeric() and all_lines[0][1].isnumeric():
            all_lines.pop(0)
        words, vecs = zip(*all_lines)
    words = list(words)
    vecs = np.array([np.fromstring(v, sep=" ") for v in vecs])
    return words, vecs


def remove_single_direction(embeddings: np.ndarray, direction: np.ndarray) \
        -> np.ndarray:
    """
    Zeros out a single direction from a set of embeddings.

    :param embeddings: A set of word embeddings. Shape: (num_embeddings,
        embedding_size)
    :param direction: The direction to zero out. Shape:
        (embedding_size,)
    :return: The embeddings, with the specified direction zeroed out
    """
    projection = embeddings @ direction
    return embeddings - projection[:, np.newaxis] * direction


class Vocabulary(object):
    """
    A container for word embeddings.
    """

    def __init__(self, words: List[str], embeddings: np.ndarray,
                 normalize: bool = False):
        """
        Loads a list of words and their embeddings.

        :param words: A list of words
        :param embeddings: The embeddings for words
        :param normalize: If True, all embeddings will be normalized to
            unit length.
        """
        self._words = words
        self._indices = {w: i for i, w in enumerate(words)}
        if normalize:
            self.embeddings = preprocessing.normalize(embeddings)
        else:
            self.embeddings = embeddings

    @property
    def words(self) -> List[str]:
        return self._words

    @words.setter
    def words(self, word_list: List[str]):
        self._words = word_list
        self._indices = {w: i for i, w in enumerate(word_list)}

    @property
    def indices(self) -> Dict[str, int]:
        return self._indices

    @indices.setter
    def indices(self, index_dict: Dict[str, int]):
        self._indices = index_dict
        items = sorted(self._indices.items(), key=lambda x: x[1])
        self._words = [w for w, _ in items]

    def normalize(self):
        self.embeddings = preprocessing.normalize(self.embeddings)

    def filter_by_frequency(self, n: int = 50000, chinese: bool = False):
        """
        Filters out all but the top n most frequent words. Words with
        non-alphanumeric characters, except for "-", will be filtered
        out.

        :param n: The number of words to keep
        :param chinese: If True, then non-Chinese words will be filtered
            out instead of non-alphanumeric words.
        """
        idx = []
        for i, w in enumerate(self.words):
            if chinese and all(u"\u4e00" <= c <= u"\u9fff" for c in w):
                idx.append(i)
            elif not chinese and w.replace("-", "").isalnum():
                idx.append(i)
            if len(idx) >= n:
                break

        self.words = [self.words[i] for i in idx]
        self.embeddings = self.embeddings[idx]

    def filter_by_word_list(self, *words, normalize: bool = False) -> \
            "Vocabulary":
        idx = [self.indices[w] for w in words]
        words = [self.words[i] for i in idx]
        embeddings = self.embeddings[idx]
        return Vocabulary(words, embeddings, normalize=normalize)

    def to_file(self, filename: str):
        """
        Saves the Vocabulary to a file.

        :param filename: The name of the file to save to
        """
        with timer("Saving Vocabulary to {}...".format(filename)):
            with open(filename, "w") as f:
                f.write("{} {}\n".format(*self.embeddings.shape))
                for word, row in zip(self.words, self.embeddings):
                    row = [str(r) for r in row]
                    f.write("{} {}\n".format(word, " ".join(row)))

    @classmethod
    def from_file(cls, filename: Optional[str] = None, normalize: bool = False,
                  cache: Optional[str] = None):
        """
        Loads word embeddings from a .txt file in word2vec or GloVe
        format.

        :param filename: The name of the file containing the embeddings
        :param normalize: If True, all embeddings will be normalized to
            unit length
        :param cache: If an argument is provided, the embeddings will be
            cached to this filename
        :return: The loaded embeddings
        """
        if filename is None and cache is None:
            raise ValueError("filename and cache can't both be None.")
        words, vecs = load_embeddings(filename, cache_filename=cache)
        return cls(words, vecs, normalize=normalize)

    def __getitem__(self, item: Union[str, int, Iterable]) -> np.ndarray:
        """
        Retrieves one or more word embeddings.

        :param item: A word form, an index, or a list or tuple of word
            forms or indices
        :return: The corresponding word embeddings
        """
        if isinstance(item, int):
            return self.embeddings[item]
        elif isinstance(item, str):
            return self.embeddings[self.indices[item]]
        elif isinstance(item, Iterable):
            idx = [self.indices[i] if isinstance(i, str) else i
                   for i in item]
            return self.embeddings[idx]

        raise ValueError("{} is not a valid index for vocab of size {}"
                         "".format(item, len(self.words)))

    def __len__(self):
        return len(self.words)

    @property
    def size(self):
        return self.embeddings.shape[1]
