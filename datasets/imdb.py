import sys, os

from tensorflow.keras import datasets as tf_ds

from ._base import _BaseDataset

class IMDB(_BaseDataset):
  LABELS = ("Negative", "Positive")
  """IMDB sentiment analysis dataset.

  Constructor args:
    path: where to cache the dataset
    num_words: max number of words to include. Only the most frequent words are
               included.
    to_torch: Convert the data to torch Tensor
    **kwargs: all arguments are passed through to the tf.keras.imdb

  Methods:
    load: loads the imdb data
    decode: decodes an "indexed" review into a sentence
    word_to_idx / idx_to_word: Converts between index and word.
  Properties:
    train: train data
    test: test data
  """
  def __init__(self, path=None, num_words=10000, **kwargs):
    if path is None:
      path = os.path.join(os.getcwd(), "data/IMDB")
    super().__init__(path)
    self.load(num_words, **kwargs)

  def load(self, num_words, **kwargs):
    """Loads the dataset from keras.

    Args:
      num_words: max number of words to include. Only the most frequent words are
                 included.
      to_torch: Convert the data to torch Tensor
      **kwargs: all arguments are passed through to the tf.keras.imdb
    Returns:
      2 tuples of Numpy arrays: (x_train, y_train), (x_test, y_test)
    """
    self._loader(tf_ds.imdb, num_words, **kwargs)

  def decode(self, review):
    """Decodes an encoded review into a word sentence.
    Unknown words are be marked as '?'
    """
    return ' '.join([self.idx_to_word(idx-3, '?') for idx in review])

  # Helpers
  def word_to_idx(self, word):
    """Converts a word to appropriate index."""
    return self._word_index.get(word, None)
  def idx_to_word(self, idx, unknown_word=None):
    """Converts an index to an appropriate word."""
    return self._reverse_word_index.get(idx, unknown_word)

  def _loader(self, loader, num_words, **kwargs):
    self._train, self._test = loader.load_data(path=self.path,
                                               num_words=num_words,
                                               **kwargs)

    self._word_index = loader.get_word_index(path=self.path+"_word_index.json")
    self._reverse_word_index = dict([(value, key)
                                  for (key, value) in self._word_index.items()])

