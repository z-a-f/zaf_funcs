
import sys, os
import abc

class _BaseDataset(abc.ABC, object):
  def __init__(self, path=None):
    self._train = None
    self._test = None
    self.path = path
    if self.path is None:
      self.path = os.path.join(os.getcwd(), "data")

  @abc.abstractmethod
  def load(self, *args, **kwargs):
    """Must set _test and _train."""
    pass

  @property
  def train(self):
    return self._train
  @train.setter
  def train(self, _):
    raise AttributeError("You cannot assign to the training data! Use 'load'!")

  @property
  def test(self):
    return self._test
  @test.setter
  def test(self, _):
    raise AttributeError("You cannot assign to the test data! Use 'load'!")
