import numpy as np

def get_mean(data, axis=None):
  if axis is None:
    axis = data.ndim-1
  x = np.array(data)
  all_idx = set(range(data.ndim))
  mean_idx = tuple(all_idx - set([axis]))
  return data.mean(axis=mean_idx)

class SubtractMean:
  """Subtracts the mean along the axis angle.

  Args:
    mean: Mean to use. If None, the mean is computed from the input images.
  """
  def __init__(self, mean=None, axis=-1):
    self.axis = axis
    self.mean = None

  def __call__(self, x, y):
    if self.axis < 0:
      self.axis = x.ndim + self.axis
    if self.mean is None:
      self.mean = get_mean(x, self.axis)
    return (x - self.mean), y, {self.__class__.__name__: self.mean}

class Transpose:
  """Transposes the input.

  Args:
    axis: New axis after the transpose
  """
  def __init__(self, axis=None):
    self.axis = axis

  def __call__(self, x, y):
    if self.axis is None:
      return x, y, dict()
    axis = tuple(self.axis)
    return np.transpose(x, axis), y, {}  # No need to save the transpose info.
