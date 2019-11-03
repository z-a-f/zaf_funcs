import copy
import torch
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


class Transformation:
  def __init__(self, ts=None):
    self.ts = ts
  def __call__(self, sample):
    for t in self.ts:
      sample = t(sample)
    return sample
  def __iter__(self):
    for t in self.ts:
      yield t
  def __getitem__(self, idx):
    return self.ts[idx]


class ToTensor:
  def __init__(self, device='cpu', deepcopy=True):
    self.device = device
    self.deepcopy = deepcopy
  def __call__(self, sample):
    if self.deepcopy:
      sample = copy.deepcopy(sample)
    sample['image'] = torch.tensor(sample['image']).permute((0, 3, 1, 2))
    sample['image'] = sample['image'].to(torch.float)
    sample['image'] = sample['image'].contiguous().to(self.device)
    sample['label'] = torch.tensor(sample['label']).contiguous()
#     sample['label'] = sample['label'].to(torch.float)
    sample['label'] = sample['label'].to(self.device)
    return sample

class RandomFlip:
  def __init__(self, p=0.5, axis=2):
    self.p = p
    self.axis = axis
  def __call__(self, sample):
    flip = np.random.random(sample['image'].shape[0]) >= self.p
    sample['image'][flip] = torch.flip(sample['image'][flip],
                                       (self.axis,))
    return sample

class Normalize:
  def __init__(self, mean, std):
    self.mean = mean
    self.std = std
  def __call__(self, sample):
    sample['image'] = sample['image'] - self.mean
    sample['image'] = sample['image'] / self.std
    return sample
  def denormalize(self, sample):
    if isinstance(sample, dict):
      img = sample['image']
    else:
      img = sample
    img = img * self.std
    img = img + self.mean
    return img

class Tuplize:
  def __call__(self, sample):
    return sample['image'], sample['label']
