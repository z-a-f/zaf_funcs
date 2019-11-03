import PIL
import imageio
import numpy as np
import os

from collections import defaultdict
from torch.utils.data import Dataset

from tqdm.autonotebook import tqdm

dir_structure_help = r"""
TinyImageNetPath
├── test
│   └── images
│       ├── test_0.JPEG
│       ├── t...
│       └── ...
├── train
│   ├── n01443537
│   │   ├── images
│   │   │   ├── n01443537_0.JPEG
│   │   │   ├── n...
│   │   │   └── ...
│   │   └── n01443537_boxes.txt
│   ├── n01629819
│   │   ├── images
│   │   │   ├── n01629819_0.JPEG
│   │   │   ├── n...
│   │   │   └── ...
│   │   └── n01629819_boxes.txt
│   ├── n...
│   │   ├── images
│   │   │   ├── ...
│   │   │   └── ...
├── val
│   ├── images
│   │   ├── val_0.JPEG
│   │   ├── v...
│   │   └── ...
│   └── val_annotations.txt
├── wnids.txt
└── words.txt
"""

def download_and_unzip(URL, root_dir):
  error_message = "Download is not yet implemented. Please, go to {URL} urself."
  raise NotImplementedError(error_message.format(URL))

def _add_channels(img, total_channels=3):
  while len(img.shape) < 3:  # third axis is the channels
    img = np.expand_dims(img, axis=-1)
  while(img.shape[-1]) < 3:
    img = np.concatenate([img, img[:, :, -1:]], axis=-1)
  return img

"""Creates a paths datastructure for the tiny imagenet.

DO NOT INSTANTIATE THIS!
This class is protected in order to enable the singleton.

Args:
  root_dir: Where the data is located
  download: Download if the data is not there

Members:
  label_id:
  ids:
  nit_to_words:
  data_dict:

"""
class _TinyImageNetPaths(object):
  def __init__(self, root_dir, download=False):
    if download:
      download_and_unzip('http://cs231n.stanford.edu/tiny-imagenet-200.zip',
                         root_dir)
    train_path = os.path.join(root_dir, 'train')
    val_path = os.path.join(root_dir, 'val')
    test_path = os.path.join(root_dir, 'test')

    wnids_path = os.path.join(root_dir, 'wnids.txt')
    words_path = os.path.join(root_dir, 'words.txt')

    self._make_paths(train_path, val_path, test_path,
                     wnids_path, words_path)

  def _make_paths(self, train_path, val_path, test_path,
                  wnids_path, words_path):
    self.ids = []
    with open(wnids_path, 'r') as idf:
      for nid in idf:
        nid = nid.strip()
        self.ids.append(nid)
    self.nid_to_words = defaultdict(list)
    with open(words_path, 'r') as wf:
      for line in wf:
        nid, labels = line.split('\t')
        labels = list(map(lambda x: x.strip(), labels.split(',')))
        self.nid_to_words[nid].extend(labels)

    self.paths = {
      'train': [],  # [img_path, id, nid, box]
      'val': [],  # [img_path, id, nid, box]
      'test': []  # img_path
    }

    # Get the test paths
    self.paths['test'] = list(map(lambda x: os.path.join(test_path, x),
                                      os.listdir(test_path)))
    # Get the validation paths and labels
    with open(os.path.join(val_path, 'val_annotations.txt')) as valf:
      for line in valf:
        fname, nid, x0, y0, x1, y1 = line.split()
        fname = os.path.join(val_path, 'images', fname)
        bbox = int(x0), int(y0), int(x1), int(y1)
        label_id = self.ids.index(nid)
        self.paths['val'].append((fname, label_id, nid, bbox))

    # Get the training paths
    train_nids = os.listdir(train_path)
    for nid in train_nids:
      anno_path = os.path.join(train_path, nid, nid+'_boxes.txt')
      imgs_path = os.path.join(train_path, nid, 'images')
      label_id = self.ids.index(nid)
      with open(anno_path, 'r') as annof:
        for line in annof:
          fname, x0, y0, x1, y1 = line.split()
          fname = os.path.join(imgs_path, fname)
          bbox = int(x0), int(y0), int(x1), int(y1)
          self.paths['train'].append((fname, label_id, nid, bbox))


class TinyImageNetPaths(object):
  instance = None
  def __new__(cls, *args, **kwargs):
    if not TinyImageNetPaths.instance:
      TinyImageNetPaths.instance = _TinyImageNetPaths(*args, **kwargs)
    return TinyImageNetPaths.instance
  def __getattr__(self, name):
    return getattr(self.instance, name, value)
  def __setattr__(self, name):
    return setattr(self.instance, name, value)


class _Loader:
  """Basic data loader."""
  def __init__(self, tin_dataset, batch_size):
    self.tin = tin_dataset
    if not self.tin.preload:
      raise NotImplementedError("Cannot use batch loader without preloading")
    self.batch_size = batch_size
  def __len__(self):
    batch_num = len(self.tin.img_data) // self.batch_size
    if self.batch_size * batch_num < len(self.tin.img_data):
      batch_num += 1
    return batch_num
  def __iter__(self):
    idx = 0
    while idx < len(self.tin):
      if self.batch_size == 1:
        yield self.tin[idx]
      else:
        imgs = self.tin.img_data[idx:idx+self.batch_size]
        lbls = None if self.tin.mode == 'test' else self.tin.label_data[idx:idx+self.batch_size]
        sample = {'image': imgs, 'label': lbls}
        if self.tin.transform and self.tin._train:
          sample = self.tin.transform(sample)
        yield self.tin.final_transform(sample)
      idx += self.batch_size

"""Datastructure for the tiny image dataset.

Args:
  root_dir: Root directory for the data
  mode: One of "train", "test", or "val"
  preload: Preload into memory
  load_transform: Transformation to use at the preload time
  transform: Transformation to use at the retrieval time
  download: Download the dataset
  max_samples: Maximum number of samples to get

Members:
  tinp: Instance of the TinyImageNetPaths
  img_data: Image data
  label_data: Label data
"""
class TinyImageNetDataset(Dataset):
  def __init__(self, root_dir, mode='train', preload=True, load_transform=None,
               transform=None, download=False, max_samples=None,
               randomize=False, final_transform=None):
    tinp = TinyImageNetPaths(root_dir, download)
    self.mode = mode
    self.label_idx = 1  # from [image, id, nid, box]
    self.preload = preload
    self.transform = transform
    self.final_transform = final_transform
    if self.final_transform is None:
      self.final_transform = lambda x: x

    self.IMAGE_SHAPE = (64, 64, 3)

    self.img_data = []
    self.label_data = []

    self.max_samples = max_samples
    self.samples = tinp.paths[mode]
    self.samples_num = len(self.samples)

    self._train = (self.mode != 'test')

    if randomize:
      self.samples = np.random.permutation(self.samples)

    if self.max_samples is not None:
      self.samples_num = min(self.max_samples, self.samples_num)
      self.samples = self.samples[:self.samples_num]

    if self.preload:
      load_desc = "Preloading {} data...".format(self.mode)
      self.img_data = np.zeros((self.samples_num,) + self.IMAGE_SHAPE,
                               dtype=np.float32)
      self.label_data = np.zeros((self.samples_num,), dtype=np.int)
      for idx in tqdm(range(self.samples_num), desc=load_desc):
        s = self.samples[idx]
        img = imageio.imread(s[0])
        img = _add_channels(img)
        self.img_data[idx] = img
        if self.mode != 'test':
          self.label_data[idx] = s[self.label_idx]

      if load_transform:
        result = load_transform({'image': self.img_data,
                                 'label': self.label_data})
        self.img_data = result['image']
        self.label_data = result['label']

  def __len__(self):
    return self.samples_num

  def __getitem__(self, idx):
    if self.preload:
      img = self.img_data[idx]
      lbl = None if self.mode == 'test' else self.label_data[idx]
    else:
      s = self.samples[idx]
      img = imageio.imread(s[0])
      lbl = None if self.mode == 'test' else s[self.label_idx]
    img = img.reshape((1, *img.shape))
    lbl = lbl.reshape((1, *lbl.shape))
    sample = {'image': img, 'label': lbl}

    if self.transform and self._train:
      sample = self.transform(sample)
    return self.final_transform(sample)

  def batch_loader(self, batch_size):
    return _Loader(self, batch_size)

  def eval(self):
    self._train = False
  def train(self):
    if self.mode == 'test':
      raise RuntimeError("Cannot train on test data")
    self._train = True

  def cuda(self):
    self.img_data = self.img_data.cuda()
    self.label_data = self.label_data.cuda()
    return self
  def cpu(self):
    self.img_data = self.img_data.cpu()
    self.label_data = self.label_data.cpu()
    return self
