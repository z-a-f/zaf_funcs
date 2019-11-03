# Helper stuff for ML

## Contents:

### `helpers.py`

- `Fitter` -- class to help fitting and evaluating a network

### `imagenet_utils.py`

- This is a copy from https://github.com/pytorch/examples/blob/master/imagenet/main.py, so it falls under BSD 3-Clause License, and was developed by the PyTorch community

### `misc.py`

- Miscellaneous function:
   - `output_side_conv2d` -- computes the side of a convolutional layer
   - `output_shape_conv2d` -- computes the shape of the output of a conv2d layer

### `nn.py`

- Some extra layers, s.a. `Flatten`

### `datasets`

- `imdb.py` -- IMDB dataset
- `reuters.py` -- News sentiment dataset
- `tin.py` -- Tiny imagenet dataset (from CS231n)

### `transforms`

- Some data transformations
