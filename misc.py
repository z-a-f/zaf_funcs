import numpy as np

def output_side_conv2d(side, k, s, p, d):
  return int(np.floor((side + 2 * p - d * (k - 1) - 1)/s + 1))

def output_shape_conv2d(N, Cin, H, W, Cout, k, s=[1, 1], p=[0, 0], d=[1, 1]):
  oH = output_side_conv2d(H, k[0], s[0], p[0], d[0])
  oW = output_side_conv2d(W, k[1], s[1], p[1], d[1])
  return N, Cout, oH, oW
