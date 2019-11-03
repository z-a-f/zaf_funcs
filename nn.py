import torch
import torch.nn as nn

class Flatten(nn.Module):
  """PyTorch layer to flatten the inputs."""
  def forward(self, x):
    return x.view(x.size()[0], -1)