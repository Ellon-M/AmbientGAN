## Author: 2022 Ellon


# Lint as python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.backends import cudnn

def main():
  # fast training
  cudnn.benchmark = True

  sagan = SAGAN()

  # make these arg params
  sagan.train(1000, 32, 5)


main()

