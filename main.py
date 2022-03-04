## Author: 2022 Ellon


# Lint as python3
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
from ambientnetwork import SAGAN
from torch.backends import cudnn


def params():
    """ Model arguments passed before training

    """
    parser = argparse.ArgumentParser()

    # train setting
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--progress-interval' type=int, default=5)

    return parser.parse_args()


config = params()


def main(config):

  # fast training
  cudnn.benchmark = True

  sagan = SAGAN(config)

  sagan.train()


print(config)
main(config)

