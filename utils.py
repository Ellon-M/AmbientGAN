import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



def normalize(x, axis=-1, order=2):
    """Normalizes a Numpy array.

  Args:
      x: Numpy array to normalize.
      axis: axis along which to normalize.
      order: Normalization order (e.g. `order=2` for L2 norm).

  Returns:
      A normalized copy of the array.

  """

  l2 = np.atleast_1d(np.linalg.norm(x, order, axis))
  l2[l2 == 0] = 1

  return x / np.expand_dims(l2, axis)


# to_categorical
def to_categorical(y, num_classes=None, dtype='float32'):
    """ Converts a class vector (integers) to binary class matrix.

    Args:
      y: Array-like with class values to be converted into a matrix
          (integers from 0 to `num_classes - 1`).
      num_classes: Total number of classes. If `None`, this would be inferred
        as `max(y) + 1`.
      dtype: The data type expected by the input. Default: `'float32'`.

      Returns:
      A binary matrix representation of the input. The class axis is placed
      last.

      """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()

    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    
    return categorical


def make_folder(path, version):
    if not os.path.exists(os.path.join(path, version)):
        os.makedirs(os.path.join(path, version))


def tensor2var(x, grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=grad)

def var2tensor(x):
    return x.data.cpu()

def var2numpy(x):
    return x.data.cpu().numpy()

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)
