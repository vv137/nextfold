from functools import partial
import logging
from typing import Callable, List, Tuple, Sequence

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from scipy.stats import truncnorm


def _calculate_fan(linear_weight_shape: Tuple[int, int], fan: str = 'fan_in') -> float:
    fan_out, fan_in = linear_weight_shape

    if fan == 'fan_in':
        f = fan_in
    elif fan == 'fan_out':
        f = fan_out
    elif fan == 'fan_avg':
        f = 0.5 * (fan_in + fan_out)
    else:
        raise ValueError(f"Invalid fan option '{fan}'")

    return f


def trunc_normal_init_(tensor: Tensor, scale: float = 1.0, fan: str = 'fan_in') -> None:
    shape = tensor.shape

    f = _calculate_fan(shape, fan)
    scale = scale / np.max(1.0, f)
    a = -2.0
    b = 2.0
    std = np.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1.0)
    size = np.prod(shape, dtype='int')

    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)

    with torch.no_grad():
        tensor.copy_(torch.tensor(samples, device=tensor.device))


def lecun_normal_init_(tensor: Tensor) -> None:
    """
    The LeCun (fan-in) initialization strategy.
    ------
    Y. LeCun et al. Efficient backprop. Neural Networks: Tricks of the Trade. Springer, 1998.
    """
    trunc_normal_init_(tensor, scale=1.0)


def he_normal_init(tensor: Tensor) -> None:
    """
    The He initialization strategy.
    ------
    K. He et al. Delving deep into rectifiers: Surpassing human-level performance on imagenet
    classification. In Proceedings of the IEEE Conference on Computer Vision, p. 1026-1034, 2015.
    """
    trunc_normal_init_(tensor, scale=2.0)


def glorot_uniform_init_(tensor: Tensor) -> None:
    """
    The Glorot initialization strategy.
    ------
    X. Glorot et al. Understanding the difficulty of training deep feedforward neural networks.
    In Proceedings of the Thirteenth International Conference on Artificial Intelligence and
    Statistics, p. 249-256, 2010.
    """
    nn.init.xavier_normal_(tensor, gain=1.0)


def final_init_(tensor: Tensor) -> None:
    with torch.no_grad():
        tensor.fill_(0.0)


def gating_init_(tensor: Tensor) -> None:
    with torch.no_grad():
        tensor.fill_(0.0)
