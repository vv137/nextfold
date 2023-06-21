from typing import Any, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

from .core import ensure_divisibility

### Utils
