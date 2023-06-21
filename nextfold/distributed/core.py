import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)

import torch.distributed as dist

_AsyncHandle = dist.Work | None
_Group = dist.ProcessGroup | None

# Data parallel group that the current rank belongs to
DATA_PARALLEL_GROUP = None
# Intra-layer model parallel group that the currnet rank belongs to
TENSOR_MODE_PARALLEL_GROUP = None


def ensure_divisibility(n: int, d: int) -> bool:
    """
    Ensure that numerator is divided by the denominator.
    """
    return n % d == 0


def set_missing_environ(environ: Dict[str, str]) -> None:
    for k, v in environ.items():
        if k not in os.environ:
            os.environ[str(k)] = str(v)


def init_distributed(tensor_mode_parallel_size: int = 1, backend: str | dist.Backend | None = 'mpi') -> None:
    """
    Initialize distributed environment.
    """

    if not dist.is_available():
        raise

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if not ensure_divisibility(world_size, tensor_mode_parallel_size):
        raise

    dist.init_process_group(
        backend=backend,
        init_method='env://',
    )

    if not dist.is_initialized():
        raise