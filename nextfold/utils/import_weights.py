"""
Jax parameter importer for AlphaFold
"""

from dataclasses import dataclass
from enum import Enum
from functools import partial
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

import numpy as np
import torch
from omegaconf import DictConfig

_NPZ_KEY_PREFIX = "alphafold/alphafold_iteration"


def _model_name_to_model_npz(model_name: str):
    without_prefix = re.findall(r'alphafold_(.*)', model_name)[0]
    return f"params_{without_prefix}.npz"


class ParameterType(Enum):
    LinearWeight = partial(lambda w: w.transpose(-1, -2))
    #
    Other = partial(lambda w: w)

    def __init__(self, func) -> None:
        self.transformation = func


@dataclass
class Parameter:
    param: torch.Tensor | List[torch.Tensor]
    param_type: ParameterType = ParameterType.Other
    stacked: bool = False


def assign(translation_dict: Dict[str, Any], tensors: torch.Tensor) -> None:
    for k, v in translation_dict.items():
        with torch.no_grad():
            tensor = torch.as_tensor(tensors[k])
            ref, param_type = v.param, v.param_type
            if v.stacked:
                tensor = torch.unbind(tensor, 0)
            else:
                tensor = [tensor]
                ref = [ref]

            try:
                tensor = list(map(param_type.transformation, tensor))
                for p, w in zip(ref, tensor):
                    p.copy_(w)
            except:
                logger.error(f"For processing '{k}', " f"'{ref[0].shape}' != '{tensor[0].shape}'")
                raise


def stack(param_dict_list: List[Dict[str, Any]], out: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Args:
        param_dict_list: A`list` of (nested) `dict`s of `Parameter` to stack.
        The structure of each `dict` must be the identical.
        There must be at least one `dict` in the `list`.
    """
    if out is None:
        out = dict()
    template = param_dict_list[0]
    for k in template:
        v = [d[k] for d in param_dict_list]
        if type(v[0]) is dict:
            out[k] = dict()
            stack(v, out=out[k])
        elif type(v[0]) is Parameter:
            stacked_param = Parameter(
                param=[param.param for param in v],
                param_type=v[0].param_type,
                stacked=True,
            )
            out[k] = stacked_param
    return out


class AlphaFoldWeightImporter:
    def __init__(
        self,
        npz_path: os.PathLike,
        cfg: DictConfig,
    ) -> None:
        self.npz_path = Path(npz_path)
        self.cfg = cfg

        self._load_npz()

    def _get_translation_dict(self, model: torch.nn.Module) -> Dict[str, Any]:
        return {}

    def _process_translation_dict(self, d: Dict[str, Any], top_layer: bool = True) -> Dict[str, Any]:
        flat = {}
        for k, v in d.items():
            if type(v) == dict:
                prefix = _NPZ_KEY_PREFIX if top_layer else ""
                sub_flat = {(prefix + '/'.join([k, kp])): vp
                            for kp, vp in self._process_translation_dict(v, top_layer=False).items()}
            else:
                k = '/' + k if not top_layer else k
                flat[k] = v
        return flat

    def _load_npz(self):
        if not self.npz_path.exists():
            raise FileNotFoundError(f"'{self.npz_path}' not exists")
        self.data = np.load(self.npz_path)
        logger.info(f"Load '{self.npz_path}'")

    def assign(self, model: torch.nn.Module) -> None:
        """
        Assign JAX weights to the model.
        """

        # Flatten keys and insert missing key prefixes
        trans = self._get_translation_dict(model)
        flat = self._process_translation_dict(trans)

        # Sanity check
        keys = list(self.data.keys())
        flat_keys = list(flat.keys())
        incorrect = [k for k in flat_keys if k not in keys]
        missing = [k for k in keys if k not in flat_keys]

        # Missing weights
        if len(incorrect) != 0:
            for k in incorrect:
                logger.error(f"Missing weights: '{k}'")
            raise RuntimeError("Missing weights found. Maybe the JAX weights file is wrong")

        # Useless weights
        for k in missing:
            logger.debug(f"Found '{k}' in the weights file, but it is not required")

        assign(flat, self.data)
