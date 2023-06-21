from pathlib import Path
from typing import Sequence

import rich
import rich.syntax
import rich.tree
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf

from nextfold.utils.logger import get_logger

logger = get_logger(__name__)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "data",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """
    Print content of `DictConfig`..

    Args:
        cfg (DictConfig)
            The Hydra configuration.
        print_order (Sequence[str], optional)
            Determines in what order config compoents are printed.
        resolve (bool, optional)
            Whether to resolve reference fields of `cfg`.
        save_to_file (bool, optional)
            Whether to export config to the Hydra output folder.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # Add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else logger.warning(
            f"Field '{field}' not found in config. Skipping '{field} config printing..."
        )

    # Add all the other fields to queue
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # Generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # Print config tree
    rich.print(tree)

    # Save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as fp:
            rich.print(tree, file=fp)
