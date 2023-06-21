import hydra
from omegaconf import DictConfig
from lightning import LightningModule

from nextfold import utils

log = utils.get_logger(__name__)


@utils.task_wrapper
def inference(cfg: DictConfig) -> None:
    """
    Evaluate given checkpoint on a `LightningDataModule` testset.

    Args:
        cfg (DictConfig)
            Configuration composed by Hydra.
    
    Returns:
        Tuple[dict, dict]: Dict with metrics and cit with all instantiated objects.
    """

    assert cfg.ckpt_path

    logger.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    logger.info("Starting inference!")


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="inference.yaml",
)
def main(cfg: DictConfig) -> None:
    utils.extras(cfg)
    inference(cfg)


if __name__ == "__main__":
    main()
