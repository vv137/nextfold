from typing import List, Tuple

import hydra
from omegaconf import DictConfig
import torch
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger

from nextfold import utils

log = utils.get_logger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """
    Evaluate given checkpoint on a `LightningDataModule` testset.

    Args:
        cfg (DictConfig)
            Configuration composed by Hydra.
    
    Returns:
        Tuple[dict, dict]: Dict with metrics and cit with all instantiated objects.
    """

    # Set seed for random number generators
    if cfg.get("seed"):
        L.seed_everything(seed=cfg.seed, workers=True)

    logger.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    logger.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    logger.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    logger.info("Instantiating loggers...")
    loggers: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    logger.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        logger.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("compile"):
        logger.info("Compiling model!")
        model = torch.compile(model)

    if cfg.get("train"):
        logger.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        logger.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        logger.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="predict.yaml",
)
def main(cfg: DictConfig) -> None:
    # Apply extra utilities
    utils.extras(cfg)

    # Train the model
    metric_dict, _ = train(cfg)

    # Safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

    # Return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
