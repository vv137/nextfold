import functools
from importlib.util import find_spec
from typing import Any, Callable
import warnings

from omegaconf import DictConfig

from nextfold.utils import get_logger, print_config_tree

logger = get_logger(__name__)


def task_wrapper(task_func: Callable) -> Callable:
    """
    Optional decorator that controls the failure behavior when executing the task function.

    This decorator can be used as following.
    - Make sure loggers are closed even if the task function raises an exception
    - Save the exception to a log file
    - Mark the run as failed with a file in the `logs/` directory (so we can find and rerun it later)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[dict, dict]:
        ...
        return metric_dict, object_dict
    """
    @functools.wraps(task_func)
    def wrap(cfg: DictConfig) -> Any:
        # Run the task
        try:
            r = task_func(cfg=cfg)
        except Exception as e:
            # Save exception to log file
            logger.exception("")
            raise e

        # Either success or exception
        finally:
            # Display output dir path in terminal
            logger.info(f"Output dir: '{cfg.paths.output_dir}'")

            # Always close wandb run
            if find_spec("wandb"):
                import wandb    # type: ignore

                if wandb.run:
                    logger.info("Closing wandb")
                    wandb.finish()

        return r

    return wrap


def extras(cfg: DictConfig) -> None:
    """
    Applies optional utilities before the task is started.
    """

    if not cfg.get("extras"):
        logger.warning("Extras config not found! <cfg.extras=null>")
        return

    # Disable Python warnings
    if cfg.extras.get("ignore_warnings"):
        logger.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # Print config tree
    if cfg.extras.get("print_config"):
        logger.info(f"Print config tree <cfg.extras.print_config=True>")
        print_config_tree(cfg, resolve=True, save_to_file=True)


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        logger.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    logger.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value
