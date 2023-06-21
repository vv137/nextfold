import hydra
from omegaconf import DictConfig


@hydra.main(
    version_base=None,
    config_path='configs',
    config_name='feature.yaml',
)
def main(cfg: DictConfig) -> None:
    pass


if __name__ == '__main__':
    main()
