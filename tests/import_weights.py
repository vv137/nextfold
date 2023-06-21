import os
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from nextfold.utils.import_weights import AlphaFoldWeightImporter, _model_name_to_model_npz


@hydra.main(
    version_base=None,
    config_path='../configs',
    config_name='predict.yaml',
)
def main(cfg: DictConfig) -> None:
    # Load parameters
    params_path = Path('/apps/vv137/alphafold.data.1/params')
    model_name = cfg.model.globals.name
    params_path /= _model_name_to_model_npz(model_name)

    print('>>>', model_name)
    importer = AlphaFoldWeightImporter(
        npz_path=params_path,
        cfg=cfg,
    )


if __name__ == '__main__':
    main()
