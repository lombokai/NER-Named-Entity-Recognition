import sys

import torch
import hydra
import lightning as L
from hydra.utils import instantiate

sys.path.append("src")
torch.set_float32_matmul_precision('medium')


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def cli_main(cfg):

    data = instantiate(cfg.data)
    module = instantiate(cfg.module, _recursive_ = False)

    checkpoint = instantiate(cfg.callbacks.checkpoint)
    early_stop = instantiate(cfg.callbacks.early_stop)

    mlflow_logger = instantiate(cfg.logger.mlflow_logger)
    tensorboard_logger = instantiate(cfg.logger.tensorboard_logger)

    trainer: L.Trainer = instantiate(
        cfg.trainer,
        callbacks=[checkpoint, early_stop],
        logger=[mlflow_logger, tensorboard_logger]
    )

    trainer.fit(module, data)
    

if __name__ == "__main__":
    cli_main()
