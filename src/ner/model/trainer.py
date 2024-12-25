import lightning as L
import torch.nn.functional as F

from omegaconf import DictConfig
from hydra.utils import instantiate


class GruModule(L.LightningModule):
    def __init__(
        self,
        model: DictConfig,
        loss_fn: DictConfig,
        metrics: DictConfig,
        optim: DictConfig
    ):
        super(GruModule, self).__init__()
        self.save_hyperparameters()

        self.model = instantiate(model)
        self.loss_fn = instantiate(loss_fn)
        self.metrics = instantiate(metrics)

        self.optim = optim

    def forward(self, x):
        out = self.model(x)
        return out
    
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        y = y.view(-1)

        loss = self.loss_fn(logits, y)

        y_class = logits.argmax(dim=-1)
        acc = self.metrics(y_class, y)

        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        y = y.view(-1)

        loss = self.loss_fn(logits, y)

        y_class = logits.argmax(dim=-1)
        acc = self.metrics(y_class, y)

        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        y = y.view(-1)

        loss = self.loss_fn(logits, y)

        y_class = logits.argmax(dim=-1)
        acc = self.metrics(y_class, y)

        self.log_dict({"test_loss": loss, "test_acc": acc}, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optim = instantiate(
            self.optim,
            params=self.parameters(),
            _convert_="partial"
        )
        return optim
