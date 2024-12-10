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
        optimizer: DictConfig,
    ):
        super().__init__()

        self.model = instantiate(model)
        self.loss_fn = instantiate(loss_fn)
        self.metrics = instantiate(metrics)

        self.optimizer = optimizer

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        _, _, num_classes = logits.shape
        logits = logits.view(-1, num_classes)
        y = y.view(-1)

        loss = self.loss_fn(logits, y)

        y_class = F.softmax(logits, dim=-1).max(dim=1)[1]
        acc = self.metrics(y_class, y)

        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        _, _, num_classes = logits.shape
        logits = logits.view(-1, num_classes)
        y = y.view(-1)

        loss = self.loss_fn(logits, y)
        y_class = F.softmax(logits, dim=-1).max(dim=1)[1]
        acc = self.metrics(y_class, y)

        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        _, _, num_classes = logits.shape
        logits = logits.view(-1, num_classes)
        y = y.view(-1)

        loss = self.loss_fn(logits, y)
        y_class = F.softmax(logits, dim=-1).max(dim=1)[1]
        acc = self.metrics(y_class, y)

        self.log_dict({"test_loss": loss, "test_acc": acc}, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = instantiate(
            self.optimizer,
            params=self.parameters(),
            _convert_="partial"
        )
        return optimizer
