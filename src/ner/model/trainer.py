import lightning as L
import torch.nn.functional as F

from omegaconf import DictConfig
from hydra.utils import instantiate


class BiLSTMModule(L.LightningModule):
    def __init__(
        self,
        model: DictConfig,
        loss_fn: DictConfig,
        metrics: DictConfig,
        optim: DictConfig
    ):
        super(BiLSTMModule, self).__init__()
        self.save_hyperparameters()

        self.model = instantiate(model)
        self.loss_fn = instantiate(loss_fn)

        self.train_acc = instantiate(metrics)
        self.val_acc = instantiate(metrics)
        self.test_acc = instantiate(metrics)

        self.optim = optim

    def forward(self, x1, x2, x3):
        out = self.model(x1, x2, x3)
        return out
    
    def training_step(self, batch, batch_idx):
        token, pos, chunk, tags = batch
        logits = self(token, pos, chunk)
        logits = logits.view(-1, logits.shape[-1])
        y = tags.view(-1)

        loss = self.loss_fn(logits, y)

        y_class = logits.argmax(dim=-1)
        acc = self.train_acc(y_class, y)

        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        token, pos, chunk, tags = batch
        logits = self(token, pos, chunk)
        logits = logits.view(-1, logits.shape[-1])
        y = tags.view(-1)

        loss = self.loss_fn(logits, y)

        y_class = logits.argmax(dim=-1)
        # print(y_class)
        # print(y)
        # print()
        acc = self.val_acc(y_class, y)

        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        token, pos, chunk, tags = batch
        logits = self(token, pos, chunk)
        logits = logits.view(-1, logits.shape[-1])
        y = tags.view(-1)

        loss = self.loss_fn(logits, y)

        y_class = logits.argmax(dim=-1)
        acc = self.test_acc(y_class, y)

        self.log_dict({"test_loss": loss, "test_acc": acc}, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optim = instantiate(
            self.optim,
            params=self.parameters(),
            _convert_="partial"
        )
        return optim
