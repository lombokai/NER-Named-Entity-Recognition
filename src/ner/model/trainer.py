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

        self.acc = instantiate(metrics.accuracy)
        self.f1score = instantiate(metrics.f1)
        self.precision = instantiate(metrics.precision)
        self.recall = instantiate(metrics.recall)

        self.optim = optim

    def forward(self, x1, x2, x3):
        out = self.model(x1, x2, x3)
        return out
    
    def training_step(self, batch, batch_idx):
        token, pos, chunk, tags = batch
        logits = self(token, pos, chunk)
        logits = logits.view(-1, logits.shape[-1])

        y_class = logits.argmax(dim=-1)
        y = tags.view(-1)

        loss = self.loss_fn(logits, y)

        acc = self.acc(y_class, y)
        f1 = self.f1score(y_class, y)
        prec = self.precision(y_class, y)
        rec = self.recall(y_class, y)

        self.log_dict({
            "train_loss": loss, 
            "train_acc": acc,
            "train_f1": f1,
            "train_prec": prec,
            "train_rec": rec
        }, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        token, pos, chunk, tags = batch
        logits = self(token, pos, chunk)
        logits = logits.view(-1, logits.shape[-1])

        y = tags.view(-1)
        y_class = logits.argmax(dim=-1)

        loss = self.loss_fn(logits, y)

        self.log_dict({"val_loss": loss}, prog_bar=True)

        self.acc.update(y_class, y)
        self.f1score.update(y_class, y)
        self.precision.update(y_class, y)
        self.recall.update(y_class, y)

        return loss
    
    def on_validation_epoch_end(self):
        acc = self.acc.compute()
        f1 = self.f1score.compute()
        prec = self.precision.compute()
        rec = self.recall.compute()

        self.log_dict({
            "val_acc": acc,
            "val_f1": f1,
            "val_prec": prec,
            "val_rec": rec
        }, prog_bar=True)
        
        self.acc.reset()
        self.f1score.reset()
        self.precision.reset()
        self.recall.reset()
    
    def test_step(self, batch, batch_idx):
        token, pos, chunk, tags = batch
        logits = self(token, pos, chunk)
        logits = logits.view(-1, logits.shape[-1])
        y = tags.view(-1)

        loss = self.loss_fn(logits, y)

        y_class = logits.argmax(dim=-1)

        acc = self.acc(y_class, y)
        f1 = self.f1score(y_class, y)
        prec = self.precision(y_class, y)
        rec = self.recall(y_class, y)

        self.log_dict({
            "test_loss": loss,
            "test_acc": acc,
            "test_f1": f1,
            "test_prec": prec,
            "test_rec": rec
        }, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optim = instantiate(
            self.optim,
            params=self.parameters(),
            _convert_="partial"
        )
        return optim
