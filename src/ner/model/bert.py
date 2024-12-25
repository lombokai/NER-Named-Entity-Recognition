import torch
import torch.nn as nn
import lightning as L
from transformers import BertModel
from hydra.utils import instantiate
from omegaconf import DictConfig


class BERTNer(nn.Module):
    def __init__(self, num_classes, pretrained_model_name="bert-base-uncased"):
        super(BERTNer, self).__init__()

        # Load pretrained BERT model
        self.bert = BertModel.from_pretrained(pretrained_model_name)

        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Extract the hidden size from the BERT model
        hidden_size = self.bert.config.hidden_size

        # Define the classification head
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )

        cls_output = outputs.last_hidden_state

        logits = self.classifier(cls_output)
        batch_size, seq_len, num_classes = logits.shape
        logits = logits.view(batch_size * seq_len, num_classes)

        return logits
    

class BERTNerModule(L.LightningModule):
    def __init__(
        self,
        model: DictConfig,
        loss_fn: DictConfig,
        metrics: DictConfig,
        optim: DictConfig
    ):
        super(BERTNerModule, self).__init__()
        self.save_hyperparameters()

        self.model = instantiate(model)
        self.loss_fn = instantiate(loss_fn)
        self.metrics = instantiate(metrics)

        self.optim = optim

    def forward(self, x, attention):
        out = self.model(x, attention_mask=attention)
        return out
    
    def training_step(self, batch, batch_idx):
        x, y, a = batch
        logits = self(x, a)
        y = y.view(-1)

        loss = self.loss_fn(logits, y)

        y_class = logits.argmax(dim=-1)
        acc = self.metrics(y_class, y)

        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, a = batch
        logits = self(x, a)
        y = y.view(-1)

        loss = self.loss_fn(logits, y)

        y_class = logits.argmax(dim=-1)
        acc = self.metrics(y_class, y)

        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y, a = batch
        logits = self(x, a)
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