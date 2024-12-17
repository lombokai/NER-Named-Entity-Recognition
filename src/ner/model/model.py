import torch.nn as nn


class GruNER(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        hidden_size: int=128,
        embedding_size: int=256,
        bidirectional: int=1,
        num_classes: int=10
    ):
        super(GruNER, self).__init__()

        self.emb = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(
            embedding_size,
            hidden_size,
            num_layers = 2,
            bidirectional=True,
            batch_first=True
        )
        self.out_layer = nn.Linear(hidden_size * (bidirectional + 1), num_classes)

    def forward(self, x):
        x = self.emb(x)
        x, _ = self.gru(x)
        logits = self.out_layer(x)

        batch_size, seq_len, num_classes = logits.shape
        logits = logits.view(batch_size * seq_len, num_classes)

        return logits
