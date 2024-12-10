import torch.nn as nn


class GruNER(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        hidden_size: int=128,
        embedding_size: int=256,
        bidirectional: int=0,
        num_classes: int=17
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
        x = self.out_layer(x)
        return x
