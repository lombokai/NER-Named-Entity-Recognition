import torch
import torch.nn as nn


class BiLSTMModel(nn.Module):
    def __init__(self, n_words, n_pos, n_chunks, n_tags):
        super(BiLSTMModel, self).__init__()
        self.word_embedding = nn.Embedding(n_words + 2, 20, padding_idx=0)
        self.pos_embedding = nn.Embedding(n_pos + 2, 20, padding_idx=0)
        self.chunk_embedding = nn.Embedding(n_chunks + 2, 20, padding_idx=0)

        self.spatial_dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(
            input_size=60,  # 20 (word) + 20 (pos) + 20 (chunk)
            hidden_size=50,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0.6,
        )
        self.output_layer = nn.Linear(50 * 2, n_tags + 1)  # Bidirectional LSTM doubles hidden size
        self.sigmoid = nn.Sigmoid()

    def forward(self, word_input, pos_input, chunk_input):
        # Embeddings
        word_emb = self.word_embedding(word_input)
        pos_emb = self.pos_embedding(pos_input)
        chunk_emb = self.chunk_embedding(chunk_input)

        # Concatenate embeddings along the last dimension
        x = torch.cat((word_emb, pos_emb, chunk_emb), dim=-1)

        # Apply spatial dropout (dropout on the entire embedding sequence)
        x = self.spatial_dropout(x)

        # LSTM
        x, _ = self.lstm(x)

        # Output layer with time-distributed dense equivalent
        x = self.output_layer(x)

        # Apply sigmoid activation for each time step
        x = self.sigmoid(x)

        return x
