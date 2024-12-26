import torch
import torch.nn as nn

import numpy as np
from itertools import repeat


class SpatialDropout(nn.Module):
    def __init__(self, drop_prob):
        super(SpatialDropout, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, inputs):
        output = inputs.clone()
        if not self.training or self.drop_prob == 0:
            return inputs
        else:
            noise = self._make_noise(inputs)
            if self.drop_prob == 1:
                noise.fill_(0)
            else:
                noise.bernoulli_(1 - self.drop_prob).div(1 - self.drop_prob)
            noise = noise.expand_as(inputs)
            output.mul_(noise)
        return output
    
    def _make_noise(self, input):
        return input.new().resize_(input.size(0), *repeat(1, input.dim() - 2), input.size(2))
    

class Embed_Layer(nn.Module):
    def __init__(
        self,
        embedding_weight = None,
        vocab_size = None,
        embedding_dim = None,
        training = False,
        dropout_emb = 0.
    ):
        super(Embed_Layer, self).__init__()

        self.training = training
        self.encoder = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout_emb)

        if not self.training:
            for p in self.encoder.parameters():
                p.requires_grad = False

        if embedding_weight is not None:
            self.encoder.weight.data.copy_(torch.from_numpy(embedding_weight))
        else:
            self.encoder.weight.data.copy_(torch.from_numpy(self.random_embedding(vocab_size, embedding_dim)))

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3. / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.dropout(x)
        return x


class BiLSTMModel(nn.Module):
    def __init__(self, n_words, n_pos, n_chunks, n_tags):
        super(BiLSTMModel, self).__init__()

        self.word_embedding = Embed_Layer(vocab_size=n_words+2, embedding_dim=20, training=True)
        self.pos_embedding = Embed_Layer(vocab_size=n_pos+2, embedding_dim=20, training=True)
        self.chunk_embedding = Embed_Layer(vocab_size=n_chunks+2, embedding_dim=20, training=True)

        # self.word_embedding = nn.Embedding(n_words + 2, 20, padding_idx=0)
        # self.pos_embedding = nn.Embedding(n_pos + 2, 20, padding_idx=0)
        # self.chunk_embedding = nn.Embedding(n_chunks + 2, 20, padding_idx=0)

        self.spatial_dropout = SpatialDropout(0.3)
        self.lstm = nn.LSTM(
            input_size=60,  # 20 (word) + 20 (pos) + 20 (chunk)
            hidden_size=200,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0.5,
        )
        self.output_layer = nn.Linear(200 * 2, n_tags + 1)
        nn.init.xavier_uniform(self.output_layer.weight)

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

        return x