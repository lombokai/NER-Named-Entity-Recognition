import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ner.model import BERT


class NERInfer:
    def __init__(self, model_path: str, token_file: str, device: str="cpu"):
        super().__init__()

        self.model_path = model_path
        self.token_file = token_file
        self.device = device

        self.vocab = self._load_vocab()
        self.token_vocab = self.vocab["token_vocab"]
        self.pos_vocab = self.vocab["pos_vocab"]
        self.chunk_vocab = self.vocab["chunk_vocab"]
        self.tags_vocab = self.vocab["tags_vocab"]

        self.class_dict = {v:k for k,v in self.tags_vocab.items()}

        self.model = self._load_model()

    def _load_vocab(self):
        vocab = torch.load(self.token_file)
        return vocab

    def _load_model(self):
        model = BERT(vocab_size=len(self.token_vocab))
        ckpt = torch.load(
            self.model_path,
            map_location = self.device,
            weights_only = False
        )
        state_dict = {
            k.replace("model.", ""):v for k, v in ckpt["state_dict"].items()
        }
        model.load_state_dict(state_dict)
        return model

    def _tokenize(self, sentence):
        tokens = sentence.split()
        return [self.token_vocab.get(token, self.token_vocab.get("<unk>")) for token in tokens]

    def preprocess(self, text):
        if isinstance(text, str):
            tokenized = self._tokenize(text)
        elif isinstance(text, list):
            tokenized = [self._tokenize(sentence) for sentence in text]
        else:
            raise ValueError("Input text must be a string or a list of sentences.")

        print(tokenized)

        # Convert to tensor
        if isinstance(tokenized[0], list):  # Batch input
            tokenized = [torch.tensor(token, dtype=torch.long, device=self.device) for token in tokenized]
            padded = pad_sequence(tokenized, batch_first=True, padding_value=self.token_vocab["<pad>"])
            return padded
        else:  # Single input
            return torch.tensor([tokenized], dtype=torch.long, device=self.device)

    def forward(self, inp, segment=None):
        self.model.eval()
        with torch.no_grad():
            out = self.model(inp, segment)
        return out

    def post_process(self, logits, tokens):
        logits_prob = F.softmax(logits, dim=1)
        class_idx = torch.argmax(logits, dim=-1)

        results = []
        for prob, tok, index in zip(logits_prob, tokens, class_idx):
            result = []
            for idx, (token, prediction) in enumerate(zip(tok, index.tolist())):
                entity = self.class_dict.get(prediction, "O")
                score = prob[idx].max()
                result.append({
                    "entity": entity,
                    "score": round(float(score), 4),
                    "index": idx,
                    "word": token
                })
            results.append(result)
        if len(results) == 1:
            return results[0]
        return results

    def predict(self, text):
        input_tensor = self.preprocess(text)
        # tokens = text.split() if isinstance(text, str) else [word for sentence in text for word in sentence.split()]
        tokens = [text.split()] if isinstance(text, str) else [word.split() for word in text]

        logits = self.forward(input_tensor, torch.zeros_like(input_tensor))
        logits = logits.view(*input_tensor.shape, 10)

        results = self.post_process(logits, tokens)
        return results
