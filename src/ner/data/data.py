import torch
import torch.nn as nn

from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class NERDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        
        self.data = []

        self._load_data()
        self._get_mapper()

    def _read_data(self):
        with open(self.data_path) as f:
            raw_data = f.readlines()
        return raw_data

    def _load_data(self):
        raw_data = self._read_data()
        data_dict = None
        idx = 0

        for line in raw_data:
            if line.startswith("-DOCSTART-"):
                continue
            if line.strip() == "":
                if data_dict:
                    self.data.append(data_dict)
                data_dict = None
            else:
                if data_dict is None:
                    data_dict = {
                        "id": idx,
                        "tokens": [],
                        "ner_outputs": []
                    }
                    idx += 1
                l = line.split()
                data_dict["tokens"].append(f"{l[0]}_{l[1]}")
                data_dict["ner_outputs"].append(l[3])

    def build_vocab(self, sentences, specials=None, special_first=True):
        
        token_counter = Counter(token for token in sentences)
        sorted_tokens = sorted(token_counter.keys(), key=lambda x: (-token_counter[x], x))

        if specials:
            if special_first:
                sorted_tokens = specials + sorted_tokens
            else:
                sorted_tokens = sorted_tokens + specials

        vocab = {token: idx for idx, token in enumerate(sorted_tokens)}
        return vocab

    def encode(self, tokens, vocab):
        return [vocab.get(token, vocab["<unk>"]) for token in tokens]
    
    def _get_mapper(self):
        sentences = []
        ner_outputs = []

        for data in self.data:
            sentences.extend(data["tokens"])
            ner_outputs.extend(data["ner_outputs"])

        self.vocab = self.build_vocab(sentences, specials=["<unk>", "<pad>"])
        self.ner_vocab = self.build_vocab(ner_outputs, specials=["<unk>", "<pad>"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        token = torch.tensor(self.encode(self.data[idx]["tokens"], self.vocab), dtype=torch.long)
        target = torch.tensor(self.encode(self.data[idx]["ner_outputs"], self.ner_vocab), dtype=torch.long)

        return token, target
        