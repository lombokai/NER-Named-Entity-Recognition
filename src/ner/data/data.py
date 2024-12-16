import os
import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader

from ner.transform import Token


class ConllDataset(Dataset):
    def __init__(
        self, 
        data_path,
        token_manager,
        ner_manager, 
        max_len: int=100
    ):
        self.data_path = data_path
        self.max_len = max_len
        self.token_manager = token_manager
        self.ner_manager = ner_manager

        self.data = self._load_data()

    def _read_data(self):
        with open(self.data_path) as f:
            raw_data = f.readlines()
        return raw_data

    def _load_data(self):
        raw_data = self._read_data()
        data = []
        data_dict = None
        idx = 0

        for line in raw_data:
            if line.startswith("-DOCSTART-"):
                continue
            if line.strip() == "":
                if data_dict:
                    data.append(data_dict)
                data_dict = None
            else:
                if data_dict is None:
                    data_dict = {"id": idx, "tokens": [], "ner_outputs": []}
                    idx += 1
                l = line.split()
                data_dict["tokens"].append(f"{l[0]}_{l[1]}")
                data_dict["ner_outputs"].append(l[3])
        return data

    def _pad_sequence(self, sequence, pad_idx):
        sequence = sequence[:self.max_len]
        return sequence + [pad_idx] * (self.max_len - len(sequence))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        token = self.token_manager.encode(sample["tokens"], is_token=True)
        ner_tags = self.ner_manager.encode(sample["ner_outputs"], is_token=False)

        token = self._pad_sequence(token, self.token_manager.token_vocab["<pad>"])
        ner_tags = self._pad_sequence(ner_tags, self.ner_manager.output_vocab["<pad>"])        
        
        return torch.tensor(token, dtype=torch.long), torch.tensor(ner_tags, dtype=torch.long)
        

class ConllDataModule(L.LightningDataModule):
    def __init__(
        self, 
        data_path, 
        max_len: int=100
    ):
        super().__init__()

        self.data_path = data_path
        self.max_len = max_len

    def setup(self, stage=None):
        train_path = os.path.join(self.data_path, "train.txt")
        val_path = os.path.join(self.data_path, "valid.txt")
        test_path = os.path.join(self.data_path, "test.txt")

        token_manager = Token()
        ner_manager = Token()

        self.train_dataset = ConllDataset(
            train_path, 
            token_manager, 
            ner_manager, 
            self.max_len
        )

        token_manager.build_vocab(self.train_dataset.data)
        ner_manager.build_output_vocab(self.train_dataset.data)

        # self.vocab_size = len(token_manager.vocab)

        self.val_dataset = ConllDataset(
            val_path, 
            token_manager, 
            ner_manager, 
            self.max_len
        )
        self.test_dataset = ConllDataset(
            test_path, 
            token_manager, 
            ner_manager, 
            self.max_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=32, 
            num_workers=11, 
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=32, 
            num_workers=11
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=32, 
            num_workers=11
        )
