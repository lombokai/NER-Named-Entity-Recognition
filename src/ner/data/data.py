import torch
import lightning as L
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from nltk.corpus.reader.conll import ConllCorpusReader


class ConllDataset(Dataset):
    def __init__(
        self, 
        root_dir: str, 
        file_id: str, 
        max_len: int,
        token_vocab: dict=None,
        pos_vocab: dict=None,
        chunk_vocab: dict=None,
        tags_vocab: dict=None
    ):

        self.root_dir = root_dir
        self.file_id = file_id
        self.max_len = max_len

        self.token_vocab = token_vocab
        self.pos_vocab = pos_vocab
        self.chunk_vocab = chunk_vocab
        self.tags_vocab = tags_vocab

        self._load_data()

    def _load_data(self):
        self.reader = ConllCorpusReader(
            self.root_dir, 
            self.file_id, 
            columntypes=("words", "pos", "chunk", "ne")
        )

        grid = self.reader._grids()
        self.sentences = list(grid)[1:]

        tags = [tag[3] for sentence in self.sentences for tag in sentence]
        tags_counter = Counter(tag for tag in tags)
        self.tags = sorted(tags_counter.keys(), key=lambda x: (-tags_counter[x], x))

        words = [w[0] for sentence in self.sentences for w in sentence]
        words_counter = Counter(word for word in words)
        self.words = sorted(words_counter.keys(), key=lambda x: (-words_counter[x], x))

        pos = [p[1] for sentence in self.sentences for p in sentence]
        pos_counter = Counter(p for p in pos)
        self.pos = sorted(pos_counter.keys(), key=lambda x: (-pos_counter[x], x))

        chunks = [c[2] for sentence in self.sentences for c in sentence]
        chunks_counter = Counter(c for c in chunks)
        self.chunks = sorted(chunks_counter.keys(), key=lambda x: (-chunks_counter[x], x))

    def setup(self, stage="train"):
        if stage=="train":
            self.token_vocab = {w: i+2 for i, w in enumerate(self.words)}
            self.token_vocab["<pad>"] = 0
            self.token_vocab["<unk>"] = 1

            self.pos_vocab = {w: i+2 for i, w in enumerate(self.pos)}
            self.pos_vocab["<pad>"] = 0
            self.pos_vocab["<unk>"] = 1

            self.chunk_vocab = {w: i+2 for i, w in enumerate(self.chunks)}
            self.chunk_vocab["<pad>"] = 0
            self.chunk_vocab["<unk>"] = 1

            self.tags_vocab = {w: i+1 for i, w in enumerate(self.tags)}
            self.tags_vocab["<pad>"] = 0

            # data = {
            #     "token_vocab": self.token_vocab,
            #     "pos_vocab": self.pos_vocab,
            #     "chunk_vocab": self.chunk_vocab,
            #     "tags_vocab": self.tags_vocab
            # }

            # torch.save(data, f"{self.root_dir}/vocab.pth")

    def _pad_sequence(self, sequence, pad_idx):
        sequence = sequence[:self.max_len]
        return sequence + [pad_idx] * (self.max_len - len(sequence))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        
        token = [self.token_vocab.get(w[0], self.token_vocab["<unk>"]) for w in sentence]
        pos = [self.pos_vocab.get(w[1], self.pos_vocab["<unk>"]) for w in sentence]
        chunk = [self.chunk_vocab.get(w[2], self.chunk_vocab["<unk>"]) for w in sentence]
        tags = [self.tags_vocab.get(w[3]) for w in sentence]

        token = self._pad_sequence(token, self.token_vocab["<pad>"])
        pos = self._pad_sequence(pos, self.pos_vocab["<pad>"])
        chunk = self._pad_sequence(chunk, self.chunk_vocab["<pad>"])
        tags = self._pad_sequence(tags, self.tags_vocab["<pad>"])

        return (
            torch.tensor(token, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(chunk, dtype=torch.long),
            torch.tensor(tags, dtype=torch.long)
        )


class ConllDataModule(L.LightningDataModule):
    def __init__(self, root_dir: str, max_len: int, batch_size: int):
        super().__init__()

        self.root_dir = root_dir
        self.max_len = max_len
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train = ConllDataset(
            root_dir = self.root_dir,
            file_id = "train.txt", 
            max_len = self.max_len
        )
        self.train.setup()

        self.val = ConllDataset(
            root_dir = self.root_dir,
            file_id = "valid.txt", 
            max_len = self.max_len,
            token_vocab = self.train.token_vocab,
            pos_vocab = self.train.pos_vocab,
            chunk_vocab = self.train.chunk_vocab,
            tags_vocab = self.train.tags_vocab
        )

        self.test = ConllDataset(
            root_dir = self.root_dir,
            file_id = "test.txt", 
            max_len = self.max_len,
            token_vocab = self.train.token_vocab,
            pos_vocab = self.train.pos_vocab,
            chunk_vocab = self.train.chunk_vocab,
            tags_vocab = self.train.tags_vocab
        )

    def train_dataloader(self):
        return DataLoader(
            self.train, 
            batch_size=self.batch_size, 
            num_workers=11, 
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, 
            batch_size=32, 
            num_workers=11
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, 
            batch_size=32, 
            num_workers=11
        )