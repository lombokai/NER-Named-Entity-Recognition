from collections import Counter


class Token:
    def __init__(self, specials=None, special_first=True):
        self.specials = specials or ["<unk>", "<pad>"]
        self.special_first = special_first
        self.vocab = {}
        self.reverse_vocab = {}

    def build_vocab(self, sentences, is_token=True):
        if is_token:
            token_counter = Counter(token for sentence in sentences for token in sentence["tokens"])
        else:
            token_counter = Counter(token for sentence in sentences for token in sentence["ner_outputs"])

        sorted_tokens = sorted(token_counter.keys(), key=lambda x: (-token_counter[x], x))

        if self.specials:
            if self.special_first:
                sorted_tokens = self.specials + sorted_tokens
            else:
                sorted_tokens = sorted_tokens + self.specials

        self.vocab = {token: idx for idx, token in enumerate(sorted_tokens)}
        self.reverse_vocab = {idx: token for idx, token in enumerate(sorted_tokens)}

    def encode(self, tokens):
        return [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]
    
    def decode(self, tokens):
        return [self.reverse_vocab.get(token, self.reverse_vocab["<unk>"]) for token in tokens]
    