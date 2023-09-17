import os
from io import open
import torch
from tqdm import tqdm
from torchtext.data.utils import get_tokenizer


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.tokenizer = get_tokenizer('spacy', language='pl_core_news_sm')
        self.dictionary = Dictionary()
        self.train = self.tokenize(path)

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            lines = f.readlines()
            for line in tqdm(lines, desc='Adding words to dict'):
                words = self.tokenizer(line)
                ids = []
                for word in words:
                    self.dictionary.add_word(word)
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids