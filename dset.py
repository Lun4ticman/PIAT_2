import torch.utils.data
from tqdm import tqdm

from prepare import Corpus


def get_data(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def prepare_data(data):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0)
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch)
    # Evenly divide the data across the bsz batches.
    data = data.view(1, -1).t().contiguous()
    return data


class CorpusDataset(torch.utils.data.Dataset):
    def __init__(self, bptt, path):
        self.x = []
        self.y = []

        self.corpus = Corpus(path)

        # data = prepare_data(self.corpus.train)
        data = self.corpus.train
        for idx, i in enumerate(tqdm(range(0, data.size(0) - 1, bptt), desc='Preparing data..')):
            data_, targets = get_data(data, i, bptt)
            if data_.size()[0] == bptt:
                self.x.append(data_)
                self.y.append(targets)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])


def get_loaders(dset, batch_size, train_size=0.9):

    # test_size = round(1 - train_size, 2)

    train_set_size = int(len(dset) * train_size)
    test_set_size = len(dset) - train_set_size

    train_dataset, test_dataset = torch.utils.data.random_split(dset, [train_set_size, test_set_size])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader
