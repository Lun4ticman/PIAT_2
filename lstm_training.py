from LSTM import LSTM
import torch
from train import train_lstm
from dset import get_loaders, CorpusDataset

batch_size = 16
#%%
max_seq_length = 40
#%%
dset = CorpusDataset(max_seq_length, 'corpus/corpus_2.txt')
#%%
src_vocab_size = len(dset.corpus.dictionary)
tgt_vocab_size = len(dset.corpus.dictionary)
d_model = 128
embedding_size = 128
num_heads = 4
num_layers = 5
d_ff = 128

dropout = 0.1
#%%
train_loader, test_loader = get_loaders(dset, batch_size)

try:
    for batch, (x, y) in enumerate(train_loader):
        pass
    print('Complete! No issue')
except RuntimeError:
    print(f'Błąd z batch {batch}')

model = LSTM(src_vocab_size, d_model, embedding_size, num_layers, dropout)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

state_dict = torch.load('best_lstm.pt')
model.load_state_dict(state_dict)

eval_batch_size = 16

device = 'cuda'

params = {
'epochs' : 50,
'model' : model.to(device),
'clip' : 0.25,
'lr' : 20,
'log_interval' : 50,
'corpus' : dset.corpus,
'train_data' : train_loader,
'test_data': test_loader,
'bptt' : max_seq_length, # sequence length
'criterion' : criterion
}

train_lstm(**params)

torch.save(model.state_dict(), 'lstm_recent.pt')