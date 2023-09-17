from transformer import Transformer
from dset import get_loaders, CorpusDataset
import torch
from train import train

#%%
# dset = torch.load('dataset.pth')
batch_size = 16
#%%
max_seq_length = 40
#%%
dset = CorpusDataset(max_seq_length, 'corpus/corpus_2.txt')
#%%
src_vocab_size = len(dset.corpus.dictionary)
tgt_vocab_size = len(dset.corpus.dictionary)
d_model = 128
num_heads = 4
num_layers = 5
d_ff = 128

dropout = 0.1
#%%
train_loader, test_loader = get_loaders(dset, batch_size)
#%%
try:
    for batch, (x, y) in enumerate(train_loader):
        pass
    print('Complete! No issue')
except RuntimeError:
    print(f'Błąd z batch {batch}')
#%%
transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
#%%
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
# optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
#%%
state_dict = torch.load('best_2.pt')
transformer.load_state_dict(state_dict)

# batch_size = 32
eval_batch_size = 16

device = 'cuda'

params = {
'epochs' : 50,
'model' : transformer.to(device),
'clip' : 0.25,
'lr' : 5,
'log_interval' : 50,
'corpus' : dset.corpus,
'train_data' : train_loader,
'test_data': test_loader,
'bptt' : max_seq_length, # sequence length
'criterion' : criterion
}

train(**params)

torch.save(transformer.state_dict(), 'recent.pt')