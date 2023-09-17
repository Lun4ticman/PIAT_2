from transformer import Transformer
from dset import CorpusDataset
import torch
from generate import lstm_generate, generate_story
from LSTM import LSTM

batch_size = 16
max_seq_length = 40
#%%
dset = CorpusDataset(max_seq_length, 'corpus_2.txt')
#%%
src_vocab_size = len(dset.corpus.dictionary)
# src_vocab_size = 185904
tgt_vocab_size = len(dset.corpus.dictionary)
# tgt_vocab_size = 185904
d_model = 128
num_heads = 4
num_layers = 5
d_ff = 128

dropout = 0.1
#%%
# train_loader, test_loader = get_loaders(dset, batch_size)
transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
lstm = LSTM(src_vocab_size, d_model, d_model, num_layers, dropout)

lstm_state_dict = torch.load('best_lstm.pt')
lstm.load_state_dict(lstm_state_dict)

transformer_state_dict = torch.load('best_2.pt')
transformer.load_state_dict(transformer_state_dict)

lstm_story = lstm_generate(dset.corpus.dictionary, lstm, 'dawno dawno temu', 400)
story = generate_story('dawno dawno temu', transformer.to('cuda'), dset.corpus.dictionary, 400)

with open('stories.txt', 'a', encoding='UTF-8') as f:
    f.writelines('Transformer: \n')
    f.writelines(''.join(story))
    f.writelines('\n')
    f.writelines('LSTM: \n')
    f.writelines(' '.join(lstm_story))



