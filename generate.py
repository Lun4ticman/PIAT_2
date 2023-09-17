import dset
from transformer import Transformer
from prepare import Dictionary
from LSTM import LSTM
import torch

import numpy as np


def generate(words: [str], model: Transformer, dictionary: Dictionary) -> str:

    ids = []
    for word in words:
        ids.append(dictionary.word2idx[word])
    idss = torch.tensor(ids)

    sequence = model.max_seq_length

    pad = sequence - len(idss)

    input_ = torch.tensor(np.pad(idss, (0, pad + 1), 'constant', constant_values=0))
    input_ = input_.to(model.device)

    with torch.no_grad():
        output = model(input_[np.newaxis, :sequence], input_[np.newaxis, :sequence])  # batch 1

    preds = torch.argmax(output[0], dim=-1)

    output_token = preds[-1].detach().cpu().numpy()
    output_word = dictionary.idx2word[output_token]

    return output_word


def generate_story(prompt: str, model, dictionary: Dictionary, story_length: int, device='cuda') -> str:

    counter = 0
    words = prompt.split()
    for i in range(0, story_length):
        x = torch.tensor([[dictionary.word2idx[w] for w in words[i:]]]).to(device)
        y_pred = model(x, x).to(device)

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().cpu().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dictionary.idx2word[word_index])

    return ' '.join(words)


def lstm_generate(dictionary: Dictionary, model: LSTM, text: str, next_words=400):
    model.eval()

    words = text.split(' ')
    state_h, state_c = model.init_state(len(words))

    for i in range(0, next_words):
        x = torch.tensor([[dictionary.word2idx[w] for w in words[i:]]])
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dictionary.idx2word[word_index])

    return ' '.join(words)