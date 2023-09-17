import torch
import prepare
import time
import math
from tqdm import tqdm

# device = torch.device("cuda")

# corpus = prepare.Corpus('')

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def train(epochs=None, model=None, clip=None, lr=None, log_interval=None, corpus=None, train_data=None, test_data = None,
          bptt=None, criterion=None,
          ):
    # Turn on training mode which enables dropout.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    device = 'cuda'
    ntokens = len(corpus.dictionary)

    best_loss = 99999
    try:
        pbar = tqdm(range(1, epochs + 1), desc='Training..', position=0)
        for epoch in pbar:
            epoch_start_time = time.time()
            model.train()
            total_loss = 0.
            start_time = time.time()


            # for batch, i in enumerate(tqdm(range(0, train_data.size(0) - 1, bptt), desc='Epoch..')):
            #     data, targets = get_batch(train_data, i, bptt)
            try:
                for batch, (x, y) in enumerate(train_data):
                    data = x.to(device)
                    # data = torch.transpose(data, 0, 1) # transpose bo seq, batch, dim

                    targets = y.to(device)
                    # targets = torch.transpose(targets, 0, 1)
                    # model.zero_grad()

                    optimizer.zero_grad()

                    output = model(data, targets)
                    # output = output.contiguosview(-1, ntokens)
                    output = output.contiguous().view(-1, ntokens)

                    targets = targets.contiguous().view(-1)

                    # loss = criterion(tgt_data[:, 1:].contiguous().view(-1))

                    loss = criterion(output, targets)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                    if batch % log_interval == 0 and batch > 0:
                        cur_loss = total_loss / log_interval
                        elapsed = time.time() - start_time
                        # print('-'*39)
                        # print('\n \r', end="")
                        # print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                        #       'loss {:5.2f}'.format(
                        #     epoch, batch, len(train_data), lr,
                        #                   elapsed * 1000 / log_interval, cur_loss), end="")
                        pbar.set_description('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                              'loss {:5.2f}'.format(
                            epoch, batch, len(train_data), lr,
                                          elapsed * 1000 / log_interval, cur_loss))
                        total_loss = 0
                        start_time = time.time()

                loss = 0

                with torch.no_grad():
                    # for batch, i in enumerate(tqdm(range(0, test_data.size(0) - 1, bptt), desc='Test epoch..')):
                    #     data, targets = get_batch(test_data, i, bptt)
                    for batch, (data, targets) in enumerate(test_data):
                        data = data.to(device)
                        targets = targets.to(device)

                        output = model(data, targets)
                        output = output.contiguous().view(-1, ntokens)

                        targets = targets.contiguous().view(-1)
                        loss += criterion(output, targets)
                    if loss < best_loss:
                        best_loss = loss
                        best_model = model.state_dict()
                    print("Epoch %d: Cross-entropy: %.4f" % (epoch, loss))

                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    for p in model.parameters():
                        p.data.add_(p.grad, alpha=-lr)

                if epoch % 2 == 0:
                    torch.save(model.state_dict(), 'checkpoint_2.pt')

            except RuntimeError:
                print(f'Błąd z {batch}')
                print('Pomijane...')
                continue

        torch.save(best_model, "best_2.pt")

    except KeyboardInterrupt:
        print('\n')
        print('Finishing early by user')
        torch.save(model.state_dict(), 'checkpoint_2.pt')


def train_lstm(epochs=None, model=None, clip=None, lr=None, log_interval=None, corpus=None, train_data=None, test_data = None,
          bptt=None, criterion=None,
          ):
    # Turn on training mode which enables dropout.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    device = 'cuda'
    ntokens = len(corpus.dictionary)

    best_loss = 99999
    model.train()
    try:
        pbar = tqdm(range(1, epochs + 1), desc='Training..', position=0)
        for epoch in pbar:
            state_h, state_c = model.init_state(bptt)
            state_h, state_c = state_h.to(device), state_c.to(device)
            epoch_start_time = time.time()

            total_loss = 0.
            start_time = time.time()


            # for batch, i in enumerate(tqdm(range(0, train_data.size(0) - 1, bptt), desc='Epoch..')):
            #     data, targets = get_batch(train_data, i, bptt)
            try:
                for batch, (x, y) in enumerate(train_data):
                    data = x.to(device)
                    # data = torch.transpose(data, 0, 1) # transpose bo seq, batch, dim

                    targets = y.to(device)
                    # targets = torch.transpose(targets, 0, 1)
                    # model.zero_grad()

                    optimizer.zero_grad()

                    output, (state_h, state_c) = model(data, (state_h, state_c))
                    state_h = state_h.detach()
                    state_c = state_c.detach()
                    # output = output.contiguosview(-1, ntokens)
                    output = output.contiguous().view(-1, ntokens)

                    targets = targets.contiguous().view(-1)

                    # loss = criterion(tgt_data[:, 1:].contiguous().view(-1))

                    loss = criterion(output, targets)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                    if batch % log_interval == 0 and batch > 0:
                        cur_loss = total_loss / log_interval
                        elapsed = time.time() - start_time
                        # print('-'*39)
                        # print('\n \r', end="")
                        # print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                        #       'loss {:5.2f}'.format(
                        #     epoch, batch, len(train_data), lr,
                        #                   elapsed * 1000 / log_interval, cur_loss), end="")
                        pbar.set_description('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                              'loss {:5.2f}'.format(
                            epoch, batch, len(train_data), lr,
                                          elapsed * 1000 / log_interval, cur_loss))
                        total_loss = 0
                        start_time = time.time()

                loss = 0

                with torch.no_grad():
                    # for batch, i in enumerate(tqdm(range(0, test_data.size(0) - 1, bptt), desc='Test epoch..')):
                    #     data, targets = get_batch(test_data, i, bptt)
                    for batch, (data, targets) in enumerate(test_data):
                        data = data.to(device)
                        targets = targets.to(device)

                        output, (_, _) = model(data, (state_h, state_c))
                        output = output.contiguous().view(-1, ntokens)

                        targets = targets.contiguous().view(-1)
                        loss += criterion(output, targets)
                    if loss < best_loss:
                        best_loss = loss
                        best_model = model.state_dict()
                    print("Epoch %d: Cross-entropy: %.4f" % (epoch, loss))

                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    for p in model.parameters():
                        p.data.add_(p.grad, alpha=-lr)

                if epoch % 2 == 0:
                    torch.save(model.state_dict(), 'checkpoint_lstm.pt')

            except RuntimeError:
                print(f'Błąd z {batch}')
                print('Pomijane...')
                continue

        torch.save(best_model, "best_lstm.pt")

    except KeyboardInterrupt:
        print('\n')
        print('Finishing early by user')
        torch.save(model.state_dict(), 'checkpoint_lstm.pt')



