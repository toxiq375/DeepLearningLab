import torch
import torch.nn as nn
from config import CONTEXT_SIZE

def prepare_data():
    text = "hello world\n"*8
    train_text = text[:72]
    val_text = text[72:]

    chars = sorted(set(train_text))
    vocab_size = len(chars)
    stoi = {ch:i for i, ch in enumerate(chars)}
    itos = {i:ch for ch, i in stoi.items()}

    train_ids = torch.tensor(encode(stoi, train_text), dtype=torch.long)
    val_ids = torch.tensor(encode(stoi, val_text), dtype=torch.long)

    X_train, Y_train = make_windows(train_ids, CONTEXT_SIZE)
    X_val, Y_val = make_windows(val_ids, CONTEXT_SIZE)

    return X_train, Y_train, X_val, Y_val, vocab_size

def encode(stoi, s):
    return [stoi[ch] for ch in s]
def decode(itos, ids):
    return "".join(itos[int(i)] for i in ids)

def make_windows(ids, context_size):
    X = []
    Y = []
    for i in range(len(ids)-context_size):
        X.append(ids[i:i+context_size])
        Y.append(ids[i+1:i+context_size+1])
    X = torch.stack(X, dim=0)
    Y = torch.stack(Y, dim=0)
    return X, Y

