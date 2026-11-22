import torch
import torch.nn as nn
from config import CONTEXT_SIZE

def prepare_data(): # prepare the input data
    text = "hello world\n"*8 # source sentence
    train_text = text[:72] # train_split
    val_text = text[72:] # eval_split

    chars = sorted(set(train_text)) # use set() to extract the chars in the text,then use sorted to sort them
    vocab_size = len(chars) # vocab_size is the numbers of the chars in the text
    stoi = {ch:i for i, ch in enumerate(chars)} # stoi: trun char into code number
    itos = {i:ch for ch, i in stoi.items()} # .items() returns the pairs in the dictionary

    train_ids = torch.tensor(encode(stoi, train_text), dtype=torch.long)
    # train_ids is code_number form of train_text data
    # encode is the function that turns chars into code_number within the stoi
    # torch.tensor() turns the list into tensor so that they can be trained
    val_ids = torch.tensor(encode(stoi, val_text), dtype=torch.long)

    X_train, Y_train = make_windows(train_ids, CONTEXT_SIZE)
    # function make_windows unites the discrete training codes into training pieces of the context_size
    X_val, Y_val = make_windows(val_ids, CONTEXT_SIZE)

    return X_train, Y_train, X_val, Y_val, vocab_size, stoi, itos

def encode(stoi, text):
    return [stoi[ch] for ch in text] 
    # encode retrive the char in the text and turns them into the code_numbers according to the stoi

def decode(itos, code_numbers):
    return "".join(itos[int(i)] for i in code_numbers)
    # decode takes the code_numbers in the list and turns them into the chars according to the itos
    # then use the "".join() to make a list of chars in the a string(text)

def make_windows(code_numbers, context_size):
    X = []
    Y = []
    for i in range(len(code_numbers)-(context_size+1-1)):
        # if context_size = l, prediction size = m, then the range should be range(len() - (context_size + m -1))
        # here m = 1
        X.append(code_numbers[i:i+context_size])
        Y.append(code_numbers[i+1:i+context_size+1])
        # model takes a sequence of the training_data and predict the next one
        # each code of the training_sequence predicts the next code, so that the model can be trained with multiple codes at once
        # the sequence has the length of context_size
        # X is the input seq and Y is the predicted seq, each matches it next position
    X = torch.stack(X, dim=0)
    Y = torch.stack(Y, dim=0) # stack the list into tensor, it equals using torch.cat(t.unsqueeze(0) for t in list), if u use torch.cat(), u must use .unsqueeze(0) to make a new dim in the front

    return X, Y