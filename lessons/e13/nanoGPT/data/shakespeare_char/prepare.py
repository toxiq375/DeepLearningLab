"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
# Create the input_file_path, os.path.dirname gets the current file's path, and os.path.join concatenates the path and the "input.txt"
# Well I guess it could be written as:
# input_file_path = Path(__file__).resolve().parent / 'input.txt'
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)
# open(path, mode) is a built-in function of python, it will open the path under the given mode, 'w' means write, and it will create one if the given path is not exist
# as f means u can modify the file by modify the 'f'
# requests.get(url) will return the HTTP info from the server, and .text will turn it into the python string, then .write will write it into the opened file/path

with open(input_file_path, 'r') as f: # Open the file as read only mode, the file couldn't be written
    data = f.read()
print(f"length of dataset in characters: {len(data):,}") # ',' will add the ',' by thousands of number, making it easier to read(for western)

# get all the unique characters that occur in this text
chars = sorted(list(set(data))) # Create the vocab
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) } # char -> int
itos = { i:ch for i,ch in enumerate(chars) } # int -> char
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)] # Split the 90% of data into the training set, with no batch now I guess
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
# np.array() is a Numpy function, it will turns the Python List into the Numpy array
# Example: [1, 2, 3] -> array([1, 2, 3], dtype=np.uint16). Additionally, uint16 will reduce the usage of the disk. However, if vocab_size > 65536, u need to find the bigger type to represent the whole vocab
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
# Remember that train_ids is now a Numpy array data
# .tofile() will write the Numpy array into the given file IN BINARY, namely path/train.bin
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
} # Use the meta dictionary to save the mapping of the data
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f: # 'wb' means write in binary
    pickle.dump(meta, f)
    # pickle: a built-in function of Python, it will serialize the given file and save it
    # U can use meta = pickle.load(f) to load a pkl file
    # Actually u can also save the meta data (dictionary) as .json file
    # However, .pkl is better at saving the original format, for example, when using .json, {"0":a} could be restore as {"0":"a"}, the Token ID is turned into a char, not integer

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens
