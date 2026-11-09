import torch
import torch.nn as nn

text = "hello world\n"*8

chars = sorted(set(text)) # set(text)：去除重复的字符，只保留1次 // sorted()按字符的 Unicode 码点顺序排列
vocab_size = len(chars)

stoi = {}
for i,char in enumerate(chars):
    stoi[char] = i

itos = chars

print(stoi, '\n', itos)

def encode(s: str):
    encode_list = []
    for char in s:
        encode_list.append(stoi[char])

    return encode_list

def decode(ids: list):
    character = [] # 先累积一个字符列表
    for i in ids:
        character.append(itos[i])
    decode_str = "".join(character) # 随后再转换为字符串

    return decode_str

encoded = encode(text)
decoded = decode(encoded)
assert decoded == text
