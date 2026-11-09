import torch
import torch.nn as nn

# 创建源数据
text = "hello world\n"*8

# 构建字符词表
chars = sorted(set(text)) # set(text)：去除重复的字符，只保留1次 // sorted()按字符的 Unicode 码点顺序排列
vocab_size = len(chars)

stoi = {} # 存储字符-ID对
for i,char in enumerate(chars):
    stoi[char] = i

itos = chars # 存储ID-字符对

print(stoi, '\n', itos)

# 编码
def encode(s: str): # 编码：将字符编码为ID
    encode_list = []
    for char in s:
        encode_list.append(stoi[char])

    return encode_list

# 解码
def decode(ids: list): # 解码：将ID解码为字符
    character = [] # 先累积一个字符列表
    for i in ids:
        character.append(itos[i])
    decode_str = "".join(character) # 随后再将列表转换为字符串

    return decode_str

encoded = encode(text) # 编码后的文本->ID
decoded = decode(encoded)
assert decoded == text

# 将编码数据转换为tensor
data = torch.tensor(encoded, dtype = torch.long) # 64位整型
x = (data[:-1]).unsqueeze(0) # 构建输入x，并增加batch维度
y = (data[1:]).unsqueeze(0) # 构建监督目标y，任务是预测

# 正式创建模型并进行训练
torch.manual_seed(42)
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)
    def forward(self, code):
        logits = self.table(code)

        return logits

model = BigramLanguageModel(vocab_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = 0.05,
    weight_decay = 0
)

def main():
    for epoch in range(50000):
        optimizer.zero_grad()
        logits = model(x).transpose(1,2)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    main()
    
    codes = torch.tensor([encode("hell")], dtype = torch.long)
    with torch.no_grad():
        for i in range(20):
            logits = model(codes)
            last_logits = logits[:, -1, :]
            probs = torch.softmax(last_logits, dim = -1) 
            assert probs.shape == (1, 9)
            next_id = torch.multinomial(probs, 1)
            codes = torch.cat([codes, next_id], dim = 1)
    str = decode(codes[0].tolist())
    print(str)