import torch
import torch.nn as nn

# 数据创建与划分部分
text = "hello world\n"*8 # 创建原始数据
train_data = text[:72]
val_data = text[72:]

chars = sorted(set(train_data)) # 不要使用全部text建立词表
stoi = {} # 字符:ID
for idx, char in enumerate(chars):
    stoi[char] = idx
itos = chars # ID:字符, decoded to tokens

def encode(s:str):
    encoded = []
    for char in s:
        encoded.append(stoi[char])
    return encoded
def decode(l:list):
    decoded = []
    for code in l:
        decoded.append(itos[code])
    return decoded

train_ids = torch.tensor(encode(train_data), dtype = torch.long) # 先转换为tensor
X_train = []
Y_train = []
for i in range(len(train_ids)-3):
    X_train.append(train_ids[i:i+3])
    Y_train.append(train_ids[i+3])
X_train = torch.stack(X_train)
Y_train = torch.stack(Y_train)
val_ids = torch.tensor(encode(val_data), dtype = torch.long)
X_val = []
Y_val = []
for i in range(len(val_ids) - 3):
    X_val.append(val_ids[i:i+3])
    Y_val.append(val_ids[i+3])
X_val = torch.stack(X_val)
Y_val = torch.stack(Y_val)

# Embedding部分
# torch.manual_seed(42)
# embedding = nn.Embedding(9, 4) # size为(vocab_size, embedding_size)
# X_emb = embedding(X_train) # shape:(69,3,4)
# X_flat = X_emb.flatten(start_dim = 1) # shape:(69,12)

# 创建MLP进行学习，不再使用上方的方法
class ContextMLP(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, 4)
        self.linear_layer_1 = nn.Linear(12, 32)
        self.linear_layer_2 = nn.Linear(32, vocab_size)
        self.act = nn.Tanh()

    def forward(self,input):
        X_emb = self.embedding(input) # shape:(69,3,4)
        X_flat = X_emb.flatten(start_dim = 1) # shape(69,12)
        X_1 = self.linear_layer_1(X_flat) # (69,32)
        X_1 = self.act(X_1)
        logits = self.linear_layer_2(X_1) # (69, 9)

        return logits

def generate(model, test_text:str, test_loops):
    with torch.no_grad():
        encoded = encode(test_text)
        encoded_tensor = torch.tensor(encoded, dtype = torch.long).unsqueeze(0)
        for test_loop in range(test_loops):
            logits = model(encoded_tensor[:,-3:])
            probs = torch.softmax(logits, dim=-1)
            pred_id = torch.multinomial(probs, 1)
            encoded_tensor = torch.cat([encoded_tensor, pred_id], dim=1)
        decoded = decode(encoded_tensor.tolist()[0])
        print("".join(decoded))

        test_contexts = ["hel", "ell"]
        encoded_contexts = torch.tensor([encode(s) for s in test_contexts], dtype = torch.long)
        logits = model(encoded_contexts[:,-3:])
        probs = torch.softmax(logits, dim=-1)
        pred_ids = probs.argmax(dim = -1)

        for context, idx, prob in zip(
            test_contexts,
            pred_ids.tolist(),
            probs
        ):
            print(
                f"{context} -> {itos[idx]!r}",
                f"probability = {prob[idx].item():.4f}"
            )

def main():
    torch.manual_seed(42)
    model = ContextMLP(len(chars))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = 0.03,
        weight_decay=0
    )
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, Y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val)
        val_loss = criterion(val_logits, Y_val)

    generate(model, "hel", 20)

if __name__ == "__main__":
    main()

