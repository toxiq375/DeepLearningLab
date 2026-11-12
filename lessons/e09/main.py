import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CONTEXT_SIZE, TRAINING_EPOCH
from data import prepare_data, encode, decode
from model import SingleHead, MultiHead, TransformerBlock, miniGPT

def compute_loss(model, X, Y, vocab_size):
    logits = model(X)

    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        Y.reshape(-1)
    )
# CrossEntropyLoss()，接收[B*T, V]或者[B,V,T],可以使用reshape(-1, V)或transpose(1, 2)

def main():
    X_train, Y_train, X_val, Y_val, vocab_size = prepare_data()

    torch.manual_seed(42)
    model = miniGPT(
        vocab_size = vocab_size,
        context_size = CONTEXT_SIZE,
        embed_dim = 16,
        input_channels = 16,
        attn_channels = 8,
        output_channels = 16,
        num_heads = 2
    ) # 记得传入参数创建miniGPT
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.03,
        weight_decay=False
    )

    model.eval()
    with torch.no_grad():
        initial_loss = compute_loss(
            model, X_train, Y_train, vocab_size
        )
    print("Initial train loss:", initial_loss.item())

    model.train()
    for epoch in range(TRAINING_EPOCH):
        model.train()
        optimizer.zero_grad()

        loss = compute_loss(
            model, X_train, Y_train, vocab_size
        )
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        final_train_loss = compute_loss(
            model, X_train, Y_train, vocab_size
        )
        val_loss = compute_loss(
            model, X_val, Y_val, vocab_size
        )

    print(f"Final train loss:", final_train_loss.item())
    print(f"Validation loss:", val_loss.item())

def generate(model, test_text, test_loops):
    with torch.no_grad:
        encoded = torch.tensor(encode(test_text), dtype=torch.long).unsqueeze(0)
        for test_loop in range(test_loops):
            logits = model(encoded)
            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.multinomial(probs, 1)
            encoded = torch.cat([encoded, pred_ids], dim=1)
        decoded = decode(encoded.tolist()[0])
        return "".join(decoded)

if __name__ == "__main__":
    main()


