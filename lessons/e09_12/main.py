import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from config import MODEL_CONFIG, TRAINING_EPOCH
from data import prepare_data, encode, decode
from model import miniGPT
from checkpoint import save_checkpoint, load_checkpoint

CHECKPOINT_PATH = (
    Path(__file__).resolve().parent
    / "outputs"
    / "minigpt.pt"
)

def compute_loss(model, X, Y, vocab_size):
    logits = model(X)

    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        Y.reshape(-1)
    )
# CrossEntropyLoss()，接收[B*T, V]或者[B,V,T],可以使用reshape(-1, V)或transpose(1, 2)

def main():
    X_train, Y_train, X_val, Y_val, vocab_size, stoi, itos = prepare_data()

    torch.manual_seed(42)
    # model = miniGPT(
    #     vocab_size = vocab_size,
    #     context_size = CONTEXT_SIZE,
    #     embed_dim = 16,
    #     input_channels = 16,
    #     attn_channels = 8,
    #     output_channels = 16,
    #     num_heads = 2
    # ) # 记得传入参数创建miniGPT
    model = miniGPT(**MODEL_CONFIG)
    context_size = MODEL_CONFIG["context_size"]
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
        logits_before = model(X_val)
        final_train_loss = compute_loss(
            model, X_train, Y_train, vocab_size
        )
        val_loss = compute_loss(
            model, X_val, Y_val, vocab_size
        )
        val_PPL = torch.exp(val_loss)

    print(f"Final train loss:", final_train_loss.item())
    print(f"Validation loss:", val_loss.item())

    save_checkpoint(
        model = model,
        model_config = MODEL_CONFIG,
        itos = itos,
        val_loss = val_loss,
        val_PPL = val_PPL,
        path = CHECKPOINT_PATH
    )

    loaded_model, loaded_stoi, loaded_itos = load_checkpoint(
        CHECKPOINT_PATH
    )

    with torch.no_grad():
        logits_after = loaded_model(X_val)

    assert torch.allclose(
        logits_before,
        logits_after,
        atol = 1e-6
    )

    generate_one(model, "hel", stoi, itos, context_size)
    generate(model, "hello wo", stoi, itos, vocab_size, context_size, 20, temperature=0.5, top_k=3)

def generate(model, prompt, stoi, itos, vocab_size, context_size, test_loops, temperature=1.0, top_k=None):
    if not prompt:
        raise ValueError("Prompt must not be empty!")
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    
    encoded = torch.tensor([encode(stoi, prompt)], dtype=torch.long)
    print("Encoded shape: ", encoded.shape)

    model.eval()
    with torch.no_grad():
        for test_loop in range(test_loops):
            idx_cond = encoded[:, -context_size:]
            logits = model(idx_cond) 
            last_logits = logits[:, -1, :]
            adjusted_logits = last_logits / temperature
            if top_k is not None:
                if top_k < 1 or top_k > vocab_size:
                    raise ValueError("top_k must >= 1 and <= vocab_size")
                # top_values, top_indices = torch.topk(
                #     adjusted_logits,
                #     k=3,
                #     dim=-1
                # )
                # probs = torch.softmax(top_values, dim=-1)
                # sample_index = torch.multinomial(probs, num_samples=1)
                # next_id = torch.gather(
                #     top_indices,
                #     dim=-1,
                #     index=sample_index
                # )
                top_values, _ = torch.topk(
                    adjusted_logits,
                    k=top_k,
                    dim=-1
                )
                threshold = top_values[:, -1:]
                filtered_logits = adjusted_logits.masked_fill(
                    adjusted_logits < threshold,
                    float("-inf")
                )
            else:
                filtered_logits = adjusted_logits
            probs = torch.softmax(filtered_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            encoded = torch.cat([encoded, next_id], dim=1)
        decoded = decode(itos, encoded.tolist()[0])
        
    print(decoded)
    return "".join(decoded)

def generate_one(model, prompt, stoi, itos, context_size):

    if not prompt:
        raise ValueError("Prompt must not be empty")

    encoded = torch.tensor([encode(stoi, prompt)], dtype=torch.long)
    print("Encoded shape: ", encoded.shape)

    idx_cond = encoded[:, -context_size:] # -context_size后记得加:

    model.eval()
    with torch.no_grad():
        logits = model(idx_cond)
        print("Logits shape: ", logits.shape)
        last_logits = logits[:, -1, :]
        print("Last logits shape: ", last_logits.shape)

        probs = torch.softmax(last_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1) # num_samples不是num_sample

    print("Next ID shape: ", next_id.shape)
    print("Next ID: ", next_id.item())
    print("Next char: ", decode(itos, next_id[0].tolist()))

    return next_id

if __name__ == "__main__":
    main()
    


