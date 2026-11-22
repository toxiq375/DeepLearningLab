import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from config import MODEL_CONFIG, TRAINING_EPOCH 
# import model configs from config.py, which defines the configs of the model to be saved or loaded
from data import prepare_data, encode, decode
# prepare_data returns the needed data for training and evaluation
from checkpoint import save_checkpoint, load_checkpoint
from model import miniGPT

CHECKPOINT_PATH = (
    Path(__file__).resolve().parent
    # (__file__) means the current file, and Path() turns it into a Path object
    # .resolve() gets its absolute path, and .parent gets its parent path
    / "outputs"
    / "minigpt.pt"
)

def compute_loss(model, X, Y, vocab_size):
    logits = model(X)

    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        Y.reshape(-1)
    )
# I will organize the shape transformation during the whole project
# Assume that we have input data x, whose size is [2, 8], for 2 means batch_size, and 8 means context_size or token_size, which is 8 char in this project
# The predicted output size should be [2, 8, 9], while 9 means the vocab_size, each token will gets a logits with size of the whole vocab_size
# When computing the cross_entropy loss, the function takes(logits, targets) as input, and will compute the CE loss automatically
# While logits' shape should be [N, C], and the targets' shape is [N], C equals the length of vocab_size
# Here, the logits' shape is [B,T,C], and the targets' shape is [B,T], so we need the use the reshape(-1) to reshape them so that we can use the CE function
# Additionally, after embedding, the shape of the embedded x is [B,T,embed_dim]

def main():
    X_train, Y_train, X_val, Y_val, vocab_size, stoi, itos = prepare_data()

    torch.manual_seed(42)
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
    print("Initial Training Loss: ", initial_loss.item())

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
        logits_val = model(X_val)
        final_training_loss = compute_loss(
            model, X_train, Y_train, vocab_size
        )
        val_loss = compute_loss(
            model, X_val, Y_val, vocab_size
        )
        val_PPL = torch.exp(val_loss)

    print(f"Final Training Loss: {final_training_loss.item()}")
    print(f"Validation Loss: {val_loss.item()}")

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
        logits_loaded = loaded_model(X_val)

    assert torch.allclose(
        logits_val,
        logits_loaded,
        atol = 1e-6
    )

    generate_one(model, "hel", stoi, itos, context_size)
    generate(model, "hello wo", stoi, itos, vocab_size, context_size, 20, temperature=0.5, top_k=3)

def generate(model, prompt, stoi, itos, vocab_size, context_size, test_loops, temperature=1.0, top_k=None):
    if not prompt:
        raise ValueError("Prompt Must Not Be Empty")
    if temperature <= 0:
        raise ValueError("Temperature Must Be Positive")
    encoded = torch.tensor(
        [encode(stoi, prompt)], # Use [] to manually increase the shape of batch as 1
        dtype = torch.long
    )
    print("Encoded Shape: ", encoded.shape)

    model.eval()
    with torch.no_grad():
        for test_loop in range(test_loops):
            code_numbers_for_prediction = encoded[:, -context_size:] # When predicting the next token, we only need the last context_size tokens
            logits = model(code_numbers_for_prediction) # Shape [B, T, vocab_size], and the T shall be the context_size
            last_logits = logits[:, -1, :] # When using the probs of the logits to predict the next token, we only needs the last logits of each batch
            temperatured_logits = last_logits / temperature # The use of temperature

            if top_k is not None:
                if top_k < 1 or top_k > vocab_size:
                    raise ValueError("top_k must >=1 and <= vocab_size")
                top_values, _ = torch.topk(
                    temperatured_logits, 
                    k=top_k,
                    dim=-1
                )
                # values, indices = torch.topk(x, k, dim)
                # values are the k largest value of the x at the dim of -1, and indices are values' respective indices
                # Here we have selected the top k values of the x at the dim -1, and they are the "scores" of the potential next token
                threshlod = top_values[:, -1:] # Defines the threshold,  we will use it to select
                filtered_logits = temperatured_logits.masked_fill(
                    temperatured_logits < threshlod,
                    float("-inf")
                ) # masked_fill() will make the logits that minus threshold "-inf", after softmax, they will be zero
            else:
                filtered_logits = temperatured_logits
            probs = torch.softmax(filtered_logits, dim=-1)
            next_code_number = torch.multinomial(probs, num_samples=1)
            encoded = torch.cat([encoded, next_code_number], dim=1) # Remember to cat the predicted token to the whole embedded token sequence
        decoded = decode(itos, encoded.tolist()[0])
        # We difine the function decode to take (itos, a list of code_numbers) as input, so we need 1.squeeze the batch size which is 1 here, 2.make them into a list

        print(decoded)

        return "".join(decoded)

def generate_one(model, prompt, stoi, itos, context_size):
    if not prompt:
        raise ValueError("Prompt Must Not Be Empty")
    encoded = torch.tensor([encode(stoi, prompt)], dtype=torch.long)
    # [] must be used on enocde(), so that the shape of encoded is [1, text_chars_number]
    print("Encoded Shape: ", encoded.shape)

    code_numbers_for_prediction = encoded[:, -context_size:]

    model.eval()
    with torch.no_grad():
        logits = model(code_numbers_for_prediction)
        print("Logits Shape: ", logits.shape)
        last_logits = logits[:, -1, :]
        print("Last Logits Shape: ", last_logits.shape)

        probs = torch.softmax(last_logits, dim=-1)
        next_code_number = torch.multinomial(probs, num_samples=1)
        # num_samples Not num_sample

    print("Next Code_Number Shape: ", next_code_number.shape)
    print("Next Code_Number: ", next_code_number.item())
    print("Next Char: ", decode(itos, next_code_number[0].tolist()))

    return next_code_number

if __name__ == "__main__":
    main()