import torch
from pathlib import Path
from model import miniGPT

def save_checkpoint(model, model_config, itos, val_loss, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "chars": [itos[i] for i in range(len(itos))],
        "val_loss": float(val_loss)
    }

    torch.save(checkpoint, path)
    print(f"Checkpoint saved to: {path}")

def load_checkpoint(path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    model_config = checkpoint["model_config"]
    model = miniGPT(**model_config)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    chars = checkpoint["chars"]
    stoi = {ch:i for i, ch in enumerate(chars)}
    itos = {i:ch for i, ch in enumerate(chars)}

    return model, stoi, itos