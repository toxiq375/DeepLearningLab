import torch
from pathlib import Path
from model import miniGPT
# import the whole model from the model.py, we need to know what kind of model we need to save or load

def save_checkpoint(model, model_config, itos, val_loss, val_PPL, path):
    # model: just save the model, containing its parameters
    # model_config: save the parameters that construct the whole model, for example, how many layers the model builds
    # path: the saving path of the checkpoint file
    # the output of the model are code_numbers, thus we need to save a itos so that we can integrate the code_numbers into chars and make them into sentences

    path = Path(path) # before this line, path is a string, after Path(), it turns into a Path object
    path.parent.mkdir(parents=True, exist_ok=True)
    # path.parnet gets the path's parent directory
    # parents=True allows it to create multiple directories
    # exist_ok=True makes it to pass if there are already parent directories and it wont raise Error

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "chars": [itos[i] for i in range(len(itos))],
        "val_loss": float(val_loss),
        "val_PPL": float(val_PPL)
    } # checkpoint is essentially a dictionary that stores the information of a whole project, not just a model

    torch.save(checkpoint, path)
    print(f"Checkpoint Saved To {path}")

def load_checkpoint(path):
    checkpoint = torch.load(
        path,
        map_location="cpu", # when load the tensor of the checkpoint, put them on the cpu
        weights_only=True # ... safety limitations?
    )

    model_config = checkpoint["model_config"]
    model = miniGPT(**model_config) # ** means unpack the dictionary

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    chars = checkpoint["chars"]
    stoi = {ch:i for i, ch in enumerate(chars)}
    itos = {i:ch for i, ch in enumerate(chars)}

    return model, stoi, itos