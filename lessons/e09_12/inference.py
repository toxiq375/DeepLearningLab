from checkpoint import load_checkpoint
from main import generate
from pathlib import Path

checkpoint_path = (
    Path(__file__).resolve().parent
    / "outputs"
    / "minigpt.pt"
)

model, stoi, itos = load_checkpoint(checkpoint_path)

text = generate(
    model, 
    "hel",
    stoi,
    itos,
    vocab_size = len(stoi),
    context_size=8,
    test_loops=20,
    top_k=3
)
