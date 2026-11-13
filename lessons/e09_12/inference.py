from checkpoint import load_checkpoint
from main import generate

model, stoi, itos = load_checkpoint("./outputs/minigpt.pt")

text = generate(
    model, 
    "hel",
    stoi,
    itos,
    context_size=8,
    test_loops=20
)

print(text)