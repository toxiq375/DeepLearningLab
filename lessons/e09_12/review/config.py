VOCAB_SIZE = 9 # vocab_size
CONTEXT_SIZE = 8 # context_size
EMBED_DIM = 16 # embedding channels dim
INPUT_CHANNELS = 16
ATTN_CHANNELS = 8
OUTPUT_CHANNELS = 16
NUM_HEADS = 2 # numbers of attention heads

MODEL_CONFIG = {
    "vocab_size": VOCAB_SIZE,
    "context_size": CONTEXT_SIZE,
    "embed_dim": EMBED_DIM,
    "input_channels": INPUT_CHANNELS,
    "attn_channels": ATTN_CHANNELS,
    "output_channels": OUTPUT_CHANNELS,
    "num_heads": NUM_HEADS
}

TRAINING_EPOCH = 1000