"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000 # Evaluate the model after each 2000 training iterations, Not Epoch
log_interval = 1
eval_iters = 200 # When Eval, extract 200 batches from the training dataset and the val set
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval. If False, only save the checkpoint when the eval loss is the best one(or better than the former one)
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

# wandb logging
# WandB means "Weight&Biases", A Visualization Tool of machine learning
wandb_log = False # disabled by default
wandb_project = 'owt' # wdb project name
wandb_run_name = 'gpt2' # 'run' + str(time.time())

# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes. Given the batch_size of 12, it is 40 * 12 = 480 batches.
# So it means the optimizer will update the gradient every 40 iterations, totally 480 batches 
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size. Every epoch inputs 12 sequences into the forward
block_size = 1024 # i.e. context length, means each sequence contains 1024 tokens

# model
n_layer = 12 # The number of TransformerBlock, which means that the model has 12 Transformerblocks
n_head = 12 # Each MHA has 12 attention heads
n_embd = 768 # Embedding shapes, each token will be embedded into 768 dims
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?

# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
# Review of the weight decay
# When using it, each time the optimizer updates the model's weight, the weight will times a num that less than 1, decreasing the model's weight
# i.e. \theta_{t+1} = (1-\lambda\times\eta)\theta_t - \eta \times gradient
beta1 = 0.9
beta2 = 0.95 # beta1 and beta2 of the AdamW optimizer
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# Clip the grad's L2-norm to the maxium of 1

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
# At the beginning of the training, the gradient is randomly initialized, so we need to slowly increase the learning rate of the model,  avoiding the fluctuation of the training
# nanoGPT uses the Linear Warmup + Cosine Decay, which means that the learning rate will increase at Warmup period, and then cosine annealing after warmup(annealing means decay)
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# DDP means Distributed Data Parallel, which allows the model to be trained on multiple GPUs
# nccl is normally used for nvidia GPU communications 

# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# globals.() is the built-in function of Python, which can return a dictionary of global variables
# For example, globals()["learning_rate"] = 6e-4, and we can modify the learning_rate by using globals()["learning_rate"] = 
# .items() returns the pair of the key and the value, the config_keys only collect the keys and avoids the keys start with _, and its values' type must be one of the (int, float, bool, str)
exec(open('configurator.py').read()) # overrides from command line or config file
# open() will open the given file, which is the Python program here. .read() will get the string of the file
# exec() means execute the Python code inside the read string, which will defines some other parameters in the configuration Python programs, and may cover some defined parameters
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# Should be used after exec(), cause exec() may cover some configs parameters
# -----------------------------------------------------------------------------

# Since I just use 1 GPU, I'll just pass the part of the ddp code :P
# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
# If there is no environment variable RANK, os.environ.get('RANK', -1) will gets string "0"
# dict.get(key, default), if key exists, return its value, else, return default value
# For a single GPU environment, there is usually no RANK variable, so os.environ.get('RANK', -1) returns -1, and then ddp = False
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    # Above code means that the program will see the current process as the only training process

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size # 40 * 1 * 12 * 1024 Tokens / 1 training iteration
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process: # Single GPU, master_process = True
    os.makedirs(out_dir, exist_ok=True) # Create the directory
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul, i.e. aloow CUDA to use TensorFloat-32 computational mode to accelerate some FP32 matrix computation
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast, check if the device string has "cuda" string part
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype] # Create a dictionary to search the actual dtype of string "dtype"
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
# nullcontext() is a manager from Python contextlib, which means operate nothing when enter or quit
# torch.amp.autocast() will mix the precision automatically
# We can see the code as a pre-prepared environment for AMP, for exmaple, with ctx: , the code after it will use AMP

# poor man's data loader
data_dir = os.path.join('data', dataset) # Join the data path, like "/" in the Path module
def get_batch(split):
    # The function is defined to take the split mode, and create a batch of data(12 seqs) from the respective data. It goes like below:
    # 1st, we need to confirm which set of data we need to create its batch
    # 2nd, since we need 12 seqs in 1 batch, we need to get their repective start index
    # 3rd, after getting the starts, slice them at the length of token_size(1024 here) from the data
    # 4th, type deformation, then stack them into the shape of [B, T]
    # 5th, move them to the repective device, which is defined globally above
    # 6th, return them

    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r') # Map the training data from disk to RAM
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # torch.randint(high, size), randomly extract integers from range[0, high), at the size of size
    # Batch size is 12, so we will extract 12 intergers from the range of range(0, len(data) - block_size), as the starting index of each sequence
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    # When read from train.bin file, dtype = np.uint16, we need to turn it into np.int64 to align on torch.long, i.e. int64
    # Remember to turn the Numpy to Tensor with torch.from_numpy()
    # Shape: [1024] -> stack -> [12, 1024]
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True) # Acceleration, Pined Memory
    else:
        x, y = x.to(device), y.to(device)
    return x, y # The function returns the training batch data x, and the target batch data y, y = x+1

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset, i.e. meta data, meta.pkl here
meta_path = os.path.join(data_dir, 'meta.pkl') # data_dir is now a global variable
meta_vocab_size = None # Initialize the meta_vocab_size as None, cause we may not find meta.pkl
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f: # Open the meta_data as f in binary, cause it is stored in binary
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size'] # meta.pkl is a binary file that stores vocab_size, stoi, itos in the dictionary form
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
# The project will create a new model here if it is trained from scratch, or load a trained model if init_form is "resume" or "gpt2"
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
# dict() will create a dictionary with the given key-value pairs to initilize the model. Btw the values have been defined globally above, and the keys are the construction parameters of the model
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.' # Somehow when using torch.compile(), there will be a strange prefix "_orig_mod."
    for k,v in list(state_dict.items()): # Use .items() to unpack the dictionary of state_dict
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k) # Slice the key's to recover its name
            # .pop() will remove the given key and return its value, so the process above is to create a new correct key, and assign the right value to the new key, then kill the wrong key name
    model.load_state_dict(state_dict) # Load the trained weights
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout) # When load from gpt2, most of the parameters have to align on the given model, except dropout
    model = GPT.from_pretrained(init_from, override_args)
    # .from pretrained is a method defined in the GPT Class
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
        # getattr() is a built-in function of Python, getattr(model.config, n_layer) equals model.config.n_layers, it's a function that can get object's attributes
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size) # .crop_block_size is a method defined in the GPT Class
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)
# Let's discuss the difference between the last two loading formats above:
# In the resume mode, the checkpoint_model_args is a dictionary, so we can use checkpoint_model_args[k] to index its value
# In the gpt2 mode, model.config is an instance in the Class GPTConfig, and the initialization function in the GPT Class will takes a config to initialize the GPT model
# To access its attributes, we need to use model.config.n_layers instead of using model.config["n_layers"]
# In the for loop, k is a Python string like 'n_layers'. But if u write 'model.config.k', it will search the attribute named 'k' instead of 'n_layers',so we must use getattr(model.config, k)
# Btw, look at the model_args of the gpt2 mode, it is not used for initializing the model. We assign it because we need it to save the checkpoint
# Why we use GPT.from_pretrained(), and use model.crop_block_size?
# 1st, we can use GPT.crop_block_size(), but we must give it model, like GPT.crop_block_size(model, block_size)
# 2nd, .from_pretrained has a Python decorator of '@classmethod', so it's a class method, it will automatically pass in the class parameter into the function
# Why we can use model.config.k, is because model is the instance of the Class GPT, to call its attributes, we need to use '.'. Inside the GPT Class, there shall be a code like'self.config = config', it takes a config and make it model's attribute, that is the source of model.config calling
# Likewise, the config we passed in is also a Class defined in the respective code, so we need to use config.n_layer to call its own attributes
# And because the Class GPTConfig use the @dataclass decorator, it doesn't need to use 'self.' to name its attributes

# initialize a GradScaler. If enabled=False scaler is a no-op
# scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
# Review of the Gradscaler: Using FP16 at training may lead some some gradients to 0 because the limitation of FP16, so before backward, enlarge the loss and then backward, so that the gradient caculated will be larger too. Then reduce the gradient
# enavled parameter means when dtype = float16, using Gradscaler, otherwise there is no need to use it 
scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(dtype=='float16')
) # Let's use the new API in PyTorch 2.10

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type) # Create the optimizer, its an instance method defined in the Class GPT
# Why dont we use the AdamW directly? Because we hope to use different strategy on different parameters according to its dimensions, it determines whether to use weight_decay or not
# Reviews of weight_decay: continously decrease the model's weight during training. When param.dim() >= 2, use it. Otherwise do not use it.
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer']) 
    # The optimizer is also updating during training, and its states keeps changing. So we need to recover the optimizer's state too.
checkpoint = None # free up memory
# The checkpoint previously save a dictionary which saves a mount of Tensor, release it

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx: # Which is defined above. When CPU, ctx = nullcontext(), execute nothing. When GPU, ctx = torch.amp.autocast(device_type = device_type, dtype=dtype)
                logits, loss = model(X, Y)
            losses[k] = loss.item() # .item() turns a Tensor into Python value, while .items() unpacks/returns key-value pairs in a dictionary
        out[split] = losses.mean()
    model.train() # Remember to turn on the train mode of the model
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
