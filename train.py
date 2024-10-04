"""
This training script can be run both on a single GPU in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 GPUs on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 GPUs across 2 nodes, example:
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
import wandb
import torch.distributed as dist
import matplotlib.pyplot as plt
import io
import argparse

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = True  # disabled by default
wandb_project = 'nanoGPT'
wandb_run_name = 'gpt2_experiment'
wandb_log_freq = 1  # log to wandb every 1 iteration
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8  # used to simulate larger batch sizes
batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
attention_order = 3  # Default to regular attention, can be overridden
# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl'  # 'nccl', 'gloo', etc.
# system
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'float16'  # 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True  # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------

# Add this near the top of the file, after imports
parser = argparse.ArgumentParser()
parser.add_argument('--verbose_logging', action='store_true', help='Enable verbose logging')
parser.add_argument('--extract_stats', action='store_true', help='Extract and log detailed statistics (slows down training)')
parser.add_argument('--window_size', type=int, default=16, help='Window size for sparse attention')
parser.add_argument('--sparse_attention', type=str, nargs='+', help='List indicating if each head uses sparse attention (1 for True, 0 for False)')
parser.add_argument('--attention_order', type=int, help='Global attention order (overridden if attention_orders are specified)')
args, _ = parser.parse_known_args()
verbose_logging = args.verbose_logging
extract_stats = args.extract_stats
window_size = args.window_size
sparse_attention_input = args.sparse_attention

# ... (rest of the configurations)

# Modify this line
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config_keys.extend(['verbose_logging', 'extract_stats', 'window_size', 'sparse_attention_input'])

exec(open('configurator.py').read())  # overrides from command line or config file

# Add verbose_logging and extract_stats to globals
globals()['verbose_logging'] = verbose_logging
globals()['extract_stats'] = extract_stats
globals()['window_size'] = window_size

# Process sparse_attention_input
if sparse_attention_input is not None:
    if len(sparse_attention_input) != n_head:
        raise ValueError(f"Length of --sparse_attention ({len(sparse_attention_input)}) must match n_head ({n_head})")
    # Convert '1'/'0' strings to boolean
    sparse_attention = [bool(int(sa)) for sa in sparse_attention_input]
else:
    # Default: all heads use dense attention
    sparse_attention = [False for _ in range(n_head)]

config = GPTConfig(
    block_size=block_size,
    vocab_size=None,  # to be set based on dataset or pre-trained model
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
    bias=bias,
    attention_order=attention_order,
    window_size=window_size,
    sparse_attention=sparse_attention
)

# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    config.vocab_size = meta_vocab_size if meta_vocab_size is not None else 50304
    model = GPT(config, extract_stats=extract_stats)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'attention_order', 'window_size', 'sparse_attention']:
        config.__dict__[k] = checkpoint_model_args[k]
    # create the model
    model = GPT(config, extract_stats=extract_stats)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    # If sparse_attention is not specified, set to all False
    if sparse_attention_input is not None:
        override_args['sparse_attention'] = sparse_attention
    else:
        override_args['sparse_attention'] = [False for _ in range(n_head)]
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'attention_order', 'window_size', 'sparse_attention']:
        config.__dict__[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < config.block_size:
    model.crop_block_size(block_size)
    config.block_size = block_size  # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

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
            with ctx:
                logits, loss, _ = model(X, Y)  # Ignore the stats during evaluation
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    wandb.init(project=wandb_project, name=wandb_run_name, config=config.__dict__)
    wandb.watch(model, log="all", log_freq=log_interval)
    n_params = model.get_num_params()
    attention_order = model.config.attention_order
    print(f"Number of parameters: {n_params:,}")
    print(f"Attention order: {attention_order}")
    wandb.log({
        "n_params": n_params,
        "attention_order": attention_order
    }, step=iter_num)

# Add this function to create and save heatmap images locally
def create_and_save_heatmap(data, title, filename):
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.savefig(filename)
    plt.close()

# Create a directory to store heatmap images
os.makedirs('heatmaps', exist_ok=True)

# training loop
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0
stats_window = {}
window_size_log = 100  # Renamed to avoid confusion with attention window_size
current_window = 0

# Keep track of the last 3 iterations for each image type
last_iterations = {'proj_values': [], 'proj_grads': [], 'attention_tensor': []}

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
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': config.__dict__,  # Save the entire config dictionary
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                attention_order_str = "_".join([str(order) for order in config.attention_orders])
                checkpoint_name = f'ckpt_order_{attention_order_str}.pt'
                print(f"saving checkpoint to {out_dir}/{checkpoint_name}")
                torch.save(checkpoint, os.path.join(out_dir, checkpoint_name))

                if wandb_log:
                    wandb.save(os.path.join(out_dir, checkpoint_name))
                    # artifact = wandb.Artifact(name=f"model_checkpoint_{attention_order_str}", type="model")
                    # artifact.add_file(os.path.join(out_dir, checkpoint_name))
                    # wandb.log_artifact(artifact))
    if iter_num == 0 and eval_only:
        break

    # forward backward update
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            if extract_stats:
                logits, loss, all_stats = model(X, Y)
            else:
                logits, loss = model(X, Y)[:2]  # Only get logits and loss
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()

        # Collect gradient statistics
        if master_process and wandb_log and extract_stats:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        all_stats[f'{name}_grad_mean'] = param.grad.mean().item()
                        all_stats[f'{name}_grad_std'] = param.grad.std().item()
                        all_stats[f'{name}_grad_max'] = param.grad.max().item()
                        all_stats[f'{name}_grad_min'] = param.grad.min().item()

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
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")
        if wandb_log and iter_num % wandb_log_freq == 0:
            wandb.log({
                "iter": iter_num,
                "train/loss": lossf,
                "lr": lr,
                "mfu": running_mfu,
            })

    # Verbose logging (only if enabled)
    if verbose_logging and master_process and extract_stats:
        # Collect statistics from the model
        flat_stats = {}
        for block, block_stats in all_stats.items():
            if isinstance(block_stats, dict):
                for stat_name, stat_value in block_stats.items():
                    flat_stats[f"{block}_{stat_name}"] = stat_value
            else:
                flat_stats[block] = block_stats

        # Create and save heatmap images
        if iter_num % 1000 == 0:  # Only create heatmaps every 1000 iterations
            os.makedirs('heatmaps', exist_ok=True)
            for i, block in enumerate(raw_model.transformer.h):
                for proj_idx, proj in enumerate(block.attn.projections):
                    filename = f'heatmaps/block_{i}_proj_{proj_idx}_values_iter_{iter_num}.png'
                    create_and_save_heatmap(proj.weight.data.cpu().numpy(),
                                            f'Block {i} Projection {proj_idx} Values (Iter {iter_num})',
                                            filename)

                    if proj.weight.grad is not None:
                        filename = f'heatmaps/block_{i}_proj_{proj_idx}_grads_iter_{iter_num}.png'
                        create_and_save_heatmap(proj.weight.grad.cpu().numpy(),
                                                f'Block {i} Projection {proj_idx} Gradients (Iter {iter_num})',
                                                filename)

            if 'attention_tensor' in flat_stats:
                attention_tensor = flat_stats['attention_tensor']
                # Depending on the attention_order and window_size, attention_tensor can be high-dimensional
                # For visualization, you might want to flatten or select specific slices
                if attention_tensor.ndim > 2:
                    # Select the first slice for visualization
                    attention_slice = attention_tensor[0, :, :attention_tensor.shape[-1] // 2]
                else:
                    attention_slice = attention_tensor
                filename = f'heatmaps/attention_tensor_iter_{iter_num}.png'
                create_and_save_heatmap(attention_slice.cpu().numpy(),
                                        f'Attention Tensor (Iter {iter_num})',
                                        filename)

        # Log per iteration stats
        if iter_num % 100 == 0:  # Only log detailed stats every 100 iterations
            print(f"Iter {iter_num} Stats:")
            for key, value in flat_stats.items():
                if isinstance(value, torch.Tensor):
                    if value.numel() == 1:
                        print(f"  {key}: {value.item():.4f}")
                    else:
                        print(f"  {key}: tensor of shape {value.shape}")
                elif isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

        # Aggregate stats
        for key, value in flat_stats.items():
            if key not in stats_window:
                stats_window[key] = 0.0
            stats_window[key] += value
        current_window += 1

        # Aggregate and log every 'window_size_log' iterations
        if current_window >= window_size_log:
            averaged_stats = {k: v / window_size_log for k, v in stats_window.items()}
            print(f"Average Stats over last {window_size_log} iterations:")
            for key, value in averaged_stats.items():
                if isinstance(value, torch.Tensor):
                    if value.numel() == 1:
                        print(f"  {key}: {value.item():.4f}")
                    else:
                        print(f"  {key}: tensor of shape {value.shape}")
                elif isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
            stats_window = {k: 0.0 for k in stats_window}
            current_window = 0

    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

if wandb_log and master_process:
    wandb.finish()
