# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

import subprocess
import os

# Output and Evaluation Settings
out_dir = 'out-shakespeare'
eval_interval = 100  # keep frequent because we'll overfit
eval_iters = 10
log_interval = 1  # don't print too too often

# Checkpointing
# We expect to overfit on this small dataset, so always save checkpoints
always_save_checkpoint = True

# Weights & Biases (wandb) Logging
wandb_log = True  # override via command line if you like
wandb_project = 'shakespeare-char-higher-order'
wandb_run_name = 'higher-order-gpt'

# Dataset Configuration
dataset = 'shakespeare'
gradient_accumulation_steps = 1
batch_size = 1
block_size = 196  # context of up to 196 previous characters

# Model Architecture
n_layer = 3
n_head = 1
n_embd = 384
dropout = 0.0
bias = False

# Attention Mechanism
attention_order = 3  # Specify the attention order for each head
attention_sparsities = [False, True, False]  # Specify sparsity for each head
window_size = 16  # Window size for sparse attention
sparse_attention = [True, False, True]  # Specify which heads use sparse attention

# Optimizer and Learning Rate Configuration
learning_rate = 1e-4  # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000  # make equal to max_iters usually
min_lr = 1e-7  # learning_rate / 10 usually
beta1 = 0.9
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small
weight_decay = 1e-1

# Learning Rate Decay Settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 100  # not super necessary potentially

# Gradient Clipping
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0

# Device and Compilation Settings
# Uncomment the following lines if running on a MacBook or CPU-only environment
# device = 'cpu'  # run on cpu only
# compile = False  # do not torch compile the model

# Additional Imports (if needed)
# These imports are placeholders and can be adjusted based on your training setup
# import subprocess
# import os

# Command to run the training script
# Example:
# python train.py --batch_size=32 --compile=False

# Execute the command
# subprocess.run(["python", "train.py", "--batch_size=32", "--compile=False"])
