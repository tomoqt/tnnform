# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-shakespeare'
eval_interval = 100 # keep frequent because we'll overfit
eval_iters = 10
log_interval = 1 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

wandb_log = True # override via command line if you like
wandb_project = 'shakespeare-char-higher-order'
wandb_run_name = 'higher-order-gpt'

dataset = 'shakespeare'
gradient_accumulation_steps = 1
batch_size = 1
block_size = 196 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 3
n_head = 3
n_embd = 768
dropout = 0
attention_order = 2  # Set the attention order to 3 for higher-order attention
learning_rate = 1e-4 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-7 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially
weight_decay = 1e-1

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
# Run the training script in the background and detach from the shell
import subprocess
import os



# Command to run the training script

# Execute the command

