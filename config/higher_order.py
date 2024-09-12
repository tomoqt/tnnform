# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-shakespeare'
eval_interval = 1 # keep frequent because we'll overfit
eval_iters = 10
log_interval = 1 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

wandb_log = True # override via command line if you like
wandb_project = 'shakespeare-char-higher-order'
wandb_run_name = 'higher-order-gpt'

dataset = 'shakespeare'
gradient_accumulation_steps = 4
batch_size = 4
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 3
n_head = 3
n_embd = 384
dropout = 0.2
attention_order = 3  # Set the attention order to 3 for higher-order attention
learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 3000
lr_decay_iters = 3000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
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

