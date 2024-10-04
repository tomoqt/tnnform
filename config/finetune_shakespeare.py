import time

out_dir = 'out'
eval_interval = 5
eval_iters = 40
wandb_log = False # feel free to turn on
wandb_project = 'shakespeare'
wandb_run_name = 'ft-' + str(time.time())
dataset = 'shakespeare'
init_from = 'resume' 
block_size = 512
n_layer = 6
n_head = 6
# only save checkpoints if the validation loss improves
always_save_checkpoint = False
device = 'cpu'
warmup_iters = 1 # how many steps to warm up for

device_type = 'cpu' # for later use in torch.autocast

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 1
max_iters = 2000

# finetune at constant LR
learning_rate = 2e-5
decay_lr = False
