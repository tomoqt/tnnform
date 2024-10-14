"""
Full definition of a GPT Language Model without MLP layers in the transformer blocks.
References:
1) The official GPT-2 TensorFlow implementation released by OpenAI:
   https://github.com/openai/gpt-2/blob/master/src/model.py
2) HuggingFace's transformers PyTorch implementation:
   https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass
import sys

import torch
import torch.nn as nn
from torch.nn import functional as F
import itertools

def generate_causal_mask(T, attention_order, device):
    # Generate all possible index combinations
    indices = torch.arange(T, device=device)
    grid = torch.meshgrid([indices for _ in range(attention_order)], indexing='ij')
    grid = torch.stack(grid, dim=-1)  # Shape: (T, T, ..., T, attention_order)

    # Create the causal mask
    # For each combination, check if the indices are in non-decreasing order
    mask = (grid.diff(dim=-1) <= 0).all(dim=-1)
    return mask  # Shape: (T, T, ..., T)

class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class TrueHigherOrderAttention(nn.Module):
    def __init__(self, config, extract_stats=False):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.attention_order = config.attention_order
        self.head_dim = config.n_embd // config.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)  # Normal scaling

        # Projections for Q and Ks
        self.projections = nn.ModuleList([
            nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            for _ in range(self.attention_order)
        ])

        # Value projections for each K (excluding Q)
        self.value_projections = nn.ModuleList([
            nn.Linear(self.head_dim, self.head_dim, bias=config.bias)
            for _ in range(self.attention_order - 1)
        ])

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.extract_stats = extract_stats

    def forward(self, x):
        B, T, C = x.size()

        # Compute projections
        projs = []
        for proj in self.projections:
            p = proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
            projs.append(p)

        # Compute higher-order attention
        y, att_stats = self.true_higher_order_attention(projs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        # Return output and stats
        stats = att_stats if self.extract_stats else {}
        return y, stats

    def true_higher_order_attention(self, projs):
        B, nh, T, hs = projs[0].shape
        attention_order = self.attention_order

        # Compute initial attention scores between Q and K1
        att = torch.einsum('bhid,bhjd->bhij', projs[0], projs[1]) * self.scale  # (B, nh, T, T)

        # Expand attention tensor by contracting with additional Ks
        for idx in range(2, attention_order):
            key = projs[idx]  # Next key
            # Expand att to include new dimension
            att = torch.einsum('bh...i,bhkd->bh...ik', att, key)  # Expand to (B, nh, T, T, ..., T)

        # Apply causal mask
        causal_mask = generate_causal_mask(T, attention_order, projs[0].device)  # Shape: (T, T, ..., T)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T, ..., T)
        att = att.masked_fill(causal_mask == 0, float('-inf'))

        # Flatten attention tensor for softmax
        att = att.view(B, nh, T, -1)  # Shape: (B, nh, T, N), where N = T^(attention_order - 1)

        # Apply softmax over the last dimension
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Build large V tensor
        # Get Vs from the keys (excluding Q)
        Vs = [self.value_projections[i](projs[i+1]) for i in range(attention_order - 1)]

        # Reshape Vs to align dimensions
        Vs_reshaped = []
        for i, V in enumerate(Vs):
            # Shape: (B, nh, [1]*i, T, [1]*(attention_order - 2 - i), hs)
            shape = [B, nh] + [1]*i + [T] + [1]*(attention_order - 2 - i) + [hs]
            Vs_reshaped.append(V.view(*shape))

        # Compute the element-wise product to form the large V tensor
        V = Vs_reshaped[0]
        for V_i in Vs_reshaped[1:]:
            V = V * V_i  # Element-wise multiplication with broadcasting

        # Flatten V to match the dimensions of att
        V = V.view(B, nh, -1, hs)  # Shape: (B, nh, N, hs)

        # Compute output by contracting att with V
        y = torch.einsum('bhij,bhjd->bhid', att, V)  # Output shape: (B, nh, T, hs)

        # Collect stats if needed
        stats = {}
        if self.extract_stats:
            stats = {
                'attention_tensor_mean': y.mean().item(),
                'attention_tensor_std': y.std().item(),
                'attention_tensor_max': y.max().item(),
                'attention_tensor_min': y.min().item()
            }

        return y, stats

class Block(nn.Module):

    def __init__(self, config, extract_stats=False):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = TrueHigherOrderAttention(config, extract_stats=extract_stats)
        # Removed the second LayerNorm and MLP
        self.extract_stats = extract_stats

    def forward(self, x):
        attn_input = self.ln_1(x)
        attn_output, attn_stats = self.attn(attn_input)
        x = x + attn_output

        # No MLP or second residual connection

        return x, attn_stats if self.extract_stats else {}

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    attention_orders: list = None  # New parameter for per-head attention orders
    attention_order: int = 4

    def __post_init__(self):
        if self.attention_orders is None:
            # Default: homogeneously distribute between 2 and max_order
            max_order = max(2, self.attention_order)  # Ensure at least order 2
            self.attention_orders = [((i % (max_order - 1)) + 2) for i in range(self.n_head)]
        else:
            assert len(self.attention_orders) == self.n_head, "Length of attention_orders must match n_head"
            assert all(order >= 2 for order in self.attention_orders), "All attention_orders must be >= 2"

class GPT(nn.Module):

    def __init__(self, config, extract_stats=False):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, extract_stats=extract_stats) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

        # Initialize all weights
        self.apply(self._init_weights)

        # Report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

        self.extract_stats = extract_stats

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # Forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # Token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # Position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        all_stats = {}
        for i, block in enumerate(self.transformer.h):
            x, block_stats = block(x)
            if self.extract_stats:
                all_stats[f'block_{i}'] = block_stats
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, all_stats if self.extract_stats else (logits, loss)

    def crop_block_size(self, block_size):
        # Model surgery to decrease the block size if necessary
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # Filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # Create optim groups
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS."""
        # First estimate the number of flops we do per iteration
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # Express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b, t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        for _ in range(max_new_tokens):
            # If the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # Forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # Pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
