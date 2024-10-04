import unittest
import torch
import torch.nn as nn
import math
from model import TrueHigherOrderAttention

def generate_causal_mask(T, attention_order, device='cpu'):
    # Generate all possible index combinations
    indices = torch.arange(T, device=device)
    grid = torch.meshgrid([indices for _ in range(attention_order)], indexing='ij')
    grid = torch.stack(grid, dim=-1)  # Shape: (T, T, ..., T, attention_order)

    # Create the causal mask
    # For each combination, check if the indices are in non-increasing order
    mask = (grid.diff(dim=-1) <= 0).all(dim=-1)
    return mask  # Shape: (T, T, ..., T)

class TestHigherOrderAttentionCausality(unittest.TestCase):
    def setUp(self):
        """
        Set up a simple higher-order attention module for testing.
        We'll mock the projection layers to produce deterministic outputs.
        """
        class MockGPTConfig:
            def __init__(self, n_embd, n_head, attention_orders, bias=True, dropout=0.0, block_size=10, vocab_size=10000):
                self.n_embd = n_embd
                self.n_head = n_head
                self.attention_orders = attention_orders
                self.bias = bias
                self.dropout = dropout
                self.block_size = block_size
                self.vocab_size = vocab_size
                self.max_attention_order = max(attention_orders)

        # Example configuration: 2 heads, one with attention_order=2, another with attention_order=3
        self.config = MockGPTConfig(
            n_embd=4,
            n_head=2,
            attention_orders=[2, 3],
            bias=False,
            dropout=0.0,
            block_size=5,
            vocab_size=100
        )

        # Initialize the higher-order attention module
        self.attention = TrueHigherOrderAttention(self.config)

        # Mock the projection layers to produce identity mappings for simplicity
        for order, proj_list in self.attention.projections.items():
            for proj in proj_list:
                nn.init.eye_(proj.weight)
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)

        # Set the output projection to identity as well
        nn.init.eye_(self.attention.c_proj.weight)
        if self.attention.c_proj.bias is not None:
            nn.init.zeros_(self.attention.c_proj.bias)

    def test_causal_mask_generation(self):
        """
        Test that the causal mask correctly enforces causality for various attention orders.
        """
        # Test for attention_order=2
        T = 3
        attention_order = 2
        mask = generate_causal_mask(T, attention_order)
        expected_mask = torch.tensor([
            [True, False, False],
            [True, True, False],
            [True, True, True]
        ])
        self.assertTrue(torch.equal(mask, expected_mask), f"attention_order=2 mask incorrect.\nExpected:\n{expected_mask}\nGot:\n{mask}")

        # Test for attention_order=3
        attention_order = 3
        mask = generate_causal_mask(T, attention_order)
        # For attention_order=3, mask[i,j,k] = True iff k <= j <= i
        expected_mask = torch.tensor([
            [
                [True, False, False],
                [False, False, False],
                [False, False, False]
            ],
            [
                [True, False, False],
                [True, True, False],
                [False, False, False]
            ],
            [
                [True, False, False],
                [True, True, False],
                [True, True, True]
            ],
        ])
        self.assertTrue(torch.equal(mask, expected_mask), f"attention_order=3 mask incorrect.\nExpected:\n{expected_mask}\nGot:\n{mask}")

    def test_attention_output_causality_order_3(self):
        """
        Test that for attention_order=3, the output at each position
        exactly matches the corresponding V token, ensuring no future token influence.
        """
        # Configure all heads to have attention_order=3
        self.config.attention_orders = [3, 3]
        self.attention = TrueHigherOrderAttention(self.config)
        for proj_list in self.attention.projections.values():
            for proj in proj_list:
                nn.init.eye_(proj.weight)
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)
        # Set c_proj to identity
        nn.init.eye_(self.attention.c_proj.weight)
        if self.attention.c_proj.bias is not None:
            nn.init.zeros_(self.attention.c_proj.bias)

        # Create a batch with a single sequence: [1, 2, 3]
        B, T, C = 1, 3, 4  # Batch size=1, Sequence length=3, Embedding dim=4
        x = torch.tensor([[[1., 0., 0., 0.],
                           [0., 2., 0., 0.],
                           [0., 0., 3., 0.]]])

        # Run the attention
        output = self.attention(x)

        # Expected output should exactly match V since attention is focused
        expected_output = torch.tensor([[
            [1., 0., 0., 0.],
            [0., 2., 0., 0.],
            [0., 0., 3., 0.]
        ]])

        self.assertTrue(torch.allclose(output, expected_output, atol=1e-4), f"Attention output incorrect for attention_order=3.\nExpected:\n{expected_output}\nGot:\n{output}")

    def test_attention_no_future_dependency_order_3(self):
        """
        Ensure that for attention_order=3, the output at position t does not depend on tokens > t.
        This is done by setting tokens > t to zero and verifying that the output at t remains unaffected.
        """
        # Configure one head with attention_order=3
        self.config.attention_orders = [3, 3]
        self.attention = TrueHigherOrderAttention(self.config)
        for proj_list in self.attention.projections.values():
            for proj in proj_list:
                nn.init.eye_(proj.weight)
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)
        # Set c_proj to identity
        nn.init.eye_(self.attention.c_proj.weight)
        if self.attention.c_proj.bias is not None:
            nn.init.zeros_(self.attention.c_proj.bias)

        # Create a batch with a single sequence: [1, 2, 3]
        B, T, C = 1, 3, 4  # Batch size=1, Sequence length=3, Embedding dim=4
        x = torch.tensor([[[1., 0., 0., 0.],
                           [0., 2., 0., 0.],
                           [0., 0., 3., 0.]]])

        # Run the attention
        output = self.attention(x)

        # Now, create a modified input where token 3 is zeroed out
        x_modified = torch.tensor([[[1., 0., 0., 0.],
                                    [0., 2., 0., 0.],
                                    [0., 0., 0., 0.]]])

        # Run the attention on modified input
        output_modified = self.attention(x_modified)

        # Expected output for modified input:
        expected_output_modified = torch.tensor([[
            [1., 0., 0., 0.],
            [0., 2., 0., 0.],
            [0., 0., 0., 0.]
        ]])

        self.assertTrue(torch.allclose(output_modified, expected_output_modified, atol=1e-4),
                        f"Attention output incorrectly depends on future tokens for attention_order=3.\nExpected:\n{expected_output_modified}\nGot:\n{output_modified}")


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
