import torch

# A simplified implementation of self-attention mechanism
# using a small set of 6 words and 3-dimensional embeddings.
# This example demonstrates the core computations of self-attention without trainable weights.
words = ["Your", "journey", "starts", "with", "one", "step"]
inputs = torch.tensor([
    [0.43, 0.15, 0.89], # Your
    [0.55, 0.87, 0.66], # journey
    [0.57, 0.85, 0.64], # starts
    [0.91, 0.12, 0.34], # with
    [0.45, 0.72, 0.11], # one
    [0.34, 0.23, 0.78], # step
])


# 1. Compute attention scores (dot products)
attention_scores = torch.empty(6, 6)
attention_scores = inputs @ inputs.T

# 2. Compute attention weights (softmax)
attention_weights = torch.softmax(attention_scores, dim=-1)

# 3. Compute context vectors (weighted sums)
# (6x6 @ 6x3 = 6x3)
context = attention_weights @ inputs # The same as torch.matmul(attention_weights, inputs) 
