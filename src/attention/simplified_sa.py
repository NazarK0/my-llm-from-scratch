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

## Naive implementation. For loops very slow for large tensors
# for i, x_i in enumerate(inputs):
#     for j, x_j in enumerate(inputs):
#         attention_scores[i, j] = torch.dot(x_i, x_j)
## test
# print("Attention Scores For loop:")
# print(attention_scores)

## Matrix multiplication implementation
# attention_scores = torch.matmul(inputs, inputs.T)
## test
# print("Attention Scores Matrix:")
# print(attention_scores)

## Matrix multiplication syntactic sugar
attention_scores = inputs @ inputs.T

# test
print("Attention Scores Syntactic Sugar:")
print(attention_scores)

# 2. Compute attention weights (softmax)
attention_weights = torch.softmax(attention_scores, dim=-1)

# test
print("Attention Weights:")
print(attention_weights)
print("Sum of Attention Weights (should be 1.0 for each row):")
print(attention_weights.sum(dim=-1))

# 3. Compute context vectors (weighted sums)
# (6x6 @ 6x3 = 6x3)
context = attention_weights @ inputs # The same as torch.matmul(attention_weights, inputs) 

print("Context Vectors:")
print(context)
