import torch
import torch.nn as nn

# A simplified implementation of self-attention mechanism
# using a small set of 6 words and 3-dimensional embeddings.
# This example demonstrates the core computations of self-attention without trainable weights.
words = ["Your", "journey", "starts", "with", "one", "step"]
inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your
        [0.55, 0.87, 0.66],  # journey
        [0.57, 0.85, 0.64],  # starts
        [0.91, 0.12, 0.34],  # with
        [0.45, 0.72, 0.11],  # one
        [0.34, 0.23, 0.78],  # step
    ]
)


# 1. Convert inputs to queries, keys, and values
x_2 = inputs[1] # A (the second word "journey")
d_in = inputs.shape[1] # B (the embedding dimension, 3 in this case)
d_out = 2 # C (the output dimension, typically larger than the input dimension, or the same as in GPT models)

torch.manual_seed(123)
w_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) # (B, C)
w_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) # (B, C)
w_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) # (B, C)

# test
print("Query, Key, Value weights:")
print("w_query:", w_query)
print("w_key:", w_key)
print("w_value:", w_value)


keys = inputs @ w_key  # (6, 3) @ (3, 2) = (6, 2)
queries = inputs @ w_query  # (6, 3) @ (3, 2) = (6, 2)
values = inputs @ w_value  # (6, 3) @ (3, 2) = (6, 2)

# 2. Compute attention scores (dot products)
attention_scores = queries @ keys.T  # (6, 2) @ (2, 6) = (6, 6) (Omega)

# 3. Compute attention weights (softmax)
## Scale the attention scores by the square root of the dimension of the keys
## to prevent large dot product values that could push the softmax into regions with very small gradients
## This is especially important when the dimension is large.
## In practice, this helps stabilize the gradients during training.
attention_scores = attention_scores / torch.sqrt(torch.tensor(d_out, dtype=torch.float32))
attention_weights = torch.softmax(attention_scores, dim=-1)

# 4. Compute context vectors (weighted sums)
# (6x6 @ 6x2 = 6x2)
context = attention_weights @ values

# Implement a class SelfAttention to encapsulate the above logic
## Derive class from nn.Module
class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out):
        super(SelfAttention, self).__init__()
        self.w_query = nn.Parameter(torch.rand(d_in, d_out))
        self.w_key = nn.Parameter(torch.rand(d_in, d_out))
        self.w_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, inputs):
        # 1. Convert inputs to queries, keys, and values
        keys = inputs @ self.w_key
        queries = inputs @ self.w_query
        values = inputs @ self.w_value

        # 2. Compute attention scores (dot products)
        attention_scores = queries @ keys.T

        # 3. Compute attention weights (softmax)
        attention_scores = attention_scores / torch.sqrt(torch.tensor(d_out, dtype=torch.float32))
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # 4. Compute context vectors (weighted sums)
        context = attention_weights @ values
        return context

# test
torch.manual_seed(789)
self_attention = SelfAttention(d_in, d_out)
context_v1 = self_attention.forward(inputs)
print("Context Vectors from SelfAttention class:")
print(context_v1)


# Implement a class SelfAttention using nn.Linear
## for more stable and effective model training
class SelfAttention2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super(SelfAttention2, self).__init__()
        self.weight_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.weight_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.weight_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, inputs):
        # 1. Convert inputs to queries, keys, and values
        ## Keys is analogous to the index in a database.
        ## It represents all tokens we can attend to.
        ## In attention mechanism each token has a key.
        ## Keys used to match against the query.
        ##
        ## Queries is analogous to the search query in a database.
        ## It represents the current token we are focusing on.
        ##
        ## Values is analogous to the actual data in a database.
        ## It represents the actual content representation of the input items.
        ## Values are what we ultimately want to retrieve based on the attention scores.
        ## The attention mechanism computes a weighted sum of the values,
        ## where the weights are determined by the similarity between the queries and keys.
        keys = self.weight_key(inputs)
        queries = self.weight_query(inputs)
        values = self.weight_value(inputs)

        # 2. Compute attention scores (dot products)
        attention_scores = queries @ keys.T

        # 3. Compute attention weights (softmax)
        attention_scores = attention_scores / torch.sqrt(
            torch.tensor(d_out, dtype=torch.float32)
        )
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # 4. Compute context vectors (weighted sums)
        context = attention_weights @ values
        return context

# test
torch.manual_seed(789)
self_attention2 = SelfAttention2(d_in, d_out)
context_v2 = self_attention2.forward(inputs)
print("Context Vectors from SelfAttention2 class:")
print(context_v2)
