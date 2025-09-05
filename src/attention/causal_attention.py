from torch import tensor, stack, ones, manual_seed, sqrt, softmax, inf, tril, triu, float32
from torch.nn import Module, Linear, Dropout
from self_attention import SelfAttention

DROPOUT_RATE = 0.5  # 50% dropout


# A Casual attention implementation with detailed comments and a simple tests
# This example uses fixed input embeddings for clarity,
# and focuses on the causal attention mechanism. It illustrates the concept clearly
# and show dropout application on attention weights.
# We are using a small set of 6 words and 3-dimensional embeddings.
# This example demonstrates the core computations of self-attention without trainable weights.
words = ["Your", "journey", "starts", "with", "one", "step"]
inputs = tensor(
    [
        [0.43, 0.15, 0.89],  # Your
        [0.55, 0.87, 0.66],  # journey
        [0.57, 0.85, 0.64],  # starts
        [0.22, 0.58, 0.33],  # with
        [0.77, 0.25, 0.10],  # one
        [0.05, 0.80, 0.55],  # step
    ]
)


# Dimensions
# the embedding dimension, 3 in this case
d_in = inputs.shape[1]
# the output dimension, typically larger than the input dimension,
# or the same as in GPT models, here we use 2 for simplicity
d_out = 2


manual_seed(789)
self_attention = SelfAttention(d_in, d_out)
context = self_attention(inputs)  # same as self_attention.forward(inputs)

queries = self_attention.weight_query(inputs)
keys = self_attention.weight_key(inputs)
attention_scores = queries @ keys.T
attention_weights = softmax(
    attention_scores / sqrt(tensor(keys.shape[-1], dtype=float32)),
    dim=1,
)
# test
print("Attention Weights:")
print(attention_weights)

# Simple Causal Masking
## 1.Create the mask where the values above the main diagonal are zeroes
context_size = inputs.shape[0]
causal_mask = tril(ones((context_size, context_size)))
# test
print("Causal Mask:")
print(causal_mask)

## 2. Apply the causal mask to the attention weights
causal_weights = attention_weights * causal_mask
# test
print("Causal Weights (after applying the mask):")
print(causal_weights)

## 3. Re-normalize the masked weights so that they sum to 1
row_sums = causal_weights.sum(dim=1, keepdim=True)
### To avoid division by zero, we add a small epsilon where row_sums are zero
epsilon = 1e-10
row_sums = row_sums + (row_sums == 0).float() * epsilon
causal_weights = causal_weights / row_sums
# test
print("Causal Weights (after re-normalization):")
print(causal_weights)

# More efficient implementation of Causal Self-Attention
mask = triu(ones(context_size, context_size), diagonal=1)
scores = attention_scores.masked_fill(mask.bool(), -inf)
# test
print("Masked Scores:")
print(scores)

weights = softmax(scores / sqrt(tensor(keys.shape[-1], dtype=float32)), dim=1)
# test
print("Causal Weights (efficient implementation):")
print(weights)

# Implement dropout on the attention weights
dropout = Dropout(DROPOUT_RATE)
example = ones(6, 6)
# test
print("Original Weights:")
print(example)

# To compensate for the dropped weights during training,
# we scale the remaining weights by 1/(1 - dropout_rate).
# This ensures that the expected sum of the weights remains the same.
# For a dropout rate of 0.5, we scale by 2.0 (1/(1-0.5) = 2).
dropped_example = dropout(example)
# test
print("Dropped Weights (50% dropout):")
print(dropped_example)


# Implement Casual Self-Attention class with Dropout
class CausalAttention(Module):
    def __init__(self, d_in, d_out, context_length, dropout_rate, qkv_bias=False):
        super(CausalAttention, self).__init__()
        mask_size = (context_length, context_length)

        self.weight_query = Linear(d_in, d_out, bias=qkv_bias)
        self.weight_key = Linear(d_in, d_out, bias=qkv_bias)
        self.weight_value = Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = Dropout(dropout_rate)
        # Buffer is automatically moved to the appropriate device with the model (cpu or gpu)
        # It is avoiding device mismatch errors during training and inference.
        self.register_buffer("causal_mask", triu(ones(mask_size), diagonal=1))

    def forward(self, inputs):
        b, num_tokens, d_in = inputs.shape
        keys = self.weight_key(inputs)
        queries = self.weight_query(inputs)
        values = self.weight_value(inputs)

        attention_scores = queries @ keys.transpose(1, 2)
        # :num_tokens to account for cases where the input length is less than context_length
        # (e.g., during inference with shorter sequences)
        attention_scores = attention_scores.masked_fill_(self.causal_mask.bool()[:num_tokens, :num_tokens], -inf)
        attention_scores /= sqrt(tensor(keys.shape[-1], dtype=float32))

        attention_weights = softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = attention_weights @ values
        return context

    manual_seed(123)

# test
# (2, 6, 3) # 2 input texts with 6 words each and 3-dimensional embeddings
batch = stack((inputs, inputs), dim=0)  
print("Batch shape:", batch.shape)
context_length = batch.shape[1]  # 6
print("d_in:", d_in, "d_out:", d_out, "context_length:", context_length)
causal_attention = CausalAttention(d_in, d_out, context_length, 0.0)

context = causal_attention(batch)
print("Causal Attention Output shape:", context.shape)
print("Causal Attention Output:")
print(context)
