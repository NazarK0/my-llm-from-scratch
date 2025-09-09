from torch import (
    tensor,
    cat,
    stack,
    manual_seed,
    triu,
    ones,
    sqrt,
    softmax,
    inf,
    float32,
)
from torch.nn import Module, Linear, Dropout
from causal_attention import CausalAttention
from torch.nn import ModuleList


# A Multi-Head Attention implementation with dropout
# This class creates multiple instances of CausalAttention and concatenates their outputs sequentially.
# Each head processes the same input independently, allowing the model to capture diverse aspects of the input data.
class MultiHeadAttentionSimple(Module):
    def __init__(self, d_in, d_out, context_length, heads, dropout_rate=0.0, qkv_bias=False):
        super(MultiHeadAttentionSimple, self).__init__()
        assert (
            d_out % heads == 0
        ), "d_out must be divisible by heads"

        self.heads = ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout_rate, qkv_bias) for _ in range(heads)]
        )

    # Final output is the concatenation of all head outputs
    def forward(self, inputs):
        return cat([head(inputs) for head in self.heads], dim=-1)
    
    
class MultiHeadAttention(Module):
    def __init__(self, d_in, d_out, context_length, heads, dropout_rate=0.0, qkv_bias=False):
        super(MultiHeadAttention, self).__init__()
        assert (
            d_out % heads == 0
        ), "d_out must be divisible by heads"

        self.dimension_out = d_out 
        self.heads = heads
        self.head_dimension = d_out // heads
        
        self.weight_query = Linear(d_in, d_out, bias=qkv_bias) # Wq
        self.weight_key = Linear(d_in, d_out, bias=qkv_bias)   # Wk
        self.weight_value = Linear(d_in, d_out, bias=qkv_bias) # Wv
        self.out_projection = Linear(d_out, d_out) # Wo
        self.dropout = Dropout(dropout_rate)
        self.register_buffer("mask", triu(ones((context_length, context_length)), diagonal=1))

    # Final output is the concatenation of all head outputs
    def forward(self, input):
        batch_size, context_length, _ = input.shape # context_length is number_of_tokens

        keys = self.weight_key(input)
        queries = self.weight_query(input)
        values = self.weight_value(input)

        # 1. Implicitly split the matrix by adding a `num_heads` dimension
        # Unroll the last dimension: (b, context_length, d_out) -> (b, context_length, heads, head_dimension)
        keys = keys.view(batch_size, context_length, self.heads, self.head_dimension)
        queries = queries.view(batch_size, context_length, self.heads, self.head_dimension)
        values = values.view(batch_size, context_length, self.heads, self.head_dimension)

        # 2.Transpose: (batch_size, context_length, heads, head_dimension) -> (batch_size, heads, context_length, head_dimension)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 3. Compute scaled dot-product attention with causal mask for each head
        attention_scores = queries @ keys.transpose(2, 3) # (batch_size, heads, context_length, context_length)

        # 4. Original mask truncated to the context length and converted to boolean
        mask_boolean = self.mask.bool()[:context_length, :context_length]

        # 5. Apply the causal mask to the attention scores
        attention_scores.masked_fill_(mask_boolean, -inf) # CHECK THIS LINE
        attention_weights = softmax(attention_scores / sqrt(tensor(self.head_dimension, dtype=float32)), dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 6. Shape: (batch_size, context_length, heads, head_dimension)
        context = (attention_weights @ values).transpose(1, 2)

        # 7. Combine the heads where  self.dimension_out = heads * head_dimension
        context = context.contiguous().view(batch_size, context_length, self.dimension_out)
        context = self.out_projection(context)

        return context


# Test the Multi-Head Attention implementation
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

batch = stack((inputs, inputs), dim=0)  # Create a batch of 2 identical sequences

manual_seed(123)  # For reproducibility
context_length = batch.shape[1]  # 6 tokens in the input sequence

attention  =  MultiHeadAttentionSimple(d_in, d_out, context_length, heads=2)
context = attention(batch)
print("Input shape:", batch.shape)  # (2, 6, 3)
print("Context shape:", context.shape)  # (2, 6, 4)
print("Context:\n", context)



# Test2

manual_seed(123)  # For reproducibility
inputs2 = tensor(
    [[0.43, 0.15, 0.89, 0.55, 0.87, 0.66],
     [0.57, 0.85, 0.64, 0.22, 0.58, 0.33],
     [0.77, 0.25, 0.10, 0.05, 0.80, 0.55]]
)
batch2 = stack((inputs2, inputs2), dim=0)

batch_size, context_length, d_in = batch2.shape
D_OUT = 6
HEADS = 2
attention2 = MultiHeadAttention(d_in, D_OUT, context_length, heads=HEADS)
context2 = attention2(batch2)
print("Input2 shape:", batch2.shape)  # (2, 3, 6)
print("Context2 shape:", context2.shape)  # (2, 3, 6)
print("Context2:\n", context2)  