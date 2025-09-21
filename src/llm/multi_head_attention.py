from torch import (
    tensor,
    triu,
    ones,
    sqrt,
    softmax,
    inf,
    float32,
)
from torch.nn import Module, Linear, Dropout


# A Multi-Head Attention implementation with dropout
# This class creates multiple instances of CausalAttention and concatenates their outputs in parallel.
# Each head processes the same input independently, allowing the model to capture diverse aspects of the input data.

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
