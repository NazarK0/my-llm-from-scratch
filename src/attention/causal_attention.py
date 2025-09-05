from torch import tensor, ones, manual_seed, sqrt, softmax, inf, triu, float32
from torch.nn import Module, Linear, Dropout

DROPOUT_RATE = 0.5  # 50% dropout


# A Causal attention implementation with dropout
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

