import torch
import torch.nn as nn


# Implement a class SelfAttention
class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super(SelfAttention, self).__init__()
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
        attention_scores = attention_scores / torch.sqrt(torch.tensor(keys.shape[-1], dtype=torch.float32))
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # 4. Compute context vectors (weighted sums)
        context = attention_weights @ values
        return context
