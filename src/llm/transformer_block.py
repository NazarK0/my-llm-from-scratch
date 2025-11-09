import torch.nn as nn

from src.llm.multi_head_attention import MultiHeadAttention
from src.llm.feed_forward_network import FeedForwardNetwork
from src.llm.layer_normalization import LayerNormalization


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadAttention(
            d_in=config["embedding_dimension"],
            d_out=config["embedding_dimension"],
            context_length=config["context_length"],
            heads=config["number_of_heads"],
            dropout_rate=config["dropout_rate"],
            qkv_bias=config["qkv_bias"]
        )
        
        self.feed_forward_network = FeedForwardNetwork(config)
        self.layer_normalization1 = LayerNormalization(config["embedding_dimension"])
        self.layer_normalization2 = LayerNormalization(config["embedding_dimension"])
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.layer_normalization1(x)
        x = self.attention(x) # Shape: (batch_size, sequence_length, embedding_dimension)
        x = self.dropout(x)
        x = x + shortcut  # Residual connection
        
        # Shortcut connection for feed-forward network
        shortcut = x
        x = self.layer_normalization2(x)
        x = self.feed_forward_network(x)
        x = self.dropout(x)
        x = x + shortcut  # Residual connection

        return x
