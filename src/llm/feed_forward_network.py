import torch.nn as nn
from llm.gelu import GeLU


# Feed-Forward Network (FFN) used in Transformer blocks
# Consists of two linear layers with a GeLU activation in between
# Expands the embedding dimension to 4 times its size and then projects it back
# This helps in capturing complex patterns in the data
# Reference: "Attention is All You Need" paper
class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(
            config["embedding_dimension"], 4 * config["embedding_dimension"]
        )
        self.activation = GeLU()
        self.linear2 = nn.Linear(
            4 * config["embedding_dimension"], config["embedding_dimension"]
        )
        # self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        # x = self.dropout(x)
        x = self.linear2(x)
        return x
