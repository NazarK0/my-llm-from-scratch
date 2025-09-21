import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    def __init__(self, embedding_dimension):
        super(LayerNormalization, self).__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embedding_dimension))
        self.shift = nn.Parameter(torch.zeros(embedding_dimension))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)

        return self.scale * x_normalized + self.shift
