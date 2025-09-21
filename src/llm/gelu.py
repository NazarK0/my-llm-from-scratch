import torch
import torch.nn as nn


# GeLU activation function
# Gaussian Error Linear Unit
# Reference: https://arxiv.org/abs/1606.08415
# It is smoother than ReLU and can improve model performance in some cases.
class GeLU(nn.Module):
    def __init__(self):
        super(GeLU, self).__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )
