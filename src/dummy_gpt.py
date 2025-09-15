import torch
import torch.nn as nn
import matplotlib.pyplot as plt


GPT_CONFIG_124M = {
    "vocabulary_size": 50257,
    "context_length": 1024,
    "embedding_dimension": 768,
    "number_of_layers": 12,
    "number_of_heads": 12,
    "dropout_rate": 0.1,
    "qkv_bias": False,
}


class DummyGPTModel:
    def __init__(self, config):
        self.token_embedding = nn.Embedding(
            config["vocabulary_size"], config["embedding_dimension"]
        )
        self.position_embedding = nn.Embedding(
            config["context_length"], config["embedding_dimension"]
        )
        self.dropout = nn.Dropout(config["dropout_rate"])

        # Placeholder for transformer block
        self.transformer_blocks = nn.Sequential(
            *[DummyTransformerBlock(config) for _ in range(config["number_of_layers"])]
        )

        # Placeholder for layer normalization
        self.final_normalization = LayerNormalization(config["embedding_dimension"])
        self.out_head = nn.Linear(
            config["embedding_dimension"], config["vocabulary_size"], bias=False
        )

    def forward(self, input_tokens):
        batch_size, sequence_length = input_tokens.shape

        # Token and position embeddings
        token_embeddings = self.token_embedding(input_tokens)
        position_indices = torch.arange(sequence_length, device=input_tokens.device)
        position_embeddings = self.position_embedding(position_indices)

        # Combine embeddings and apply dropout
        x = self.dropout(token_embeddings + position_embeddings)

        # Pass through transformer blocks
        x = self.transformer_blocks(x)

        # Final normalization and output head
        x = self.final_normalization(x)
        logits = self.out_head(x)

        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, config):
        super(DummyTransformerBlock, self).__init__()

    def forward(self, x):
        return x


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
        
        
# Feed-Forward Network (FFN) used in Transformer blocks
# Consists of two linear layers with a GeLU activation in between
# Expands the embedding dimension to 4 times its size and then projects it back
# This helps in capturing complex patterns in the data
# Reference: "Attention is All You Need" paper
class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(config["embedding_dimension"], 4 * config["embedding_dimension"])
        self.activation = GeLU()
        self.linear2 = nn.Linear(4 * config["embedding_dimension"], config["embedding_dimension"])
        # self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        # x = self.dropout(x)
        x = self.linear2(x)
        return x


# Example
torch.manual_seed(123)
batch_example = torch.randn(2, 5)
# ReLU function convert negative values to zero
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
output = layer(batch_example)
print(output)

mean = output.mean(dim=-1, keepdim=True)
variance = output.var(dim=-1, keepdim=True)
print("Mean:", mean)
print("Variance:", variance)
output_normalized = (output - mean) / torch.sqrt(variance)
mean_normalized = output_normalized.mean(dim=-1, keepdim=True)
variance_normalized = output_normalized.var(dim=-1, keepdim=True)
print("Output after normalization:", output_normalized)
print("Mean after normalization:", mean_normalized)
print("Variance after normalization:", variance_normalized)

torch.set_printoptions(sci_mode=False)
print("Mean after normalization:", mean_normalized)
print("Variance after normalization:", variance_normalized)

ln = LayerNormalization(embedding_dimension=5)
output_ln = ln(batch_example)
mean_ln = output_ln.mean(dim=-1, keepdim=True)
variance_ln = output_ln.var(dim=-1, keepdim=True, unbiased=False)
print("Output after LayerNorm:", output_ln)
print("Mean after LayerNorm:", mean_ln)
print("Variance after LayerNorm:", variance_ln)


# Compare Gelu vs ReLU
gelu, relu = GeLU(), nn.ReLU()
x = torch.linspace(-4, 4, steps=100)
y_gelu, y_relu = gelu(x), relu(x)

plt.figure(figsize=(10, 5))
for i, (y, name) in enumerate(zip([y_gelu, y_relu], ["GeLU", "ReLU"]), start=1):
    plt.subplot(1, 2, i)
    plt.plot(x.numpy(), y.numpy())
    plt.title(f"{name} activation function")
    plt.xlabel("Input")
    plt.ylabel(f"{name}(Input)")
    plt.grid(True)
plt.tight_layout()
plt.show()

# Test FeedForwardNetwork
ffn = FeedForwardNetwork(GPT_CONFIG_124M)
x = torch.randn(2, 3, GPT_CONFIG_124M["embedding_dimension"])
output = ffn(x)
print("FeedForwardNetwork output shape:", output.shape)


# Test shortcut connections
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super(ExampleDeepNeuralNetwork, self).__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GeLU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GeLU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GeLU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GeLU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GeLU()),
            
        ])

    def forward(self, x):
        for layer in self.layers:
            # Compute the output of the current layer
            layer_output = layer(x)
            # If using shortcut connections, add the input to the output
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
                
        return x
    
def print_gradients(model, x):
    # Forward pass
    output = model(x)
    target = torch.tensor([[0.0]])
    
    # Calculate loss based on how close the target
    # and output are
    loss = nn.MSELoss()
    loss_value = loss(output, target)
    
    # Backward pass to calculate the gradients
    loss_value.backward()
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")



layer_sizes = [3,3,3,3,3,1]
sample_input = torch.tensor([[1.0, 0.0, -1.0]])
torch.manual_seed(123)

# Show the effect of vanishing gradients
# without shortcut connections
# The gradients will be very small
# with shortcut connections
# The gradients will be larger and more stable
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)
print("Output without shortcut connections:")
print_gradients(model_without_shortcut, sample_input)

torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
print("Output with shortcut connections:")
print_gradients(model_with_shortcut, sample_input)