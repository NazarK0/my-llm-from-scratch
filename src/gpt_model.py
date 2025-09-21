import torch
import torch.nn as nn
from llm.layer_normalization import LayerNormalization
from transformer_block import TransformerBlock


GPT_CONFIG_163M = {
    "vocabulary_size": 50257,
    "context_length": 1024,
    "embedding_dimension": 768,
    "number_of_layers": 12,
    "number_of_heads": 12,
    "dropout_rate": 0.1,
    "qkv_bias": False,
}

# GPT-2 architecture model
class GPTModel(nn.Module):
    def __init__(self, config):
        super(GPTModel, self).__init__()
        self.token_embedding = nn.Embedding(
            config["vocabulary_size"], config["embedding_dimension"]
        )
        self.position_embedding = nn.Embedding(
            config["context_length"], config["embedding_dimension"]
        )
        self.dropout = nn.Dropout(config["dropout_rate"])

        # Placeholder for transformer block
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["number_of_layers"])]
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
        # token_embeddings + position_embeddings = input_embeddings
        x = self.dropout(token_embeddings + position_embeddings)

        # Pass through transformer blocks
        x = self.transformer_blocks(x)

        # Final normalization and output head
        x = self.final_normalization(x)
        logits = self.out_head(x)

        return logits
    


torch.manual_seed(123)
batch = torch.tensor([
    [6109, 3626, 6100, 345],
    [6109, 1110, 6622, 257]
])
model_163M = GPTModel(GPT_CONFIG_163M)
out = model_163M(batch)
total_params = sum(p.numel() for p in model_163M.parameters() if p.requires_grad)
total_size_bytes = total_params * 4  # Assuming 4 bytes per parameter (float32)
total_size_mb = total_size_bytes / (1024 ** 2)
print("Input batch:\n", batch)
print("\nOptput shape:", out.shape)  # Should be (2, 4, 50257)
print("Output logits:\n", out)
print(f"\nTotal trainable parameters: {total_params:,}")
print(f"Total size in memory (MB): {total_size_mb:.2f} MB")
