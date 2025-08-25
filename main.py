import torch
from src.gpt_dataset import create_dataloader_v1

with open("./data/the-verdict.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

# Create Input-Target pairs
CONTEXT_SIZE = 4 # Length of the input context
# The CONTEXT_SIZE of 4 means that the model is trained to look at a sequence of 4 words (or tokens)
# to predict the next word (token) in the sequence.
# The input sequence (x) consists of the first 4 tokens, while the target sequence (y) consists of the next 4 tokens
# shifted by one.

# Example:
# Input (x): [4, 56, 19, 99]
# Target (y): [56, 19, 99, 801]

# Implement a Data Loader
dataloader = create_dataloader_v1(raw_text, batch_size=CONTEXT_SIZE * 2, max_length=CONTEXT_SIZE, stride=CONTEXT_SIZE, shuffle=False)

# test
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
# print("Targets:\n", targets)
print("\nInputs shape:\n", inputs.shape)
# print("Targets shape:", targets.shape)

# Create token embeddings
input_ids = torch.tensor([2, 3, 5, 7])
vocabulary_size = 50257
output_dimensions = 256

torch.manual_seed(123)
token_embedding_layer = torch.nn.Embedding(vocabulary_size, output_dimensions)
print(token_embedding_layer.weight)

token_embeddings = token_embedding_layer(inputs)

# test
print("\nToken Embeddings:\n", token_embeddings)
print("\nToken Embeddings shape:\n", token_embeddings.shape)

# Create positional embeddings
positional_embedding_layer = torch.nn.Embedding(CONTEXT_SIZE, output_dimensions)
positional_embeddings = positional_embedding_layer(torch.arange(CONTEXT_SIZE))

# test
print("\nPositional Embeddings:\n", positional_embeddings)
print("\nPositional Embeddings shape:\n", positional_embeddings.shape)
