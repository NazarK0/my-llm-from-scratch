import torch
from src.gpt_dataset import create_dataloader_v1


# The CONTEXT_SIZE of 4 means that the model is trained to look at a sequence of 4 words (or tokens)
# to predict the next word (token) in the sequence.
# The input sequence (x) consists of the first 4 tokens, while the target sequence (y) consists of the next 4 tokens
# shifted by one.
#
# Example:
# Input (x): [4, 56, 19, 99]
# Target (y): [56, 19, 99, 801]
CONTEXT_SIZE = 4
VOCABULARY_SIZE = 50257
EMBEDDING_DIMENSIONS = 256
EMBEDDINGS_SEED = 123

# Load raw text data
with open("./data/the-verdict.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

# Data preparation
## Implement a Data Loader
dataloader = create_dataloader_v1(raw_text, batch_size=CONTEXT_SIZE * 2, max_length=CONTEXT_SIZE, stride=CONTEXT_SIZE, shuffle=False)

## Create token embeddings
torch.manual_seed(EMBEDDINGS_SEED)
data_iter = iter(dataloader)  # Convert to iterator
inputs, targets = next(data_iter)

token_embedding_layer = torch.nn.Embedding(VOCABULARY_SIZE, EMBEDDING_DIMENSIONS)
token_embeddings = token_embedding_layer(inputs)

## Create positional embeddings
positional_embedding_layer = torch.nn.Embedding(CONTEXT_SIZE, EMBEDDING_DIMENSIONS)
positional_embeddings = positional_embedding_layer(torch.arange(CONTEXT_SIZE))

## Input embeddings
input_embeddings = token_embeddings + positional_embeddings
