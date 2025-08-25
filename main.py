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
print("Targets:\n", targets)
