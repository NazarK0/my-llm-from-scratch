import torch
import tiktoken

from src.llm.config.loader import config_loader
from src.utils.convert import tokenIds_to_text
from src.llm.gpt_model import GPTModel
from src.utils.loss_fn import cross_entropy_loss, perplexity_loss

gpt_config_163m = config_loader("src/llm/config/gpt_163m.json")
print("GPT 163M Config:", gpt_config_163m)
torch.manual_seed(123)
model = GPTModel(gpt_config_163m)
model.eval()  # Set the model to evaluation mode to disable dropout

tokenizer = tiktoken.get_encoding("gpt2")

# Example input (batch of 2 sequences, each of length 3)
inputs = torch.tensor([
    [16833, 3626, 6100], # Every effort moves
    [40, 1107,588]       # I really like
    ])

targets = torch.tensor([
    [3626, 6100, 345],  # effort moves you
    [1107, 588, 11311]  # really like chocolate
    ])


# Cross entropy loss ( must be close to 0.0 since we are using argmax )

# Step by step approach to compute the approximation of cross-entropy loss
# 1. Get the logits from the model
# 2. Convert logits to probabilities using softmax
# 3. Get the probabilities of the target tokens
# 4. Compute the average probability for each token
# 5. Compute negative average log probability as loss
with torch.no_grad():
    logits = model(inputs)

    probabilities = torch.softmax(logits, dim=-1) # Convert logits to probabilities
    print("Probabilities shape:", probabilities.shape)  # Should be (batch_size, sequence_length, vocab_size)

token_ids = torch.argmax(probabilities, dim=-1, keepdim=True)  # Get the token IDs with the highest probability
print("Token Ids:", token_ids)
print(f"Target (batch #1): {tokenIds_to_text(targets[0], tokenizer)}")
print(f"Output (batch #1): {tokenIds_to_text(token_ids[0].flatten(0), tokenizer)}")


text_idx_0 = 0
target_probabilities_0 = probabilities[text_idx_0, [0, 1, 2], targets[text_idx_0]]
print("Target Probabilities idx[0]:", target_probabilities_0)

text_idx_1 = 1
target_probabilities_1 = probabilities[text_idx_1, [0, 1, 2], targets[text_idx_1]]
print("Target Probabilities idx[1]:", target_probabilities_1)

log_probabilities = torch.log(torch.cat((target_probabilities_0, target_probabilities_1)))
print("Log Probabilities:", log_probabilities)

# Compute the average probability for each token
average_log_probability = torch.mean(log_probabilities)
print("Average Log Probability:", average_log_probability.item())

# Compute negative average log probability as loss
negative_average_log_probability = -average_log_probability
print("Negative Average Log Probability (Loss):", negative_average_log_probability.item())


# Using the implemented loss functions
loss = cross_entropy_loss(logits, targets)
print("Cross Entropy Loss:", loss)

perplexity = perplexity_loss(logits, targets)
print("Perplexity:", perplexity)
