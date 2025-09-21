import torch
import tiktoken

from gpt_model import GPTModel, GPT_CONFIG_163M


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Generate text using a simple loop, one token at a time.
    
    Args:
        model: The language model to use for generation.
        idx: Initial input tokens (tensor of shape (batch_size, sequence_length)).
        max_new_tokens: Number of new tokens to generate.
        context_size: Maximum context size for the model.
    Returns:
        Generated tokens (tensor of shape (batch_size, sequence_length + max_new_tokens)).
    """
    for _ in range(max_new_tokens):
        # Ensure the input does not exceed the model's context size
        # E.g., if context_size=1024 and idx has 1050 tokens, we only take the last 1024
        idx_cond = idx[:, -context_size:]

        # Get the model's predictions
        with torch.no_grad():  # Disable gradient calculation for inference
            logits = model(idx_cond)  # Shape: (batch_size, sequence_length, vocabulary_size)

        # Focus on the last time step's logits
        logits = logits[:, -1, :]  # Shape: (batch_size, vocabulary_size)

        # Convert logits to probabilities
        probabilities = torch.softmax(logits, dim=-1)  # Shape: (batch_size, vocabulary_size)

        # Sample the next token from the probability distribution
        next_token = torch.argmax(probabilities, dim=-1, keepdim=True)  # Shape: (batch_size, 1)

        # Append the sampled token to the input sequence
        idx = torch.cat((idx, next_token), dim=1)  # Shape: (batch_size, sequence_length + 1)

    return idx

# Example usage:
tokenizer = tiktoken.get_encoding("gpt2")
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # Shape: (1, sequence_length)
print("Encoded:", encoded)
print("Encoded tensor shape:", encoded_tensor.shape)

model_163M = GPTModel(GPT_CONFIG_163M)
model_163M.eval()  # Set the model to evaluation mode
out = generate_text_simple(
    model_163M, encoded_tensor, 
    max_new_tokens=6, 
    context_size=GPT_CONFIG_163M["context_length"]
    )
print("Output:", out)
print("Output length:", len(out[0]))

decoded_text = tokenizer.decode(out[0].tolist())
print("Decoded text:", decoded_text)