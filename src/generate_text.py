import torch

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

def generate_text(model, idx, max_new_tokens, context_size, temperature=1.0,  top_k=None, eos_id=None):
    """
    Generate text using temperature scaling and top-k sampling.
    
    Workflow: Obtain logits from the model, apply temperature scaling,
        optionally filter with top-k sampling, convert to probabilities, and sample the next token.
        Repeat for the specified number of new tokens.
    
    Args:
        model: The language model to use for generation.
        idx: Initial input tokens (tensor of shape (batch_size, sequence_length)).
        max_new_tokens: Number of new tokens to generate.
        context_size: Maximum context size for the model.
        temperature: Temperature for scaling logits. Higher values increase randomness.
        top_k: If specified, restrict sampling to the top_k most probable tokens.
        eos_id: If specified, stop generation when this token ID is generated.
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
        logits = logits[:, -1, :]  # Shape: (batch_size, vocabulary_size)

        # Filter logits using top-k sampling if specified
        # Top-k sampling helps to limit the next token choices to the k most probable tokens
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_value = top_logits[:,-1]
            
            # Focus on the last time step's logits
            logits = logits[:, -1] 
            logits = torch.where(logits < min_value, torch.tensor(float('-inf')).to(logits.device), logits)
        
        # Apply temperature scaling
        # Temperature scaling adjusts the "confidence" of the model's predictions
        # Higher temperatures lead to more random samples, while lower temperatures make the model more confident
        if temperature > 0:
            logits = logits / temperature
            probabilities = torch.softmax(logits, dim=-1)  # Shape: (batch_size, vocabulary_size)
            # Sample the next token from the probability distribution
            # Multinomial function behaves differently than argmax
            # it samples from the distribution defined by the probabilities
            next_token = torch.multinomial(probabilities, num_samples=1)  # Shape: (batch_size, 1)
        else:
            # Otherwise, use greedy sampling
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # Shape: (batch_size, 1)


        # If eos_id is specified and any generated token matches it, stop generation for that batch item
        if  next_token == eos_id:
            break
        
        # Append the sampled token to the input sequence
        idx = torch.cat((idx, next_token), dim=1)  # Shape: (batch_size, sequence_length + 1)

    return idx
