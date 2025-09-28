import torch

def text_to_tokedIds(text, tokenizer):
    """
    Convert text to token IDs using the specified tokenizer.

    Args:
        text: Input text string.
        tokenizer: The tokenizer to use.

    Returns:
        List of token IDs.
    """
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # Shape: (1, sequence_length)

    return encoded_tensor

def tokenIds_to_text(token_ids, tokenizer):
    """
    Convert token IDs back to text using the specified tokenizer.
    
    Args:
        token_ids: List or tensor of token IDs.
        tokenizer: The tokenizer to use.
        
    Returns:
        Decoded text string.
    """
    flat = token_ids.squeeze(0)
    decoded_text = tokenizer.decode(flat.tolist())

    return decoded_text
