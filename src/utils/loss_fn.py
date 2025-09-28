# Loss function provides a way to measure how well the model's predictions align with the actual target values.
# It quantifies the difference between predicted outputs and true outputs, guiding the optimization process during training.
import torch


def cross_entropy_loss(logits, targets):
    """
    Compute the cross-entropy loss between logits and targets.

    Args:
        logits: Predicted logits from the model (shape: batch_size x sequence_length x vocab_size).
        targets: True target token IDs (shape: batch_size x sequence_length).
    Returns:
        Cross-entropy loss value.
    """
    loss_fn = torch.nn.CrossEntropyLoss()
    # Reshape logits and targets to match the expected input shape for CrossEntropyLoss
    logits_reshaped = logits.view(-1, logits.size(-1))  # Shape: (batch_size * sequence_length, vocab_size)
    targets_reshaped = targets.view(-1)  # Shape: (batch_size * sequence_length)
    loss = loss_fn(logits_reshaped, targets_reshaped)
    
    return loss.item()

def perplexity_loss(logits, targets):
    """
        Compute the perplexity loss between logits and targets.
        It is more interpretable to use perplexity loss.
        It shows how many tokens are possible in average.
        For example, a perplexity of 10 means that on average
        the model is choosing between 10 possible tokens at each step

        Args:
            logits: Predicted logits from the model (shape: batch_size x sequence_length x vocab_size).
            targets: True target token IDs (shape: batch_size x sequence_length).
        Returns:
            Perplexity loss value.
    """
    loss_fn = torch.nn.CrossEntropyLoss()
    # Reshape logits and targets to match the expected input shape for CrossEntropyLoss
    logits_reshaped = logits.view(-1, logits.size(-1))  # Shape: (batch_size * sequence_length, vocab_size)
    targets_reshaped = targets.view(-1)  # Shape: (batch_size * sequence_length)
    loss = loss_fn(logits_reshaped, targets_reshaped)
    perplexity = torch.exp(loss)

    return perplexity.item()
