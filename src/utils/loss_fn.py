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

def batch_loss(input_batch, target_batch, model, device):
    """
    Compute the loss for a batch of input and target sequences.

    Args:
        input_batch: Batch of input token IDs (shape: batch_size x sequence_length).
        target_batch: Batch of target token IDs (shape: batch_size x sequence_length).
        model: The language model to evaluate.
        device: The device to run the computations on (e.g., 'cpu' or 'cuda').
    Returns:
        Loss value for the batch.
    """  
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    logits = model(input_batch)
    loss  = loss_fn(logits.flatten(0, 1), target_batch.flatten())

    return loss

def loader_loss(dataloader, model, device, max_batches=None):
    """
    Compute the average loss over a DataLoader.

    Args:
        dataloader: DataLoader providing batches of input and target sequences.
        model: The language model to evaluate.
        device: The device to run the computations on (e.g., 'cpu' or 'cuda').
        max_batches: Optional maximum number of batches to process (for quick evaluation).
    Returns:
        Average loss value over the DataLoader.
    """
    
    total_loss = 0.0

    if len(dataloader) == 0:
        return float('nan')
    elif max_batches is None:
        max_batches = len(dataloader)
    else:
        # Reduce the number of batches to match the total number of batches in the dataloader
        # If max_batches is greater than the total number of batches
        max_batches = min(max_batches, len(dataloader))
        
    for batch_idx, (input_batch, target_batch) in enumerate(dataloader):
        if batch_idx < max_batches:
            loss = batch_loss(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
        
    average_loss = total_loss / max_batches
    
    return average_loss