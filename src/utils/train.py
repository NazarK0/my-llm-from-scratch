import torch
from src.generate_text import generate_text
from src.utils.convert import text_to_tokenIds, tokenIds_to_text
from src.utils.loss_fn import batch_loss, loader_loss


def train_model_simple(model, train_loader, validation_loader, 
                        optimizer, device, num_epochs, evaluation_frequency, 
                        evaluation_steps, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses = []
    validation_losses = []
    track_tokens_seen = []
    tokens_seen = 0
    global_step = -1
    
    # Main training loop
    for epoch in range(num_epochs):
        model.train() # Set the model to training mode
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Clear previous gradients
            loss = batch_loss(input_batch, target_batch, model, device)
            loss.backward()  # Backpropagation, calculate loss gradients
            optimizer.step()  # Update model parameters
            tokens_seen += input_batch.numel()  # Update tokens seen
            global_step += 1
            
            # Optional evaluation
            if global_step % evaluation_frequency == 0:
                train_loss, validation_loss = evaluate_model(
                    model, train_loader, validation_loader, device, 
                    evaluation_steps
                )
                
                train_losses.append(train_loss)
                validation_losses.append(validation_loss)
                track_tokens_seen.append(tokens_seen)
                
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{global_step}], "
                    f"Train Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}, "
                    f"Tokens Seen: {tokens_seen}")
        
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, validation_losses, track_tokens_seen
        
        
        
def evaluate_model(model, train_loader, validation_loader, device, evaluation_steps):
    model.eval()  # Set the model to evaluation mode to disable dropout
    with torch.no_grad():
        train_loss = loader_loss(train_loader, model, device, evaluation_steps)
        validation_loss = loader_loss(validation_loader, model, device, evaluation_steps)
    
    model.train()  # Switch back to training mode
    
    return train_loss, validation_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()  # Set the model to evaluation mode to disable dropout
    
    context_size = model.position_embedding.weight.shape[0]
    encoded = text_to_tokenIds(start_context, tokenizer).to(device)
    
    with torch.no_grad():
        torch.manual_seed(123)
        token_ids = generate_text(model=model, idx=encoded, max_new_tokens=50, context_size=context_size, top_k=25, temperature=1.4)
        
    decoded_text = tokenIds_to_text(token_ids, tokenizer) 
    
    print(decoded_text.replace("\n", " "))
    
    model.train()