import torch
import tiktoken
import os
import time
from datetime import datetime

from src.data_preparation.gpt_dataset import create_dataloader
from src.llm.config.loader import config_loader
from src.llm.gpt_model import GPTModel
from src.utils.host import get_cpu_cores, get_device
from src.utils.loss_fn import loader_loss
from src.utils.train import train_model_simple

tokenizer = tiktoken.get_encoding("gpt2")
gpt_config_163m = config_loader("src/llm/config/gpt_163m.json")
file_path = os.path.join("data", "the-verdict.txt")

# If not exists, create directory for saving the model weights and optimizer state
model_save_path = os.path.join("model", f"gpt_163m-{datetime.now().strftime("%d.%m.%Y")}.pth")
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)


device = get_device()
TASKS: int = 0

if device.type == "cpu":
    cores = get_cpu_cores() # for DataLoader
    tasks_per_core = 1
    TASKS = cores * tasks_per_core
    print(f"Using {cores} CPU cores for DataLoader and {tasks_per_core} tasks per core")

# Read data
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

total_tokens = len(tokenizer.encode(text))
print("First 50 characters of the text:", text[:50])
print("Total characters:", len(text))
print("Total tokens in the text:", total_tokens)

train_ratio = 0.9
split_idx = int(train_ratio * len(text))
train_data = text[:split_idx]
validation_data = text[split_idx:]

torch.manual_seed(123)

train_loader = create_dataloader(
    train_data,
    batch_size=2,
    max_length=gpt_config_163m["context_length"],
    stride=gpt_config_163m["context_length"],
    num_workers=TASKS,
)
validation_loader = create_dataloader(
    validation_data,
    batch_size=2,
    max_length=gpt_config_163m["context_length"],
    stride=gpt_config_163m["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=TASKS,
)

# Sanity check: Decode the first batch of input_ids and target_ids
if total_tokens * train_ratio < gpt_config_163m["context_length"]:
    print("Warning: The training data is smaller than the context length. "
        "Adjust the context length or provide more data.")

if total_tokens * (1 - train_ratio) < gpt_config_163m["context_length"]:
    print("Warning: The validation data is smaller than the context length. "
        "Adjust the context length or provide more data.")


model = GPTModel(gpt_config_163m)
model.eval()  # Set the model to evaluation mode to disable dropout
model.to(device)

torch.manual_seed(123)
with torch.no_grad():
    train_loss = loader_loss(train_loader, model, device)
    validation_loss = loader_loss(validation_loader, model, device)

print(f"Train Loss: {train_loss}")
print(f"Validation Loss: {validation_loss}")


start_time = time.time()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

# Load the model if it exists, otherwise initialize a new model
if os.path.exists(model_save_path):
    print("Loading the model...")
    
    checkpoint = torch.load(model_save_path, map_location=device)
    model = GPTModel(gpt_config_163m)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.train()
    
    print("Model loaded successfully.")
else:
    print("No saved model found, proceeding to train a new model.")
    
    torch.manual_seed(123)
    model = GPTModel(gpt_config_163m)
    model.to(device)


num_epochs = 10

train_losses, validation_losses, tokens_seen = train_model_simple(
    model, train_loader, validation_loader, optimizer, device, num_epochs,
    evaluation_frequency=5, evaluation_steps=5, start_context="Every effort moves you", tokenizer=tokenizer
)

end_time = time.time()
exec_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {exec_time_minutes:.2f} minutes")

print("Saving the model...")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    }, model_save_path)
