import torch
import tiktoken
import os
import time

from src.data_preparation.gpt_dataset import create_dataloader
from src.llm.config.loader import config_loader
from src.llm.gpt_model import GPTModel
from src.utils.host import get_cpu_cores, get_device
from src.utils.loss_fn import loader_loss
from src.utils.train import train_model_simple


device = get_device()
TASKS: int = 0
if device.type == "cpu":
    cores = get_cpu_cores() # for DataLoader
    tasks_per_core = 1
    TASKS = cores * tasks_per_core
    print(f"Using {cores} CPU cores for DataLoader and {tasks_per_core} tasks per core")

tokenizer = tiktoken.get_encoding("gpt2")
gpt_config_163m = config_loader("src/llm/config/gpt_163m.json")
file_path = os.path.join("data", "the-verdict.txt")

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


print("Train loader")
for x, y in train_loader:
    print(f"x shape: {x.shape}, y shape: {y.shape}")

print("\nValidation loader")
for x, y in validation_loader:
    print(f"x shape: {x.shape}, y shape: {y.shape}")

print(len(train_loader))

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

torch.manual_seed(123)
model = GPTModel(gpt_config_163m)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 10

train_losses, validation_losses, tokens_seen = train_model_simple(
    model, train_loader, validation_loader, optimizer, device, num_epochs,
    evaluation_frequency=5, evaluation_steps=5, start_context="Every effort moves you", tokenizer=tokenizer
)

end_time = time.time()
exec_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {exec_time_minutes:.2f} minutes")
