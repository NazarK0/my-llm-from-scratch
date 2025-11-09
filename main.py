import torch
import tiktoken
import os

from src.generate_text import generate_text
from src.llm.config.loader import config_loader
from src.llm.gpt_model import GPTModel
from src.utils.convert import text_to_tokenIds, tokenIds_to_text
from src.utils.load_pgt2_weights.load_weights_into_model import load_weights_into_model
from src.utils.load_pgt2_weights.gpt_weights_loader import load_gpt_weights
from src.utils.host import get_cpu_cores, get_device

tokenizer = tiktoken.get_encoding("gpt2")
gpt_config_163m = config_loader("src/llm/config/gpt_163m.json")
file_path = os.path.join("data", "the-verdict.txt")


if __name__ == '__main__':
    device = get_device()
    TASKS: int = 0

    if device.type == "cpu":
        cores = get_cpu_cores() # for DataLoader
        tasks_per_core = 1
        TASKS = cores * tasks_per_core
        print(f"Using {cores} CPU cores for DataLoader and {tasks_per_core} tasks per core")
        
    settings, params = load_gpt_weights(model_size="124M", models_dir="gpt2")
    print("GPT-2 settings:", settings)
    print("GPT-2 params keys:", params.keys())
    
    model_configs = {
        "gpt2-small (124M)": {"embedding_dimension": 768, "number_of_layers": 12, "number_of_heads": 12},
        "gpt2-medium (355M)": {"embedding_dimension": 1024, "number_of_layers": 24, "number_of_heads": 16},
        "gpt2-large (774M)": {"embedding_dimension": 1280, "number_of_layers": 36, "number_of_heads": 20},
        "gpt2-xl (1558M)": {"embedding_dimension": 1600, "number_of_layers": 48, "number_of_heads": 25},
    }
    
    model_name = "gpt2-small (124M)"
    new_config = gpt_config_163m.copy()
    new_config.update(model_configs[model_name])
    new_config.update({"context_length": 1024, "qkv_bias": True})
    print("Model configuration:", new_config)
    
    gpt = GPTModel(new_config)
    gpt.eval()
    
    load_weights_into_model(gpt, params)
    gpt.to(device)
    
    idx = text_to_tokenIds("Every effort moves you", tokenizer).to(device)
    
    torch.manual_seed(123)
    token_ids = generate_text(
        model=gpt, 
        idx= idx, 
        max_new_tokens=25, 
        context_size=new_config["context_length"], 
        temperature=1.5, 
        top_k=50
        )
    
    generated_text = tokenIds_to_text(token_ids, tokenizer)
    
    print("Generated text:\n", generated_text)
