import json

def config_loader(config_path: str):
    try:
        with open(config_path, "r") as file:
            config = json.load(file)
        
        return config
    except FileNotFoundError:
        print(f"Error: '{config_path}' not found. Please create the file.")
