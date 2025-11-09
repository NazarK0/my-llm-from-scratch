import os
import json
import tensorflow as tf
from src.utils.load_pgt2_weights.gpt_download3 import (
    download_and_load_gpt2,
    load_gpt2_params_from_tf_ckpt,
)


def load_gpt_weights(model_size: str, models_dir: str):
    # Download and load GPT-2 model parameters
    
    # Settings contain hyperparameters of the model
    settings = {}
    # Parameters contain model weights
    # Keys:
        # 'blocks': list of dicts with weights for each transformer block
        # 'wte': word token embeddings
        # 'wpe': position embeddings
        # 'b': final layer norm beta (weight)
        # 'g': final layer norm gamma (shift)
    # Block part contains:
        # Q, K, V weights and biases
        # 'transformer/h{i}/attn/c_attn/w': weights for query, key, value linear layer
        # 'transformer/h{i}/attn/c_attn/b': biases for query, key, value linear layer
        
        # Feed-Forward Network weights and biases
        # 'transformer/h{i}/mlp/c_fc/w': weights for feed-forward first linear layer
        # 'transformer/h{i}/mlp/c_fc/b': biases for feed-forward first linear layer
        # 'transformer/h{i}/mlp/c_proj/w': weights for feed-forward output linear layer
        # 'transformer/h{i}/mlp/c_proj/b': biases for feed-forward output linear layer
        
        # Output projection weights
        # 'transformer/h{i}/attn/c_proj/w': weights for attention output linear layer
        # 'transformer/h{i}/attn/c_proj/b': biases for attention output linear layer
        
        # Layer Normalization weights and biases
        # 'transformer/h{i}/ln_1/g': layer normalization 1 gamma (scale)
        # 'transformer/h{i}/ln_1/b': layer normalization 1 beta (shift)
        # 'transformer/h{i}/ln_2/g': layer normalization 2 gamma (scale)
        # 'transformer/h{i}/ln_2/b': layer normalization 2 beta (shift)
        
    params: dict[str, list[dict]] = {}

    if os.path.exists(f"{models_dir}/{model_size}"):
        model_dir = os.path.join(models_dir, model_size)
        print("Downloading GPT-2 model parameters is Done.")
        print("Loading GPT-2 model parameters...")
        tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
        settings = json.load(open(os.path.join(model_dir, "hparams.json")))
        params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
    else:
        print("Downloading and loading GPT-2 model parameters...")
        settings, params = download_and_load_gpt2(model_size, models_dir)

    print("Loading GPT-2 model parameters is Done successfully.")

    return settings, params
