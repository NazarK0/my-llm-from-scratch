import numpy as np

from src.utils.load_pgt2_weights.assign import assign

def load_weights_into_model(model, params):
    """
    Load weights from params into the given model.
    
    Args:
        model: The GPT model instance.
        params: A dictionary containing model weights.
    """
    model.position_embeddings.weight = assign(model.position_embeddings.weight, params['wpe'])
    model.token_embeddings.weight = assign(model.token_embeddings.weight, params['wte'])

    for b in range(len(params['blocks'])):
        q_w, k_w, v_w = np.split((params['blocks'][b]['attn']['c_attn'])['w'], 3, axis=-1)
        model.transformer_blocks[b].attention.weight_query.weight = assign(model.transformer_blocks[b].attention.weight_query.weight, q_w.T)
        model.transformer_blocks[b].attention.weight_key.weight = assign(model.transformer_blocks[b].attention.weight_key.weight, k_w.T)
        model.transformer_blocks[b].attention.weight_value.weight = assign(model.transformer_blocks[b].attention.weight_value.weight, v_w.T)

        q_b, k_b, v_b = np.split((params['blocks'][b]['attn']['c_attn'])['b'], 3, axis=-1)
        model.transformer_blocks[b].attention.weight_query.bias = assign(model.transformer_blocks[b].attention.weight_query.bias, q_b)
        model.transformer_blocks[b].attention.weight_key.bias = assign(model.transformer_blocks[b].attention.weight_key.bias, k_b)
        model.transformer_blocks[b].attention.weight_value.bias = assign(model.transformer_blocks[b].attention.weight_value.bias, v_b)

        model.transformer_blocks[b].attention.out_projection.weight = assign(model.transformer_blocks[b].attention.out_projection.weight, params['blocks'][b]['attn']['c_proj']['w'].T)
        model.transformer_blocks[b].attention.out_projection.bias = assign(model.transformer_blocks[b].attention.out_projection.bias, params['blocks'][b]['attn']['c_proj']['b'])

        model.transformer_blocks[b].feed_forward_network.linear1.weight = assign(model.transformer_blocks[b].feed_forward_network.linear1.weight, params['blocks'][b]['mlp']['c_fc']['w'].T)
        model.transformer_blocks[b].feed_forward_network.linear1.bias = assign(model.transformer_blocks[b].feed_forward_network.linear1.bias, params['blocks'][b]['mlp']['c_fc']['b'])

        model.transformer_blocks[b].feed_forward_network.linear2.weight = assign(model.transformer_blocks[b].feed_forward_network.linear2.weight, params['blocks'][b]['mlp']['c_proj']['w'].T)
        model.transformer_blocks[b].feed_forward_network.linear2.bias = assign(model.transformer_blocks[b].feed_forward_network.linear2.bias, params['blocks'][b]['mlp']['c_proj']['b'])

        model.transformer_blocks[b].layer_normalization1.scale = assign(model.transformer_blocks[b].layer_normalization1.scale, params['blocks'][b]['ln_1']['g'])
        model.transformer_blocks[b].layer_normalization1.shift = assign(model.transformer_blocks[b].layer_normalization1.shift, params['blocks'][b]['ln_1']['b'])

        model.transformer_blocks[b].layer_normalization2.scale = assign(model.transformer_blocks[b].layer_normalization2.scale, params['blocks'][b]['ln_2']['g'])
        model.transformer_blocks[b].layer_normalization2.shift = assign(model.transformer_blocks[b].layer_normalization2.shift, params['blocks'][b]['ln_2']['b'])

    model.final_normalization.scale = assign(model.final_normalization.scale, params['g'])
    model.final_normalization.shift = assign(model.final_normalization.shift, params['b'])
    model.out_head.weight = assign(model.out_head.weight, params['wte'])
