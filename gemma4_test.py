from safetensors import safe_open

with safe_open("gemma-4-e2b/model.safetensors", framework="pt") as f:
    w = f.get_tensor("model.language_model.layers.0.input_layernorm.weight")
    print(w[:8])
    print("평균:", w.float().mean())