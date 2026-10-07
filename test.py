from safetensors import safe_open
with safe_open("gemma-4-e2b/model.safetensors", "pt") as f:
    for k in [
        "model.vision_tower.patch_embedder.input_proj.weight",
        "model.vision_tower.patch_embedder.position_embedding_table",
        "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight",
        "model.vision_tower.encoder.layers.0.self_attn.q_proj.input_min",
        "model.vision_tower.encoder.layers.0.self_attn.q_norm.weight",
        "model.embed_vision.embedding_projection.weight",
    ]:
        print(k, f.get_slice(k).get_dtype())
    print(f.get_tensor("model.vision_tower.encoder.layers.0.self_attn.q_proj.input_min"))
    