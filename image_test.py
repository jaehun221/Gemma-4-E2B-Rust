import torch, numpy as np
from transformers import AutoProcessor, Gemma4ForConditionalGeneration
from PIL import Image

MODEL = "gemma-4-e2b"
processor = AutoProcessor.from_pretrained(MODEL)
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL, attn_implementation="eager", dtype=torch.float32
).eval()


image = Image.open("test_480x288.png")
prompt = f"{processor.image_token}Describe this image."
inputs = processor(text=prompt, images=image, max_soft_tokens=70, return_tensors="pt")
print(processor.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
print(inputs.keys())

print("pixel_values", inputs["pixel_values"].shape)
print("image_position_ids", inputs["image_position_ids"].shape)
print("image_token_id", model.config.image_token_id,
      "count", (inputs["input_ids"] == model.config.image_token_id).sum().item())


captured = {}

def hook(name):
    def fn(module, args, output):
        captured[name] = output[0] if isinstance(output, tuple) else output
    return fn

vt = model.model.vision_tower
vt.patch_embedder.register_forward_hook(hook("patch_embed"))       # L621 출력
vt.encoder.layers[0].register_forward_hook(hook("enc_layer0"))     # 인코더 층 0 출력
vt.encoder.register_forward_hook(hook("enc_out"))                  # L1058, BaseModelOutput → .last_hidden_state
vt.pooler.register_forward_hook(hook("pooled"))                    # L688, (hidden, mask) 튜플
vt.register_forward_hook(hook("vision_out"))                       # L2050, standardize 후
model.model.embed_vision.register_forward_hook(hook("image_features"))   # L2077

lm = model.model.language_model
_orig_fwd = lm.forward
def fwd_wrap(*a, **kw):
    captured["inputs_embeds"] = kw["inputs_embeds"]                # L1639 시점, 결합 완료본
    return _orig_fwd(*a, **kw)
lm.forward = fwd_wrap

_orig_proj = lm.project_per_layer_inputs
def proj_wrap(*a, **kw):
    out = _orig_proj(*a, **kw)
    captured["per_layer_inputs"] = out                             # L1788 반환값, identity+context 합산본
    return out
lm.project_per_layer_inputs = proj_wrap


with torch.no_grad():
    out = model(**inputs)
captured["logits"] = out.logits


def to_np(t):
    if hasattr(t, "last_hidden_state"): t = t.last_hidden_state
    return t.detach().float().cpu().numpy()

np.save("ref/input_ids.npy", inputs["input_ids"].numpy())
np.save("ref/pixel_values.npy", to_np(inputs["pixel_values"]))
np.save("ref/image_position_ids.npy", inputs["image_position_ids"].numpy())
for k, v in captured.items():
    np.save(f"ref/{k}.npy", to_np(v))

print({k: to_np(v).shape for k, v in captured.items()})


tc, vc = model.config.text_config, model.config.vision_config
print("use_bidirectional_attention:", getattr(tc, "use_bidirectional_attention", None))
print("pad_token_id:", tc.pad_token_id)
print("vision:", vc.num_attention_heads, getattr(vc, "num_key_value_heads", None),
      getattr(vc, "head_dim", None), vc.use_clipped_linears, vc.standardize,
      vc.position_embedding_size, vc.rope_parameters, vc.rms_norm_eps, vc.hidden_activation)
