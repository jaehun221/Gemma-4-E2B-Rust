import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gemma-4-e2b", dtype=torch.float32)
model.eval()

# 레이어 14와 15 둘 다 hook
captured = {}
def make_hook(name):
    def hook(module, inp, out):
        captured[name] = out
    return hook

h14 = model.model.language_model.layers[14].register_forward_hook(make_hook('l14'))
h15 = model.model.language_model.layers[15].register_forward_hook(make_hook('l15'))

token_ids = torch.tensor([[2, 100, 500]])
with torch.no_grad():
    model(token_ids)

h14.remove(); h15.remove()

for name in ['l14', 'l15']:
    out = captured[name]
    if isinstance(out, tuple): out = out[0]
    print(f"{name} [0, 0, :8]:", out[0, 0, :8])