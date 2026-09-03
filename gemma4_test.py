import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gemma-4-e2b", dtype=torch.float32)
model.eval()

token_ids = torch.tensor([[2, 100, 500]])

with torch.no_grad():
    out = model(token_ids)

logits = out.logits   # [1, 3, vocab_size]

print("logits [0, 0, :8]:", logits[0, 0, :8])