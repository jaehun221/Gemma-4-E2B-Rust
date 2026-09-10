import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "gemma-4-e2b"

# eager attention으로 로드 (output_attentions 지원)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32,
    attn_implementation="eager",
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

input_ids = tokenizer("The capital of France is", return_tensors="pt").input_ids

with torch.no_grad():
    out = model(input_ids, output_attentions=True)

input_ids = tokenizer("can you speak korean?", return_tensors="pt").input_ids
output = model.generate(input_ids, max_new_tokens=20, do_sample=False)
print(tokenizer.decode(output[0]))