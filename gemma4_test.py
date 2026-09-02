import torch
import torch.nn.functional as F
from safetensors import safe_open

MODEL_PATH = "gemma-4-e2b/model.safetensors"

with safe_open(MODEL_PATH, framework="pt") as f:
    ple_table = f.get_tensor("model.language_model.embed_tokens_per_layer.weight").float()
    model_proj = f.get_tensor("model.language_model.per_layer_model_projection.weight").float()
    proj_norm = f.get_tensor("model.language_model.per_layer_projection_norm.weight").float()

num_layers = 35
ple_dim = 256
hidden_size = 1536

# 고정 입력
token_ids = [2, 100, 500]   # 예시 토큰
embeds = torch.zeros(3, 1536)
embeds[0,0]=1.0; embeds[0,1]=0.5; embeds[1,0]=-1.0; embeds[2,0]=2.0

# --- token-identity ---
identity = ple_table[token_ids]                    # [3, 8960]
identity = identity * (ple_dim ** 0.5)             # ×16
identity = identity.reshape(3, num_layers, ple_dim)  # [3, 35, 256]

# --- context-aware ---
context = embeds @ model_proj.T                     # [3, 8960]
context = context * (hidden_size ** -0.5)           # ×(1/√1536)
context = context.reshape(3, num_layers, ple_dim)   # [3, 35, 256]
# RMSNorm
mean_sq = context.pow(2).mean(-1, keepdim=True)
context = context * torch.rsqrt(mean_sq + 1e-6) * proj_norm

# --- 결합 ---
ple = (context + identity) * (2.0 ** -0.5)          # ×(1/√2)

# 출력
print("identity [0,0,:8]:", identity[0,0,:8])
print("context [0,0,:8]:", context[0,0,:8])
print("최종 ple [0,0,:8]:", ple[0,0,:8])