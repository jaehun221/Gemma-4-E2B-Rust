import torch
import torch.nn.functional as F
from safetensors import safe_open


MODEL_PATH = "gemma-4-e2b/model.safetensors"
LAYER = 0            
SEED = 42                      


with safe_open(MODEL_PATH, framework="pt") as f:
    gate_proj = f.get_tensor(f"model.language_model.layers.{LAYER}.mlp.gate_proj.weight").float()
    up_proj   = f.get_tensor(f"model.language_model.layers.{LAYER}.mlp.up_proj.weight").float()
    down_proj = f.get_tensor(f"model.language_model.layers.{LAYER}.mlp.down_proj.weight").float()

print("gate_proj:", gate_proj.shape)   
print("up_proj:  ", up_proj.shape)     
print("down_proj:", down_proj.shape)   


torch.manual_seed(SEED)
hidden_size = gate_proj.shape[1]      
# torch.manual_seed(SEED) / torch.randn 대신
x = torch.zeros(2, 1536)
x[0, 0] = 1.0
x[0, 1] = 0.5
x[0, 2] = -0.3
x[1, 0] = -1.0
x[1, 1] = 2.0
# 나머지 0    


gate = x @ gate_proj.T                 
up   = x @ up_proj.T                   
hidden = F.gelu(gate, approximate='tanh') * up   
out = hidden @ down_proj.T


print("\n입력 x 첫 행 앞 8개:")
print(x[0, :8])

print("\nMLP 출력 첫 행 앞 8개:")
print(out[0, :8])

print("\nMLP 출력 둘째 행 앞 8개:")
print(out[1, :8])