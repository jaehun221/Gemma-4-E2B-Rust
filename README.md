## Download Weights

```
# huggingface_hub install
pip install huggingface_hub

# model download
huggingface-cli download google/gemma-4-E2B \
  --local-dir gemma-4-e2b \
  --include "config.json" \
             "generation_config.json" \
             "tokenizer.json" \
             "tokenizer_config.json" \
             "processor_config.json" \
             "model.safetensors"
```
