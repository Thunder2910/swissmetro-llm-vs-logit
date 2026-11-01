# Model weights

The GGUF files are not in git. `bash run.sh` downloads any that are missing and checks the byte count. You can also download them yourself into this directory.

All three are Qwen2.5-Instruct, quantisation `Q4_K_M`, from Hugging Face. No account and no token are required for these public files.

| Model | File | Bytes |
| --- | --- | ---: |
| Qwen2.5-1.5B-Instruct | `qwen2.5-1.5b-instruct-q4_k_m.gguf` | 1,117,320,736 |
| Qwen2.5-3B-Instruct | `qwen2.5-3b-instruct-q4_k_m.gguf` | 2,104,932,768 |
| Qwen2.5-7B-Instruct | `qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf` | 3,993,201,344 |
| Qwen2.5-7B-Instruct | `qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf` | 689,872,288 |

The 7B model is published as two shards. Keep both files in this directory. llama.cpp loads the pair when it is given the first shard.

```text
https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf
```

`src/model_specs.py` is the list the code actually uses.
