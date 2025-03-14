# gguf-to-openvino
An example that reads GGUF and creates OpenVINO on the fly.

## Usage
1. Download GGUF file:
```sh
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_0.gguf --local-dir models
```

2. Convert the model:
```sh
python convert_gguf.py --org_model_path models/qwen2.5-0.5b-instruct-q4_0.gguf --ov_model_path models/qwen-ov
```
