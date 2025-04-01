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

Notice:
Use python 3.13 to serialize tokenizer model IR on Windows, python 3.11 and 3.12 will fail.