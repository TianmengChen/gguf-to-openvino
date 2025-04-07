# gguf-to-openvino

An example that reads GGUF and creates OpenVINO on the fly. 
This example draws on two repos: [gguf-to-openvino](https://github.com/AlexKoff88/gguf-to-openvino.git) and [pygguf](https://github.com/99991/pygguf.git).

You can use this script to quickly convert **Qwen/Llama**<sup>*</sup> GGUF files to OpenVINO models on the fly. When generating an OpenVINO model, we do not do dequantization, but instead unpack the quantized data types of GGUF directly to the quantized data types supported by OpenVINO and build a graph.

*verified model: qwen2.5-0.5b-instruct, qwen2.5-3b-instruct, qwen2.5-7b-instruct, Meta-Llama-3.1-8B. 


## Support matrix

|  GGUF   | converted OpenVINO  |
| :----:  |:----: |
| Q4_0  | INT4 |
| Q4_K  | INT4/INT8<sup>*</sup> |
| Q6_K  | INT8 |
| Q8_0  | INT8 |
| FP16  | FP16 |

*: INT8 is used for token_embd layers in order to align with OpenVINO structure, but you can change it to INT4 if you want by adding option --all_layer using convert_gguf.py.

## Usage
1. Requirements install:
```sh
pip install -r requirements.txt
``` 

2. Download GGUF file:
```sh
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_0.gguf --local-dir models
```

3. Convert the model:
```sh
python convert_gguf.py --org_model_path models/qwen2.5-0.5b-instruct-q4_0.gguf --ov_model_path models/qwen-ov --model_id Qwen/Qwen2.5-0.5B-Instruct --all_layer
```

4. Test the model:
```sh
python test.py --ov_model_path models/qwen-ov --prompt 请解释一下欧拉角 --device CPU
```