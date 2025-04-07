
import openvino_genai
from transformers import set_seed
import argparse

def main():
    set_seed(42)
    parser = argparse.ArgumentParser("")
    parser.add_argument("--ov_model_path", type=str, nargs="?", default="qwen-ov")
    parser.add_argument("--prompt", type=str, nargs="?", default="请解释一下欧拉角")
    parser.add_argument("--device", type=str, nargs="?", default="CPU")
    args = parser.parse_args()

    model_dir = args.ov_model_path
    prompt= args.prompt
    device = args.device    

    pipe = openvino_genai.LLMPipeline(model_dir, device)

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 100
    
    pipe.start_chat()
    result = pipe.generate(prompt, config)
    print(result)
    pipe.finish_chat()


if '__main__' == __name__:
    main()