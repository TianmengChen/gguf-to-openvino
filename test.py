
import openvino_genai
from transformers import set_seed

def main():
    set_seed(42)
    model_dir = 'qwen-ov'

    device = 'CPU'  # GPU can be used as well
    pipe = openvino_genai.LLMPipeline(model_dir, device)

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 100
    prompt="请解释一下欧拉角"
    pipe.start_chat()
    result = pipe.generate(prompt, config)
    print(result)
    pipe.finish_chat()


if '__main__' == __name__:
    main()