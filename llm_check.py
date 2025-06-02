from vllm import LLM, SamplingParams
import logging

logging.basicConfig(level=logging.INFO)

model_path = "/models/falcon-rw-1b"  # local path to your model
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=50)

llm = LLM(model=model_path, device="cuda")  # 👈 Force CUDA

prompt = "What is the capital of France?"
outputs = llm.generate(prompt, sampling_params)

print("✅ Output:", outputs[0].outputs[0].text)
