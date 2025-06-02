from vllm import LLM, SamplingParams
import logging

logging.basicConfig(level=logging.INFO)

model_path = "models/falcon-rw-1b"
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=50)

llm = LLM(
    model=model_path,
    device="cuda",
    enforce_eager=True  # 👈 disables async output that breaks in Docker
)

prompt = "What is the capital of France?"
outputs = llm.generate(prompt, sampling_params)

print("✅ Output:", outputs[0].outputs[0].text)
