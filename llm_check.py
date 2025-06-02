from vllm import LLM, SamplingParams
import logging

logging.basicConfig(level=logging.INFO)

model_path = "models/falcon-rw-1b"
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=50)

llm = LLM(
    model=model_path,
    device="cuda",              # ✅ explicitly set to CUDA
    enforce_eager=True,         # ✅ disable async output (causes crash in Docker)
    trust_remote_code=True      # ✅ for models with custom code
)

prompt = "What is the capital of France?"
outputs = llm.generate(prompt, sampling_params)

print("✅ Output:", outputs[0].outputs[0].text)
