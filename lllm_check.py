from vllm import LLM, SamplingParams
import logging

logging.basicConfig(level=logging.INFO)

# Load a small model that supports GPU (you can replace it with your model)
model_name = "tiiuae/falcon-rw-1b"  # Small and fast, works with GPU
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=50)

# Initialize LLM (it will use GPU if available)
llm = LLM(model=model_name)

# Run inference
prompt = "What is the capital of France?"
outputs = llm.generate(prompt, sampling_params)

# Print output
print("✅ Output:", outputs[0].outputs[0].text)
