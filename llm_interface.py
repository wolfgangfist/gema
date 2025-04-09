from llama_cpp import Llama
from typing import List

class LLMInterface:
    def __init__(self, model_path: str, max_tokens: int = 8192, n_threads: int = 8, gpu_layers: int = -1):
        """Initialize the LLM interface using llama-cpp-python with a GGUF model."""
        self.llm = Llama(
            model_path=model_path,
            n_ctx=max_tokens,
            n_threads=n_threads,
            n_gpu_layers=gpu_layers,
            use_mlock=True,
        )

    def generate_response(self, system_prompt: str, user_message: str, conversation_history: str = "") -> str:
        """Generate a response from the LLM using chat-style prompt formatting."""
        prompt = f"""<|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>

{conversation_history}

<|start_header_id|>user<|end_header_id|>\n{user_message}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>\n"""

        output = self.llm(
            prompt=prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repeat_penalty=1.1,
        )
        stop_tokens = ["</s>", "<|endoftext|>", "<<USR>>", "<</USR>>", "<</SYS>>", "<</USER>>", "<</ASSISTANT>>", "<|end_header_id|>", "<<ASSISTANT>>", "<|eot_id|>", "<|im_end|>", "user:", "User:", "user :", "User :"]
        output = self.llm(prompt, **{
                    "max_tokens": 512,
                    "stop": stop_tokens,
                    "echo": False,
                    "top_k": 400,
                    "top_p": 0.95,
                    "temperature": 0.7,
                    "repeat_penalty": 1.2
                })
        return output["choices"][0]["text"].strip()

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text using llama-cpp's tokenizer."""
        return self.llm.tokenize(text)

    def get_token_count(self, text: str) -> int:
        """Return token count of the input text."""
        return len(self.tokenize(text))
