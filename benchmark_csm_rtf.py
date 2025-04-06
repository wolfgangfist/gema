from typing import Dict, List, Tuple, Optional
import time
import torch
import torchaudio
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from generator import load_csm_1b, Segment, Generator
from huggingface_hub import hf_hub_download

@dataclass
class BenchmarkResult:
    total_time: float
    text_tokenization_time: float
    generation_time: float
    audio_decoding_time: float
    audio_duration: float
    rtf: float
    frames_per_second: float

class GeneratorWithTimers(Generator):
    def __init__(self, model):
        super().__init__(model)
        # Remove watermarking
        self._watermarker = None
    
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        self._model.reset_caches()
        
        timers = {}
        start_total = time.time()
        
        # Tokenization timing
        start_tokenize = time.time()
        max_generation_len = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)
        timers["tokenization"] = time.time() - start_tokenize

        # Generation timing
        start_generation = time.time()
        samples = []
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        max_seq_len = 2048
        max_context_len = max_seq_len - max_generation_len
        if curr_tokens.size(1) >= max_context_len:
            raise ValueError(
                f"Inputs too long, must be below max_seq_len - max_generation_len: {max_context_len}"
            )
        
        frame_times = []
        for _ in range(max_generation_len):
            frame_start = time.time()
            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            frame_times.append(time.time() - frame_start)
            
            if torch.all(sample == 0):
                break  # eos

            samples.append(sample)

            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1
        
        timers["generation"] = time.time() - start_generation
        timers["frame_times"] = frame_times
        
        # Audio decoding timing
        start_decode = time.time()
        audio = self._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0)).squeeze(0).squeeze(0)
        timers["audio_decoding"] = time.time() - start_decode
        
        timers["total"] = time.time() - start_total
        
        return audio.detach(), timers

def load_prompt_audio(audio_path: str, target_sample_rate: int) -> torch.Tensor:
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = audio_tensor.squeeze(0)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
    )
    return audio_tensor

def prepare_prompt(text: str, speaker: int, audio_path: str, sample_rate: int) -> Segment:
    audio_tensor = load_prompt_audio(audio_path, sample_rate)
    return Segment(text=text, speaker=speaker, audio=audio_tensor)

def run_benchmark(
    output_dir: str = "benchmark_results",
    device: str = "cuda",
    num_runs: int = 2,
    temperature: float = 0.9,
    max_audio_length_ms: float = 10000,
) -> None:
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load model
    model = load_csm_1b(device)
    
    # Replace with our instrumented generator
    generator = GeneratorWithTimers(model._model)
    
    # Load prompts from HF
    prompt_filepath_a = hf_hub_download(
        repo_id="sesame/csm-1b",
        filename="prompts/conversational_a.wav"
    )
    prompt_a_text = (
        "like revising for an exam I'd have to try and like keep up the momentum because I'd "
        "start really early I'd be like okay I'm gonna start revising now and then like "
        "you're revising for ages and then I just like start losing steam I didn't do that "
        "for the exam we had recently to be fair that was a more of a last minute scenario "
        "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
        "sort of start the day with this not like a panic but like a"
    )
    
    prompt_a = prepare_prompt(prompt_a_text, 0, prompt_filepath_a, generator.sample_rate)
    
    test_utterances = [
        "Hello, how are you today?",
        "This is a medium length utterance for testing the real-time factor of the model.",
        "Now I am going to speak for a longer time to ensure we have a good benchmark of the model's performance with longer utterances. This will help us understand how the model scales with input length and how efficient it is at generating speech from longer text inputs."
    ]
    
    results = []
    
    for utterance in test_utterances:
        print(f"Benchmarking: {utterance}")
        utterance_results = []
        
        for run in range(num_runs):
            print(f"  Run {run+1}/{num_runs}")
            # Run generation with timing
            audio, timers = generator.generate(
                text=utterance,
                speaker=0,
                context=[prompt_a],
                max_audio_length_ms=max_audio_length_ms,
                temperature=temperature,
            )
            
            # Calculate metrics
            audio_duration = len(audio) / generator.sample_rate
            rtf = timers["total"] / audio_duration
            num_frames = len(timers["frame_times"])
            fps = num_frames / timers["generation"] if timers["generation"] > 0 else 0
            
            result = BenchmarkResult(
                total_time=timers["total"],
                text_tokenization_time=timers["tokenization"],
                generation_time=timers["generation"],
                audio_decoding_time=timers["audio_decoding"],
                audio_duration=audio_duration,
                rtf=rtf,
                frames_per_second=fps
            )
            utterance_results.append(result)
            
            # Save the audio - Fixed: detach the tensor before saving
            if run == 0:
                torchaudio.save(
                    output_path / f"sample_{len(utterance.split())}_words.wav",
                    audio.unsqueeze(0).cpu(),
                    generator.sample_rate
                )
        
        # Compute averages
        avg_result = {
            "utterance": utterance,
            "word_count": len(utterance.split()),
            "avg_total_time": np.mean([r.total_time for r in utterance_results]),
            "avg_tokenization_time": np.mean([r.text_tokenization_time for r in utterance_results]),
            "avg_generation_time": np.mean([r.generation_time for r in utterance_results]),
            "avg_audio_decoding_time": np.mean([r.audio_decoding_time for r in utterance_results]),
            "avg_audio_duration": np.mean([r.audio_duration for r in utterance_results]),
            "avg_rtf": np.mean([r.rtf for r in utterance_results]),
            "avg_fps": np.mean([r.frames_per_second for r in utterance_results]),
        }
        results.append(avg_result)
        
        print(f"  Average RTF: {avg_result['avg_rtf']:.2f}")
        print(f"  Average FPS: {avg_result['avg_fps']:.2f}")
    
    # Print summary
    print("\nBenchmark Summary:")
    for result in results:
        print(f"\nUtterance ({result['word_count']} words): {result['utterance'][:30]}...")
        print(f"  Average RTF: {result['avg_rtf']:.2f}")
        print(f"  Average FPS: {result['avg_fps']:.2f}")
        print(f"  Tokenization: {result['avg_tokenization_time']*1000:.2f} ms")
        print(f"  Generation: {result['avg_generation_time']:.2f} s")
        print(f"  Audio decoding: {result['avg_audio_decoding_time']*1000:.2f} ms")
        print(f"  Audio duration: {result['avg_audio_duration']:.2f} s")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark CSM model RTF performance")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run benchmark on (cuda, cpu)")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Directory to save benchmark results and samples")
    parser.add_argument("--num-runs", type=int, default=5,
                        help="Number of runs for each test case")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Temperature for generation")
    parser.add_argument("--max-length", type=float, default=10000,
                        help="Maximum audio length in milliseconds")
    
    args = parser.parse_args()
    
    run_benchmark(
        output_dir=args.output_dir,
        device=args.device,
        num_runs=args.num_runs,
        temperature=args.temperature,
        max_audio_length_ms=args.max_length,
    ) 