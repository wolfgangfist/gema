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
    watermarking_time: Optional[float] = None

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
    num_runs: int = 5,
    temperature: float = 0.9,
    topk: int = 50,
    max_audio_length_ms: float = 15000,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"Using device: {device}")
    generator = load_csm_1b(device)

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
    
    print("Performing warmup run...")
    try:
        _ = generator.generate(
            text="This is a short sentence for warmup.",
            speaker=0,
            context=[prompt_a],
            max_audio_length_ms=2000,
            temperature=temperature,
            topk=topk,
            return_timers=False,
            disable_watermark=True,
        )
        print("Warmup complete.")
    except Exception as e:
        print(f"Warmup failed: {e}. Proceeding with benchmark runs, but results might be less stable.")

    for utterance in test_utterances:
        print(f"\nBenchmarking: \"{utterance[:60]}...\"")
        utterance_results = []
        
        for run in range(num_runs):
            print(f"  Run {run+1}/{num_runs}")
            try:
                audio, timers = generator.generate(
                    text=utterance,
                    speaker=0,
                    context=[prompt_a],
                    max_audio_length_ms=max_audio_length_ms,
                    temperature=temperature,
                    topk=topk,
                    return_timers=True,
                    disable_watermark=True,
                )

                audio_duration = len(audio) / generator.sample_rate if generator.sample_rate > 0 else 0
                rtf = timers["total"] / audio_duration if audio_duration > 0 else float('inf')
                num_frames = int(timers.get("num_frames", 0))
                fps = num_frames / timers["generation"] if timers["generation"] > 0 else 0
                
                result = BenchmarkResult(
                    total_time=timers["total"],
                    text_tokenization_time=timers["tokenization"],
                    generation_time=timers["generation"],
                    audio_decoding_time=timers["audio_decoding"],
                    audio_duration=audio_duration,
                    rtf=rtf,
                    frames_per_second=fps,
                    watermarking_time=timers.get("watermarking")
                )
                utterance_results.append(result)

                if run == 0 and audio_duration > 0:
                    filename_prefix = "".join(c if c.isalnum() else "_" for c in utterance[:20])
                    save_path = output_path / f"sample_{filename_prefix}_{len(utterance.split())}_words.wav"
                    print(f"    Saving sample to {save_path}")
                    torchaudio.save(
                        save_path,
                        audio.unsqueeze(0).cpu(),
                        generator.sample_rate
                    )
            except ValueError as e:
                print(f"  Run {run+1} failed: {e}")
                utterance_results.append(None)
            except Exception as e:
                print(f"  Run {run+1} failed unexpectedly: {e}")
                import traceback
                traceback.print_exc()
                utterance_results.append(None)

        valid_results = [r for r in utterance_results if r is not None and r.audio_duration > 0]
        num_valid_runs = len(valid_results)

        if num_valid_runs == 0:
            print(f"  No valid runs completed for this utterance. Skipping average calculation.")
            results.append({
                "utterance": utterance,
                "word_count": len(utterance.split()),
                "error": "All runs failed or produced no audio.",
                "num_valid_runs": 0,
            })
            continue

        avg_result = {
            "utterance": utterance,
            "word_count": len(utterance.split()),
            "avg_total_time": np.mean([r.total_time for r in valid_results]),
            "avg_tokenization_time": np.mean([r.text_tokenization_time for r in valid_results]),
            "avg_generation_time": np.mean([r.generation_time for r in valid_results]),
            "avg_audio_decoding_time": np.mean([r.audio_decoding_time for r in valid_results]),
            "avg_audio_duration": np.mean([r.audio_duration for r in valid_results]),
            "avg_rtf": np.mean([r.rtf for r in valid_results]),
            "avg_fps": np.mean([r.frames_per_second for r in valid_results]),
            "num_valid_runs": num_valid_runs,
        }
        results.append(avg_result)

        print(f"  Average RTF: {avg_result['avg_rtf']:.3f}")
        print(f"  Average FPS: {avg_result['avg_fps']:.2f}")
        print(f"  (Based on {num_valid_runs}/{num_runs} valid runs)")
    
    print("\n--- Benchmark Summary ---")
    for result in results:
        print(f"\nUtterance ({result['word_count']} words): \"{result['utterance'][:60]}...\"")
        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Avg RTF (Total Time / Audio Duration): {result['avg_rtf']:.3f}")
            print(f"  Avg FPS (Frames / Generation Time):  {result['avg_fps']:.2f}")
            print(f"  Avg Total Time: {result['avg_total_time']:.3f} s")
            print(f"    Avg Tokenization:  {result['avg_tokenization_time']*1000:.2f} ms")
            print(f"    Avg Generation:    {result['avg_generation_time']:.3f} s")
            print(f"    Avg Audio Decoding:{result['avg_audio_decoding_time']*1000:.2f} ms")
            print(f"  Avg Audio Duration: {result['avg_audio_duration']:.3f} s")
            print(f"  Valid Runs: {result['num_valid_runs']}/{num_runs}")
    print("--- End Benchmark Summary ---")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark CSM model RTF performance using refactored Generator")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run benchmark on (cuda, cpu)")
    parser.add_argument("--output-dir", type=str, default="benchmark_results_refactored",
                        help="Directory to save benchmark results and samples")
    parser.add_argument("--num-runs", type=int, default=5,
                        help="Number of runs for each test case (after warmup)")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Temperature for generation")
    parser.add_argument("--topk", type=int, default=50,
                        help="Top-k sampling parameter")
    parser.add_argument("--max-length", type=float, default=15000,
                        help="Maximum audio length in milliseconds for generation")
    
    args = parser.parse_args()
    
    if "cuda" in args.device and not torch.cuda.is_available():
        print(f"Warning: Requested device '{args.device}' but CUDA is not available. Falling back to CPU.")
        args.device = "cpu"

    run_benchmark(
        output_dir=args.output_dir,
        device=args.device,
        num_runs=args.num_runs,
        temperature=args.temperature,
        topk=args.topk,
        max_audio_length_ms=args.max_length,
    ) 