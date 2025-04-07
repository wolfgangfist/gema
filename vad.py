import numpy as np
import torch
from typing import Callable, Dict, List
from collections import deque

class VoiceActivityDetector:
    def __init__(
        self,
        model,
        utils,
        sample_rate: int = 16000,
        threshold: float = 0.2,  # Much lower threshold to detect softer speech
        min_speech_frames: int = 1,  # Only need 1 frame to start speech
        min_silence_frames: int = 15,  # Reduced silence requirement to end speech
        max_silence_frames: int = 30  # Shorter maximum silence timeout
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_frames = min_speech_frames
        self.min_silence_frames = min_silence_frames
        self.max_silence_frames = max_silence_frames
        self.frame_size = 512

        # Make sure model is on CPU for better compatibility
        self.model = model.to('cpu')
        self.get_speech_timestamps, *_ = utils

        self.is_speaking = False
        self.speech_frames = 0
        self.silence_frames = 0
        self.energy_history = deque(maxlen=20)
        
        # Add hysteresis to avoid rapid state changes
        self.hysteresis_window = deque(maxlen=3)  # Last 3 frames
        
        # Enhanced background noise adaptation
        self.background_noise_level = 0
        self.adaptation_rate = 0.95  # Faster adaptation rate

    def process_audio_chunk(self, audio_chunk: np.ndarray) -> bool:
        # Prepare audio chunk
        if audio_chunk.ndim > 1:
            audio_chunk = np.mean(audio_chunk, axis=1)
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Calculate energy of the chunk for noise adaptation
        energy = np.mean(np.abs(audio_chunk))
        self.energy_history.append(energy)
        
        # Dynamic noise adaptation - adjust threshold based on background noise
        if len(self.energy_history) >= 10 and not self.is_speaking:
            # Update background noise level slowly when not speaking
            self.background_noise_level = (self.adaptation_rate * self.background_noise_level + 
                                          (1 - self.adaptation_rate) * np.mean(self.energy_history))

        # Convert to tensor and make sure it's on CPU
        audio_tensor = torch.from_numpy(audio_chunk).to('cpu')

        # Pad or truncate to match frame size
        if len(audio_tensor) < self.frame_size:
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, self.frame_size - len(audio_tensor)))
        else:
            audio_tensor = audio_tensor[:self.frame_size]

        # Get speech probability using Silero VAD
        try:
            # Direct model inference instead of using iterator
            with torch.no_grad():
                speech_prob_output = self.model(audio_tensor, self.sample_rate)
                
                # Handle different return types from the model
                if isinstance(speech_prob_output, dict):
                    # New version returns a dict with probabilities
                    if 'speech_prob' in speech_prob_output:
                        speech_prob = speech_prob_output['speech_prob'].item()
                    else:
                        # If we can't find the right key, use a default
                        print(f"Warning: Unexpected dict keys: {speech_prob_output.keys()}")
                        speech_prob = 0.0
                elif isinstance(speech_prob_output, torch.Tensor):
                    # Older version returns a tensor directly
                    speech_prob = speech_prob_output.item()
                else:
                    # Unexpected return type
                    print(f"Warning: Unexpected return type: {type(speech_prob_output)}")
                    speech_prob = 0.0
                
            # Boost probability for higher sensitivity
            boosted_prob = min(1.0, speech_prob * 1.5)  # Boost by 50%
            
            # Apply hysteresis to avoid rapid state changes
            self.hysteresis_window.append(boosted_prob > self.threshold)
            is_speech_frame = sum(self.hysteresis_window) >= 1  # Any of last 3 frames is speech
            
            # Always print VAD probability for debugging
            print(f"VAD prob: {speech_prob:.2f} (boosted: {boosted_prob:.2f}), is_speech: {is_speech_frame}, frames - speech: {self.speech_frames}, silence: {self.silence_frames}, speaking: {self.is_speaking}")
                
        except Exception as e:
            print(f"Error in VAD processing: {e}")
            is_speech_frame = False
            speech_prob = 0.0

        # Update state counters
        if is_speech_frame:
            self.speech_frames += 1
            self.silence_frames = max(0, self.silence_frames - 2)  # Faster reset of silence frames
        else:
            self.silence_frames += 1
            self.speech_frames = max(0, self.speech_frames - 1)  # Gradually decrease speech frames

        # State transitions
        if not self.is_speaking:
            if self.speech_frames >= self.min_speech_frames:
                self.is_speaking = True
                self.speech_frames = 0
                print("VAD: Speech started")
                return False
        else:
            if self.silence_frames >= self.min_silence_frames:
                self.is_speaking = False
                self.silence_frames = 0
                print("VAD: Speech ended")
                return True
            elif self.silence_frames >= self.max_silence_frames:
                self.is_speaking = False
                self.silence_frames = 0
                print("VAD: Extended silence, forced end")
                return True

        return False

    def reset(self):
        self.is_speaking = False
        self.speech_frames = 0
        self.silence_frames = 0
        self.energy_history.clear()
        self.hysteresis_window.clear()
        print("VAD: Reset")


class AudioStreamProcessor:
    def __init__(
        self,
        model,
        utils,
        sample_rate: int = 16000,
        chunk_size: int = 512,
        vad_threshold: float = 0.2,  # Lower threshold for higher sensitivity
        callbacks: Dict[str, Callable] = None
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        # Force model to CPU for reliable performance
        if hasattr(model, 'to'):
            model = model.to('cpu')

        self.vad = VoiceActivityDetector(
            model=model,
            utils=utils,
            sample_rate=sample_rate,
            threshold=vad_threshold,
            min_speech_frames=1,  # Ultra sensitive - just 1 frame to trigger
            min_silence_frames=15,  # Less silence needed to end speech
            max_silence_frames=30  # Faster timeout
        )

        self.audio_buffer = []
        self.is_collecting = False
        self.callbacks = callbacks or {}
        print(f"AudioStreamProcessor initialized with ULTRA-SENSITIVE threshold: {vad_threshold}")

    def process_audio(self, audio_chunk: np.ndarray):
        # Ensure proper audio format
        if audio_chunk.ndim > 1:
            audio_chunk = np.mean(audio_chunk, axis=1)

        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Pre-speech buffer to capture the beginning of utterances
        # Add a tiny bit of pre-buffering
        if not self.is_collecting and len(self.audio_buffer) < 5:
            self.audio_buffer.append(audio_chunk)
        elif not self.is_collecting:
            # Keep a rolling buffer of the last 5 chunks
            self.audio_buffer.pop(0)
            self.audio_buffer.append(audio_chunk)

        # Process with VAD
        is_turn_end = self.vad.process_audio_chunk(audio_chunk)

        # Start collecting on speech detection
        if self.vad.is_speaking and not self.is_collecting:
            self.is_collecting = True
            # Don't reset audio buffer to keep pre-speech data
            print(f"Starting to collect audio with {len(self.audio_buffer)} pre-buffered chunks...")
            if "on_speech_start" in self.callbacks:
                self.callbacks["on_speech_start"]()

        # Collect audio while speaking
        if self.is_collecting:
            # Don't add again if we're already collecting
            if not self.audio_buffer or self.audio_buffer[-1] is not audio_chunk:
                self.audio_buffer.append(audio_chunk)

        # Handle end of speech
        if is_turn_end and self.is_collecting:
            self.is_collecting = False
            if self.audio_buffer:
                print(f"Processing collected audio of length: {len(self.audio_buffer)} chunks")
                complete_audio = np.concatenate(self.audio_buffer)
                if "on_speech_end" in self.callbacks:
                    self.callbacks["on_speech_end"](complete_audio, self.sample_rate)
                else:
                    print("Warning: No on_speech_end callback registered")
                # Reset buffer but keep some silence for next pre-buffer
                self.audio_buffer = self.audio_buffer[-2:] if len(self.audio_buffer) >= 2 else []

    def reset(self):
        self.vad.reset()
        self.audio_buffer = []
        self.is_collecting = False
        print("AudioStreamProcessor reset")