import traceback
from huggingface_hub import hf_hub_download
import torch
import torchaudio
from generator import Segment, load_csm_1b

prompt_filepath_conversational_a = hf_hub_download(
    repo_id="sesame/csm-1b",
    filename="prompts/conversational_a.wav"
)

SPEAKER_PROMPTS = {
    "conversational_a": {
        "text": (
            "like revising for an exam I'd have to try and like keep up the momentum because I'd "
            "start really early I'd be like okay I'm gonna start revising now and then like "
            "you're revising for ages and then I just like start losing steam I didn't do that "
            "for the exam we had recently to be fair that was a more of a last minute scenario "
            "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
            "sort of start the day with this not like a panic but like a"
        ),
        "audio": prompt_filepath_conversational_a
    }
}

class Generator:
    def __init__(self, device=None):
        # Prioritize CUDA if available
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
                print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                print("CUDA not available, using CPU")
        
        print(f"Using device: {device}")

        # Load model
        try:
            self.generator = load_csm_1b(device)

            audio_tensor = self.load_prompt_audio(SPEAKER_PROMPTS["conversational_a"]["audio"], self.generator.sample_rate)
            # Move audio tensor to the correct device and ensure it's float32
            audio_tensor = audio_tensor.to(device).to(torch.float32)
            self.defaultSegment = [ Segment(text=SPEAKER_PROMPTS["conversational_a"]["text"], speaker=0, audio=audio_tensor) ]
            self.segments = {}
            self.inputBuffer = {}
            self.device = device
        except Exception as e:
            print(f"Error initializing CSM generator: {e}")
            traceback.print_exc()
            raise

    def generate(self, text: str, speaker_id: int, context_id: str, sample_rate: int = 24000, eos: bool = False) -> torch.Tensor:
        """Generate audio from text using CSM model"""
        audio_tensor = None
        
        try:
            # Initialize context if it doesn't exist
            if context_id not in self.segments:
                print(f"Creating new context for {context_id}")
                self.segments[context_id] = self.defaultSegment.copy()
            
            # Check if context exists
            if context_id not in self.segments:
                print(f"ERROR: Context {context_id} still not initialized after attempt")
                return None

            # Initialize or update input buffer
            if context_id not in self.inputBuffer:
                self.inputBuffer[context_id] = text if text else ""
            elif text:
                self.inputBuffer[context_id] += text

            # Process input if there's content to process
            input_text = self.inputBuffer[context_id].strip()
            if input_text:
                print(f"Generating audio for context {context_id} with \"{input_text}\"")
                print(f"Current context segments: {len(self.segments[context_id])}")
                
                try:
                    audio_tensor = self._generate(
                        text=input_text,
                        speaker=speaker_id,
                        context=self.segments[context_id],
                        target_sample_rate=sample_rate,
                    )
                    
                    # If successful generation, update context
                    if audio_tensor is not None:
                        # Make a copy of the audio tensor as float32 for context
                        # The model expects float type for internal processing
                        print(f"Audio tensor shape: {audio_tensor.shape}, device: {audio_tensor.device}")
                        
                        # Use clone() to create a completely separate copy of the tensor
                        context_audio = audio_tensor.clone().to(self.device).to(torch.float32)
                        
                        # Create a new segment with defensive copies of all data
                        new_segment = Segment(
                            text=input_text, 
                            speaker=speaker_id, 
                            audio=context_audio
                        )
                        
                        # Limit context size to prevent memory issues (keep only last 5 segments)
                        if len(self.segments[context_id]) >= 5:
                            print(f"Trimming context for {context_id} - removing oldest segment")
                            self.segments[context_id] = self.segments[context_id][-4:]
                        
                        # Add the new segment
                        self.segments[context_id].append(new_segment)
                        
                        print(f"Successfully generated audio of length: {audio_tensor.shape[0]} samples")
                        print(f"Updated context to {len(self.segments[context_id])} segments")
                    else:
                        print("Audio generation returned None")
                
                except Exception as e:
                    print(f"Error during audio generation: {e}")
                    traceback.print_exc()
                    # Return a minimal audio tensor rather than None to avoid downstream errors
                    audio_tensor = torch.zeros(sample_rate // 2, dtype=torch.int16, device=self.device)
                    
                # Clear the input buffer after attempting generation
                self.inputBuffer[context_id] = ""
        
        except Exception as e:
            print(f"Outer error in generate method: {e}")
            traceback.print_exc()
            # Return a minimal audio tensor rather than None
            audio_tensor = torch.zeros(sample_rate // 2, dtype=torch.int16, device=self.device)
            
        return audio_tensor

    def load_prompt_audio(self, audio_path: str, target_sample_rate: int) -> torch.Tensor:
        """Load and preprocess audio from file"""
        try:
            audio_tensor, sample_rate = torchaudio.load(audio_path)
            audio_tensor = audio_tensor.squeeze(0)
            
            # Resample to target sample rate
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
            )
            
            # Ensure the audio is in float32 format for model processing
            audio_tensor = audio_tensor.to(torch.float32)
            
            return audio_tensor
            
        except Exception as e:
            print(f"Error loading prompt audio: {e}")
            traceback.print_exc()
            # Return a small dummy tensor if loading fails
            return torch.zeros(target_sample_rate, dtype=torch.float32)

    def _generate(self, text: str, speaker: int, context: list, target_sample_rate: int) -> torch.Tensor:
        """Internal method to generate audio from text"""
        try:
            print(f"Generating audio with device: {self.device}")
            
            # Create a fresh copy of the context to avoid modifying the original
            context_copy = []
            for segment in context:
                # Explicitly clone the audio tensor to create a completely independent copy
                audio_float = segment.audio.clone().to(torch.float32)
                # Create a new segment with a copy of the text
                context_copy.append(Segment(text=segment.text, speaker=segment.speaker, audio=audio_float))
            
            print(f"Context copy created with {len(context_copy)} segments")
            
            # Generate audio using the CSM model
            # Wrap in a try-catch block to handle specific generator errors
            try:
                print("Calling model.generate...")
                audio_tensor = self.generator.generate(
                    text=text,
                    speaker=speaker,
                    context=context_copy,
                    max_audio_length_ms=10_000,
                )
                print(f"Generated raw audio tensor: {audio_tensor.shape}")
            except Exception as e:
                print(f"Error in model.generate: {e}")
                traceback.print_exc()
                # Return a small dummy tensor on error
                return torch.zeros(target_sample_rate // 2, dtype=torch.int16, device=self.device)

            # Ensure the audio tensor is on the correct device before processing
            if audio_tensor.device != self.device:
                audio_tensor = audio_tensor.to(self.device)

            # Resample to target sample rate
            print("Resampling audio...")
            audio_tensor = torchaudio.functional.resample(
                audio_tensor.squeeze(0), orig_freq=self.generator.sample_rate, new_freq=target_sample_rate
            )

            # Convert float audio to int16 format for output
            print("Converting to int16...")
            audio_tensor = audio_tensor * 32767.0
            audio_tensor = torch.clamp(audio_tensor, -32768, 32767)
            audio_tensor = audio_tensor.to(torch.int16)

            return audio_tensor
            
        except Exception as e:
            print(f"Error generating audio: {e}")
            traceback.print_exc()
            # Return a small dummy tensor on error
            return torch.zeros(target_sample_rate // 2, dtype=torch.int16, device=self.device)
