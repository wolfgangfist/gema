import traceback
import os
from huggingface_hub import hf_hub_download
import torch
import torchaudio
from generator import Segment, load_csm_1b

# Define local and remote paths
PROMPT_FILENAME = "prompts/conversational_a.wav"
LOCAL_PROMPT_PATH = PROMPT_FILENAME  # Relative path to locally downloaded file

# Try to use local file first, fall back to Hugging Face download
def get_prompt_path(filename):
    if os.path.exists(filename):
        print(f"Using local prompt file: {filename}")
        return filename
    else:
        print(f"Local prompt file not found, downloading from Hugging Face: {filename}")
        return hf_hub_download(
            repo_id="sesame/csm-1b",
            filename=filename
        )

# Get the path to the prompt file
prompt_filepath_conversational_a = get_prompt_path(LOCAL_PROMPT_PATH)

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
            print("\n=== Initializing CSM Generator ===")
            self.generator = load_csm_1b(device)
            print("CSM 1B model loaded successfully")

            print(f"\nLoading prompt audio from: {SPEAKER_PROMPTS['conversational_a']['audio']}")
            print(f"Prompt text: {SPEAKER_PROMPTS['conversational_a']['text'][:50]}...")

            audio_tensor = self.load_prompt_audio(SPEAKER_PROMPTS["conversational_a"]["audio"], self.generator.sample_rate)
            
            # Check if audio tensor has valid data
            if audio_tensor.numel() == 0 or torch.all(audio_tensor == 0):
                print("WARNING: Loaded audio tensor appears to be empty or all zeros!")
                print("This will affect voice quality - check the prompt file!")
            else:
                print(f"Loaded prompt audio tensor with shape: {audio_tensor.shape}")
                print(f"Audio stats - Min: {audio_tensor.min():.4f}, Max: {audio_tensor.max():.4f}")
                
            # Move audio tensor to the correct device and ensure it's float32
            audio_tensor = audio_tensor.to(device).to(torch.float32)

            # Create default segment with the prompt
            print("\nCreating default segment with prompt...")
            self.defaultSegment = [ Segment(text=SPEAKER_PROMPTS["conversational_a"]["text"], speaker=0, audio=audio_tensor) ]
            print(f"Default segment created with audio shape: {self.defaultSegment[0].audio.shape}")

            self.segments = {}
            self.inputBuffer = {}
            self.responseBuffer = {}  # Store AI response text
            self.device = device
            print("=== Initialization Complete ===\n")
        except Exception as e:
            print(f"Error initializing CSM generator: {e}")
            traceback.print_exc()
            raise

    def store_input(self, text: str, speaker_id: int, context_id: str):
        """
        Store user input in the conversation context without generating audio
        
        Args:
            text: The user's input text
            speaker_id: Speaker ID (1 for user, 0 for AI)
            context_id: Unique ID for this conversation
        """
        try:
            # Initialize context if it doesn't exist
            if context_id not in self.segments:
                print(f"Creating new context for {context_id}")
                self.segments[context_id] = self.defaultSegment.copy()
                
            if not text or not text.strip():
                print("Empty text input, nothing to store")
                return
                
            print(f"Storing input for speaker {speaker_id}: '{text}'")
            
            # Create a dummy audio tensor for the user's input
            # We don't need actual audio, just a placeholder since the model requires it
            dummy_audio = torch.zeros(16000, dtype=torch.float32, device=self.device)
            
            # Create a new segment with the user's text
            new_segment = Segment(
                text=text.strip(),
                speaker=speaker_id,
                audio=dummy_audio
            )
            
            # Limit context size to prevent memory issues (keep only last 5 segments)
            if len(self.segments[context_id]) >= 5:
                print(f"Trimming context for {context_id} - removing oldest segment")
                self.segments[context_id] = self.segments[context_id][-4:]
            
            # Add the user segment to the context
            self.segments[context_id].append(new_segment)
            print(f"Added user input to context. Now {len(self.segments[context_id])} segments")
            
            # Store the input in the response buffer for this context
            if context_id not in self.responseBuffer:
                self.responseBuffer[context_id] = ""
                
        except Exception as e:
            print(f"Error storing user input: {e}")
            traceback.print_exc()

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

            # Initialize response buffer for this context if needed
            if context_id not in self.responseBuffer:
                self.responseBuffer[context_id] = ""
                
            # Add user text to the response buffer if provided
            if text and text.strip():
                self.responseBuffer[context_id] += text.strip() + " "
                
            # Get the AI's response text to generate
            response_text = self.responseBuffer[context_id].strip()
            
            # Process input if there's content in the context
            if len(self.segments[context_id]) > 0:
                print(f"Generating audio response with speaker_id={speaker_id} for context {context_id}")
                print(f"Current context has {len(self.segments[context_id])} segments")
                print(f"Response text: '{response_text if response_text else '[Using empty text with context]'}'")
                
                try:
                    # Generate response from the AI (speaker_id=0)
                    audio_tensor = self._generate(
                        text=response_text,  # Can be empty, will generate based on context
                        speaker=speaker_id,  # This should be 0 for AI responses
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
                        # This is the AI's response
                        response_segment = Segment(
                            text=response_text if response_text else "[AI Response]", 
                            speaker=speaker_id,  # This should be 0 for AI
                            audio=context_audio
                        )
                        
                        # Limit context size to prevent memory issues (keep only last 5 segments)
                        if len(self.segments[context_id]) >= 5:
                            print(f"Trimming context for {context_id} - removing oldest segment")
                            self.segments[context_id] = self.segments[context_id][-4:]
                        
                        # Add the AI response segment
                        self.segments[context_id].append(response_segment)
                        
                        print(f"Successfully generated audio of length: {audio_tensor.shape[0]} samples")
                        print(f"Updated context to {len(self.segments[context_id])} segments")
                    else:
                        print("Audio generation returned None")
                
                except Exception as e:
                    print(f"Error during audio generation: {e}")
                    traceback.print_exc()
                    # Return a minimal audio tensor rather than None to avoid downstream errors
                    audio_tensor = torch.zeros(sample_rate // 2, dtype=torch.int16, device=self.device)
                    
                # Clear the response buffer after generation
                self.responseBuffer[context_id] = ""
        
        except Exception as e:
            print(f"Outer error in generate method: {e}")
            traceback.print_exc()
            # Return a minimal audio tensor rather than None
            audio_tensor = torch.zeros(sample_rate // 2, dtype=torch.int16, device=self.device)
            
        return audio_tensor

    def load_prompt_audio(self, audio_path: str, target_sample_rate: int) -> torch.Tensor:
        """Load and preprocess audio from file"""
        try:
            print(f"Loading prompt audio from path: {audio_path}")
            if not os.path.exists(audio_path):
                print(f"WARNING: Audio file does not exist: {audio_path}")
                return torch.zeros(target_sample_rate, dtype=torch.float32)
                
            audio_tensor, sample_rate = torchaudio.load(audio_path)
            print(f"Loaded audio with shape: {audio_tensor.shape}, sample rate: {sample_rate}")
            
            audio_tensor = audio_tensor.squeeze(0)
            
            # Resample to target sample rate
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
            )
            
            # Ensure the audio is in float32 format for model processing
            audio_tensor = audio_tensor.to(torch.float32)
            
            # Verify audio has valid values
            if torch.isnan(audio_tensor).any() or torch.isinf(audio_tensor).any():
                print("WARNING: Audio contains NaN or Inf values")
                # Replace with zeros
                audio_tensor = torch.zeros_like(audio_tensor)
            elif audio_tensor.abs().max() == 0:
                print("WARNING: Audio tensor is all zeros")
                
            return audio_tensor
            
        except Exception as e:
            print(f"Error loading prompt audio: {e}")
            traceback.print_exc()
            # Return a small dummy tensor if loading fails
            return torch.zeros(target_sample_rate, dtype=torch.float32)

    def _generate(self, text: str, speaker: int, context: list, target_sample_rate: int) -> torch.Tensor:
        """Internal method to generate audio from text"""
        try:
            print(f"\n=== Generation Details ===")
            print(f"Generating audio with device: {self.device}")
            print(f"Input text: '{text}'")
            print(f"Speaker ID: {speaker}")
            
            # Create a fresh copy of the context to avoid modifying the original
            context_copy = []
            print("\nContext sequence:")
            for idx, segment in enumerate(context):
                # Explicitly clone the audio tensor to create a completely independent copy
                audio_float = segment.audio.clone().to(torch.float32)
                # Create a new segment with a copy of the text
                context_copy.append(Segment(text=segment.text, speaker=segment.speaker, audio=audio_float))
                print(f"  {idx}: Speaker {segment.speaker} - Text: '{segment.text[:50]}...'")
                print(f"     Audio shape: {audio_float.shape}, Range: ({audio_float.min():.4f}, {audio_float.max():.4f})")
            
            print(f"\nContext copy created with {len(context_copy)} segments")
            
            # Log the speaker IDs in the context for debugging
            speaker_sequence = [s.speaker for s in context_copy]
            print(f"Speaker sequence in context: {speaker_sequence}")
            
            # Generate audio using the CSM model
            try:
                print(f"\nCalling model.generate with speaker={speaker}...")
                audio_tensor = self.generator.generate(
                    text=text,
                    speaker=speaker,
                    context=context_copy,
                    max_audio_length_ms=10_000,
                )
                print(f"Generated raw audio tensor: {audio_tensor.shape}")
                if torch.isnan(audio_tensor).any() or torch.isinf(audio_tensor).any():
                    print("WARNING: Generated audio contains NaN or Inf values!")
                else:
                    print(f"Audio range: ({audio_tensor.min():.4f}, {audio_tensor.max():.4f})")
            except Exception as e:
                print(f"Error in model.generate: {e}")
                traceback.print_exc()
                # Return a small dummy tensor on error
                return torch.zeros(target_sample_rate // 2, dtype=torch.int16, device=self.device)

            # Ensure the audio tensor is on the correct device before processing
            if audio_tensor.device != self.device:
                audio_tensor = audio_tensor.to(self.device)

            # Resample to target sample rate
            print("\nResampling audio...")
            audio_tensor = torchaudio.functional.resample(
                audio_tensor.squeeze(0), orig_freq=self.generator.sample_rate, new_freq=target_sample_rate
            )

            # Convert float audio to int16 format for output
            print("Converting to int16...")
            audio_tensor = audio_tensor * 32767.0
            audio_tensor = torch.clamp(audio_tensor, -32768, 32767)
            audio_tensor = audio_tensor.to(torch.int16)

            print("=== Generation Complete ===\n")
            return audio_tensor
            
        except Exception as e:
            print(f"Error generating audio: {e}")
            traceback.print_exc()
            # Return a small dummy tensor on error
            return torch.zeros(target_sample_rate // 2, dtype=torch.int16, device=self.device)