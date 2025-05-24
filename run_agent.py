import os
# Assuming NO_TORCH_COMPILE was needed for run_csm.py to prevent Triton/compiler issues.
# Set this before importing torch.
os.environ["NO_TORCH_COMPILE"] = "1"

import torch
import torchaudio

# Assuming generator.py is in the same directory or accessible in PYTHONPATH
from generator import load_csm_1b, Segment 
from agent import VoiceAgent # Assuming agent.py is in the same directory

def main():
    """
    Main function to run the VoiceAgent and generate speech responses.
    """
    # 1. Setup Device
    if torch.cuda.is_available():
        device = "cuda"
    # elif torch.backends.mps.is_available(): # Optional: MPS can be slower or have issues
    #     device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # 2. Load CSM Generator
    print("Loading CSM model...")
    try:
        generator = load_csm_1b(device)
        print("CSM model loaded successfully.")
    except Exception as e:
        print(f"Error loading CSM model: {e}")
        print("Please ensure that the model weights are available and generator.py is correctly set up.")
        print("If you are running this in an environment where the model was downloaded by run_csm.py, it should be found.")
        print("Try running `python -m generator` if you suspect model download issues.")
        return

    # 3. Initialize Agent
    agent = VoiceAgent()
    print("VoiceAgent initialized.")

    # 4. User Guidance
    print("\nStarting interactive agent. Type 'exit' or 'quit' to end.")
    print("-" * 30)

    # 5. Interactive Processing Loop
    while True:
        try:
            command_text = input("You: ")
        except KeyboardInterrupt: # Allow Ctrl+C to exit gracefully
            print("\nExiting agent...")
            break

        if command_text.lower() in ["exit", "quit"]:
            print("Exiting agent...")
            break

        if not command_text.strip(): # Handle empty input
            print("Agent: Please enter a command.")
            print("-" * 30)
            continue

        # b. Get the text response from the agent
        text_response = agent.process_command(command_text)
        print(f"Agent: {text_response}")

        # d. Generate Speech
        # Avoid speaking generic "I don't understand" messages if desired,
        # but for now, we'll attempt to speak all non-empty responses.
        if not text_response or text_response.strip() == "":
            print("Agent response is empty, skipping audio generation.")
            print("-" * 30)
            continue
        
        # A more specific check if we want to avoid speaking "I don't understand"
        # generic_error_message = "Sorry, I didn't understand that." 
        # if generic_error_message in text_response:
        #     print("Skipping audio for generic unrecognized command.")
        #     print("-" * 30)
        #     continue

        print("Generating audio...")
        try:
            audio_tensor = generator.generate(
                text=text_response,
                speaker=0,  # Default speaker
                context=[], # No context for simplicity
                max_audio_length_ms=20_000 # Increased slightly
            )
        except Exception as e:
            print(f"Error during audio generation: {e}")
            print("-" * 30)
            continue
        
        # e. Save Audio
        output_filename = "last_agent_response.wav"
        try:
            # Ensure the audio tensor is 2D [channels, samples] for torchaudio.save
            # and move to CPU if it's not already.
            audio_to_save = audio_tensor.unsqueeze(0).cpu()
            torchaudio.save(output_filename, audio_to_save, generator.sample_rate)
            print(f"Saved audio response to {output_filename}")
        except Exception as e:
            print(f"Error saving audio file {output_filename}: {e}")
        
        print("-" * 30) # Separator

    print("\nFinished interactive session.")

# 6. Add an if __name__ == "__main__": block
if __name__ == "__main__":
    main()
