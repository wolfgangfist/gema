# CSM-WebUI

This is a fork of the original CSM project that creates a complete Gradio-based web interface, making CSM accessible through an intuitive UI. The fork adds Windows and WSL compatibility and various usability improvements to make speech generation easy and accessible.

## 🚀 Key Enhancements

- **Windows Compatibility**: Full support for Windows with triton-windows package
- **WSL Support**: Dedicated setup script for Linux/WSL environments
- **Model Format**: Updated to use safetensors format (more efficient and secure)
- **Model Flexibility**: Support for both local models and HuggingFace-hosted models
- **Improved Local Model Loading**: Fixed issues with Llama-3.2-1B model loading
- **Windows-Specific Generator**: Separate generator file for Windows environments
- **Robust Setup Scripts**: Comprehensive setup for both Windows and WSL
- **Git Integration**: Optional Git-based model downloading for better reliability

## 📋 Technical Details

### Model Loading Improvements

The original project had issues loading the models because:

1. The models are gated on HuggingFace and require authentication
2. The Windows version had compatibility issues
3. The tokenizer wasn't configured to use local files

Our solution:
- Use the non-gated [unsloth/Llama-3.2-1B](https://huggingface.co/unsloth/Llama-3.2-1B) version
- Create a Windows-specific generator with proper path handling
- Add `local_files_only=True` parameter to prevent download attempts
- Provide multiple fallback methods for model loading
- Create a standalone fix tool that can be run anytime

### Package Compatibility

Our setup ensures compatible versions of key packages:

- **Windows**:
  - triton-windows instead of triton
  - numpy==1.26.4
  - scipy==1.11.4
  - torch with CUDA 11.8 support

- **WSL/Linux**:
  - Standard triton package
  - Compatible numpy and scipy versions

## 📦 Model Information

The project uses two models:
- **CSM-1B**: Non-gated SAFETENSORS file Available at [drbaph/CSM-1B](https://huggingface.co/drbaph/CSM-1B/tree/main)
- **Llama-3.2-1B**: Non-gated version available at [unsloth/Llama-3.2-1B](https://huggingface.co/unsloth/Llama-3.2-1B/tree/main)

Models are stored in specific directories:
- Windows: `models/model.safetensors` and `models/llama/*`
- WSL: Same structure, but will default to the original model paths if not found locally

## ⚙️ Installation

### Windows Installation

```batch
# Clone the repository
git clone https://github.com/YOUR_USERNAME/CSM-WebUI.git
cd CSM-WebUI

# Run the Windows setup script
win-setup.bat

# After setup completes, run the application with:
run_win_gradio.bat  # Uses Windows-specific generator
# OR
run_gradio.bat      # Uses standard generator
```

### WSL/Linux Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/CSM-WebUI.git
cd CSM-WebUI

# Run the WSL setup script
bash wsl-setup.sh

# After setup completes, run the application
python gradio.py
```

## 🔧 Troubleshooting

### Windows Model Loading Issues

If you encounter model loading errors on Windows:

```batch
# Run the fix tool
python fix_win_model.py

# Then run the Windows-specific batch file
run_win_gradio.bat
```

The fix tool will:
1. Create a Windows-specific generator file (win_generator.py)
2. Modify it to use local model files
3. Create a specialized batch file to run with this generator

### WSL/Linux Model Loading Issues

If you encounter model loading errors on WSL/Linux:

```bash
# Make sure you're using the correct paths
python gradio.py --model-path models/csm/model.safetensors
```

## 🔄 Switching Between Windows and WSL

This fork is designed to let you use both environments without conflicts:
- Windows will use the triton-windows package and a Windows-specific generator file
- WSL will use the standard triton package and the original generator file

## 🗂️ Directory Structure

```
CSM-WebUI/
├── models/
│   ├── csm/              # CSM model files (where setup scripts save model)
│   │   └── model.safetensors
│   └── llama/            # Llama model files
│       ├── model.safetensors
│       ├── config.json
│       ├── generation_config.json
│       ├── special_tokens_map.json
│       ├── tokenizer.json
│       └── tokenizer_config.json
├── generator.py          # Original generator for WSL
├── win_generator.py      # Windows-specific generator
├── gradio.py             # Original UI (for WSL)
├── win-gradio.py         # Windows-specific UI (default model path is "models/model.safetensors")
├── wsl-setup.sh          # Setup script for WSL/Linux
├── win-setup.bat         # Setup script for Windows
├── run_gradio.bat        # Run script for Windows (standard)
├── run_win_gradio.bat    # Run script for Windows (with win_generator)
└── fix_win_model.py      # Utility to fix model loading issues
```

## 💡 Key Differences from Original

1. **Model Storage**: Original required manual downloads, our version automates this
2. **File Format**: Changed from .ckpt to .safetensors for better security and compatibility 
3. **Windows Support**: Added comprehensive Windows support with separate generator file
4. **Dual Environments**: Support for both Windows and WSL without conflicts
5. **Robust Error Handling**: Multiple fallback methods for model loading
6. **Streamlined UI**: Unified interface across platforms
7. **Non-Gated Models**: Using alternative sources to avoid authentication issues

---

# Original README Below

# CSM

**2025/03/13** - We are releasing the 1B CSM variant. The checkpoint is [hosted on Hugging Face](https://huggingface.co/sesame/csm_1b).

---

CSM (Conversational Speech Model) is a speech generation model from [Sesame](https://www.sesame.com) that generates RVQ audio codes from text and audio inputs. The model architecture employs a [Llama](https://www.llama.com/) backbone and a smaller audio decoder that produces [Mimi](https://huggingface.co/kyutai/mimi) audio codes.

A fine-tuned variant of CSM powers the [interactive voice demo](https://www.sesame.com/voicedemo) shown in our [blog post](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice).

A hosted [Hugging Face space](https://huggingface.co/spaces/sesame/csm-1b) is also available for testing audio generation.

## Requirements

* A CUDA-compatible GPU
* The code has been tested on CUDA 12.4 and 12.6, but it may also work on other versions
* Similarly, Python 3.10 is recommended, but newer versions may be fine
* For some audio operations, `ffmpeg` may be required
* Access to the following Hugging Face models:
  * [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
  * [CSM-1B](https://huggingface.co/sesame/csm-1b)

### Setup

```bash
git clone git@github.com:SesameAILabs/csm.git
cd csm
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows Setup

The `triton` package cannot be installed in Windows. Instead use `pip install triton-windows`.

## Usage

Generate a sentence

```python
from huggingface_hub import hf_hub_download
from generator import load_csm_1b
import torchaudio
import torch

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
generator = load_csm_1b(model_path, device)
audio = generator.generate(
    text="Hello from Sesame.",
    speaker=0,
    context=[],
    max_audio_length_ms=10_000,
)

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

CSM sounds best when provided with context. You can prompt or provide context to the model using a `Segment` for each speaker's utterance.

```python
speakers = [0, 1, 0, 0]
transcripts = [
    "Hey how are you doing.",
    "Pretty good, pretty good.",
    "I'm great.",
    "So happy to be speaking to you.",
]
audio_paths = [
    "utterance_0.wav",
    "utterance_1.wav",
    "utterance_2.wav",
    "utterance_3.wav",
]

def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    return audio_tensor

segments = [
    Segment(text=transcript, speaker=speaker, audio=load_audio(audio_path))
    for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
]
audio = generator.generate(
    text="Me too, this is some cool stuff huh?",
    speaker=1,
    context=segments,
    max_audio_length_ms=10_000,
)

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

## FAQ

**Does this model come with any voices?**

The model open-sourced here is a base generation model. It is capable of producing a variety of voices, but it has not been fine-tuned on any specific voice.

**Can I converse with the model?**

CSM is trained to be an audio generation model and not a general-purpose multimodal LLM. It cannot generate text. We suggest using a separate LLM for text generation.

**Does it support other languages?**

The model has some capacity for non-English languages due to data contamination in the training data, but it likely won't do well.

## Misuse and abuse ⚠️

This project provides a high-quality speech generation model for research and educational purposes. While we encourage responsible and ethical use, we **explicitly prohibit** the following:

- **Impersonation or Fraud**: Do not use this model to generate speech that mimics real individuals without their explicit consent.
- **Misinformation or Deception**: Do not use this model to create deceptive or misleading content, such as fake news or fraudulent calls.
- **Illegal or Harmful Activities**: Do not use this model for any illegal, harmful, or malicious purposes.

By using this model, you agree to comply with all applicable laws and ethical guidelines. We are **not responsible** for any misuse, and we strongly condemn unethical applications of this technology.

---

## Authors
Johan Schalkwyk, Ankit Kumar, Dan Lyth, Sefik Emre Eskimez, Zack Hodari, Cinjon Resnick, Ramon Sanabria, Raven Jiang, and the Sesame team.
