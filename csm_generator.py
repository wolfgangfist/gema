

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
    def __init__(self):
        # Select the best available device, skipping MPS due to float64 limitations
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        print(f"Using device: {device}")

        # Load model
        self.generator = load_csm_1b(device)

        audio_tensor = self.load_prompt_audio(SPEAKER_PROMPTS["conversational_a"]["audio"], self.generator.sample_rate)
        self.defaultSegment = [ Segment(text=SPEAKER_PROMPTS["conversational_a"]["text"], speaker=0, audio=audio_tensor) ]
        self.segments = {}
        self.inputBuffer = {}

    def generate(self, text: str, speaker_id: int, context_id: str, sample_rate: int = 24000, eos: bool = False) -> torch.Tensor:
        audio_tensor = None
        if context_id not in self.segments:
            self.segments[context_id] = self.defaultSegment.copy()

        if context_id not in self.inputBuffer:
            self.inputBuffer[context_id] = text if text else ""
        elif text:
            self.inputBuffer[context_id] += text

        endWithSpaceOrPunctuation = len(self.inputBuffer[context_id]) > 64 and self.inputBuffer[context_id][-1] in [" ", ".", ",", "!", "?"]
        eosAndBufferNotEmpty = eos and (len(self.inputBuffer[context_id].strip()) > 0)

        if eosAndBufferNotEmpty or endWithSpaceOrPunctuation:
            print(f"Generating audio for context {context_id} with \"{self.inputBuffer[context_id]}\"")
            inputText = self.inputBuffer[context_id]

            audio_tensor = self._generate(
                text=inputText,
                speaker=speaker_id,
                context=self.segments[context_id],
                target_sample_rate=sample_rate,
            )

            self.inputBuffer[context_id] = ""

        return audio_tensor

    def load_prompt_audio(self, audio_path: str, target_sample_rate: int) -> torch.Tensor:
        audio_tensor, sample_rate = torchaudio.load(audio_path)
        audio_tensor = audio_tensor.squeeze(0)
        # Resample is lazy so we can always call it
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
        )

        return audio_tensor

    def _generate(self, text: str, speaker: int, context: list, target_sample_rate: int) -> torch.Tensor:
            audio_tensor = self.generator.generate(
                text=text,
                speaker=speaker,
                context=context,
                max_audio_length_ms=10_000,
            )

            audio_tensor = torchaudio.functional.resample(
                audio_tensor.squeeze(0), orig_freq=self.generator.sample_rate, new_freq=target_sample_rate
            )

            #context.append(Segment(text=text, speaker=speaker, audio=audio_tensor))

            # multiply by 32767 to convert to int16
            audio_tensor = audio_tensor * 32767.0
            # clip to int16 range
            audio_tensor = torch.clamp(audio_tensor, -32768, 32767)
            # convert to int16
            audio_tensor = audio_tensor.to(torch.int16)

            return audio_tensor
