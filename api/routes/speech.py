import io
from run_api import app

from fastapi import HTTPException, Depends
from fastapi.responses import Response

from pydantic import BaseModel

from api.authentication import verify_api_key
#---------------------- end of headers ----------------------


class SpeechRequest(BaseModel):
    text: str = "Hello Josh"
    speaker: int = 0
    max_audio_length: int = 10000

@app.post("/speech")
def generate_speech(req: SpeechRequest, _: str = Depends(verify_api_key)):
    print(f"🎙️ Generating speech for: '{req.text}' with speaker {req.speaker}")
    try:
        # Generate audio
        #audio = generator.generate(
        #    text=req.text,
        #    speaker=req.speaker,
        #    context=[],
        #    max_audio_length_ms=req.max_audio_length,
        #)

        buffer = io.BytesIO()
        #torchaudio.save(buffer, audio.unsqueeze(0).cpu(), generator.sample_rate, format="wav")
        buffer.seek(0)

        print(f"✅ Audio generated successfully")

        return Response(
            content=buffer.read(),
            media_type="audio/wav",
            headers={
                "Content-Type": "audio/wav",
                "Content-Disposition": "inline; filename=output.wav",
                "Cache-Control": "no-cache",
                "Content-Length": str(buffer.getbuffer().nbytes)
            }
        )

    except Exception as e:
        print(f"❌ Error generating audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))