import asyncio
import base64
import websockets
import json
import os

from api import Request, RequestOperation, Response, ResponseType
from csm_generator import Generator

async def echo(websocket):
    async for message in websocket:
        print(f"Received message: {message}")
        request = Request(**json.loads(message))

        if request.operation == RequestOperation.EOS:
            audio = audioGenerator.generate(
                text=request.text,
                speaker_id=0,
                context_id=request.contextId,
                sample_rate=16000,
                eos=True
            )

            if audio is not None:
                # convert audio tensor to bytes
                audio_bytes = audio.unsqueeze(0).cpu().numpy().tobytes()
                audio_bytes_str = base64.b64encode(audio_bytes).decode('utf-8')

                response = Response(
                    type="chunk",
                    contextId=request.contextId,
                    data=audio_bytes_str,
                    timestamps=None,
                    error=None
                )

                await websocket.send(json.dumps(response, default=lambda o: o.__dict__))

            print("End of stream operation received. Closing connection.")
            await websocket.close()
            return
        elif request.operation == RequestOperation.CLEAR:
            pass
        elif request.operation:  # invalid operation
            response = Response(
                type=ResponseType.ERR,
                contextId=request.contextId,
                data=None,
                timestamps=None,
                error="Invalid operation"
            )

            await websocket.send(json.dumps(response, default=lambda o: o.__dict__))
        else:
            audio = audioGenerator.generate(text=request.text, speaker_id=0, context_id=request.contextId, sample_rate=16000)
            if audio is None:
                continue

            # convert audio tensor to bytes
            audio_bytes = audio.unsqueeze(0).cpu().numpy().tobytes()
            audio_bytes_str = base64.b64encode(audio_bytes).decode('utf-8')

            # Process the request and create a response
            response = Response(
                type="chunk",
                contextId=request.contextId,
                data=audio_bytes_str,
                timestamps=None,
                error=None
            )

            await websocket.send(json.dumps(response, default=lambda o: o.__dict__))

async def main():
    global audioGenerator
    audioGenerator = Generator()

    # get port from environment variable
    port = os.getenv("PORT", 8765)

    async with websockets.serve(echo, "localhost", port):
        print("WebSocket server started on ws://localhost:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
