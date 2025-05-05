import asyncio
import base64
import websockets
import json
import os
import torch
import numpy as np
import traceback
import gc  # For garbage collection

from api import Request, RequestOperation, Response, ResponseType
from csm_generator import Generator

def check_cuda_availability():
    """Check CUDA availability and report details"""
    print("\nCUDA Environment Check:")
    print("-----------------------")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        try:
            device_count = torch.cuda.device_count()
            print(f"CUDA device count: {device_count}")
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                print(f"CUDA Device {i}: {props.name}")
                print(f"  Memory: {props.total_memory / (1024**3):.2f} GB")
                print(f"  Compute capability: {props.major}.{props.minor}")
            
            # Do a simple CUDA operation to verify it's working
            x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
            y = x + 1.0
            print("CUDA test operation successful")
            return "cuda"
        except Exception as e:
            print(f"Error during CUDA test: {e}")
            print("Falling back to CPU")
            return "cpu"
    else:
        print("CUDA is not available - using CPU instead")
        return "cpu"

def prepare_audio_response(audio, context_id):
    """Prepare audio tensor for sending over WebSocket"""
    if audio is None:
        print("Warning: Received None audio tensor")
        # Return a dummy response
        response = Response(
            type=ResponseType.ERR,
            contextId=context_id,
            data=None,
            metadata=None,
            timestamps=None,
            error="Failed to generate audio: empty result"
        )
        return response
    
    try:
        # Ensure audio is on CPU and detached
        if audio.requires_grad:
            audio_cpu = audio.cpu().detach()
        else:
            audio_cpu = audio.cpu()
        
        # Safeguard: verify tensor is valid
        if torch.isnan(audio_cpu).any() or torch.isinf(audio_cpu).any():
            print(f"Warning: Audio tensor contains NaN or Inf values for context {context_id}")
            # Create a safe dummy tensor
            audio_cpu = torch.zeros(16000, dtype=torch.int16)
        
        # Convert audio tensor to numpy array
        audio_np = audio_cpu.numpy().astype(np.int16)
        
        # Pack the bytes directly
        audio_bytes = audio_np.tobytes()
        
        # Encode as base64
        audio_bytes_str = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Create metadata for better client-side handling
        audio_metadata = {
            "format": "int16",
            "channels": 1, 
            "sample_rate": 16000,
            "length_samples": audio_np.shape[0],
            "duration_seconds": audio_np.shape[0] / 16000,
            "min_value": int(audio_np.min()),
            "max_value": int(audio_np.max())
        }
        
        print(f"Prepared audio response: {audio_np.shape[0]} samples, "
              f"min={audio_metadata['min_value']}, max={audio_metadata['max_value']}")
        
        # Create response object
        response = Response(
            type=ResponseType.AUDIO_CHUNK,
            contextId=context_id,
            data=audio_bytes_str,
            metadata=audio_metadata,
            timestamps=None,
            error=None
        )
        
        # Explicitly clear references to large objects
        del audio_cpu, audio_np, audio_bytes
        
        return response
    except Exception as e:
        print(f"Error preparing audio response: {e}")
        traceback.print_exc()
        
        # Return error response
        return Response(
            type=ResponseType.ERR,
            contextId=context_id,
            data=None,
            metadata=None,
            timestamps=None,
            error=f"Failed to prepare audio: {str(e)}"
        )

# Function to record user's input in the conversation context
def record_user_input(context_id, text):
    """
    Record the user's input in the conversation context with speaker_id=1
    This doesn't generate audio, just stores the text in the context
    """
    try:
        # Record the user's input with speaker_id=1
        print(f"Recording user input with speaker_id=1: '{text}'")
        audioGenerator.store_input(
            text=text,
            speaker_id=1,  # User is always speaker 1
            context_id=context_id
        )
        return True
    except Exception as e:
        print(f"Error recording user input: {e}")
        traceback.print_exc()
        return False

async def echo(websocket):
    async for message in websocket:
        print(f"Received message: {message}")
        request = None
        try:
            request = Request(**json.loads(message))
            
            if request.operation == RequestOperation.EOS:
                print(f"EOS operation received for context {request.contextId}")
                
                # Only process if there's text in the request
                if request.text and request.text.strip():
                    print(f"Processing final text: '{request.text}' for context {request.contextId}")
                    try:
                        # First record the user's input with speaker_id=1
                        record_user_input(request.contextId, request.text)
                        
                        # Then generate AI response with speaker_id=0
                        audio = audioGenerator.generate(
                            text="",  # Empty text because we're using the EOS to trigger generation
                            speaker_id=0,  # AI is always speaker 0
                            context_id=request.contextId,
                            sample_rate=16000,
                            eos=True
                        )
                        
                        if audio is not None and audio.numel() > 0:
                            response = prepare_audio_response(audio, request.contextId)
                            if response:
                                print(f"Sending final audio response for context {request.contextId}")
                                await websocket.send(json.dumps(response, default=lambda o: o.__dict__))
                                
                            # Clean up tensor references
                            del audio
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            gc.collect()
                        else:
                            print(f"No valid audio generated for EOS operation")
                    except Exception as e:
                        print(f"Error during EOS audio generation: {e}")
                        traceback.print_exc()
                
                # Send acknowledgment for the EOS operation
                ack_response = Response(
                    type=ResponseType.ACK,
                    contextId=request.contextId,
                    data=None,
                    metadata=None,
                    timestamps=None,
                    error=None
                )
                await websocket.send(json.dumps(ack_response, default=lambda o: o.__dict__))
                print(f"Sent EOS acknowledgment for context {request.contextId}")
                    
            elif request.operation == RequestOperation.CLEAR:
                print(f"Clearing context for {request.contextId}")
                # Acknowledge the clear operation
                response = Response(
                    type=ResponseType.ACK,
                    contextId=request.contextId,
                    data=None,
                    metadata=None,
                    timestamps=None,
                    error=None
                )
                await websocket.send(json.dumps(response, default=lambda o: o.__dict__))
                
            elif request.operation:  # invalid operation
                response = Response(
                    type=ResponseType.ERR,
                    contextId=request.contextId,
                    data=None,
                    metadata=None,
                    timestamps=None,
                    error="Invalid operation"
                )

                await websocket.send(json.dumps(response, default=lambda o: o.__dict__))
            else:
                # Only process if there is text
                if request.text and request.text.strip():
                    print(f"Processing text: '{request.text}' for context {request.contextId}")
                    try:
                        # First record the user's input with speaker_id=1
                        record_user_input(request.contextId, request.text)
                        
                        audio = audioGenerator.generate(
                            text=request.text,  # Just pass the input text directly
                            speaker_id=0,  # AI is always speaker 0 
                            context_id=request.contextId,
                            sample_rate=16000
                        )
                        
                        if audio is not None and audio.numel() > 0:
                            response = prepare_audio_response(audio, request.contextId)
                            if response and response.type != ResponseType.ERR:
                                print(f"Sending audio response for context {request.contextId}")
                                await websocket.send(json.dumps(response, default=lambda o: o.__dict__))
                            elif response:
                                print(f"Sending error response: {response.error}")
                                await websocket.send(json.dumps(response, default=lambda o: o.__dict__))
                            else:
                                print("Failed to create any response")
                                error_response = Response(
                                    type=ResponseType.ERR,
                                    contextId=request.contextId,
                                    data=None,
                                    metadata=None,
                                    timestamps=None,
                                    error="Failed to create response"
                                )
                                await websocket.send(json.dumps(error_response, default=lambda o: o.__dict__))
                            
                            # Clean up tensor references
                            del audio
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            gc.collect()
                        else:
                            print(f"No valid audio generated")
                            error_response = Response(
                                type=ResponseType.ERR,
                                contextId=request.contextId,
                                data=None,
                                metadata=None,
                                timestamps=None,
                                error="Failed to generate audio"
                            )
                            await websocket.send(json.dumps(error_response, default=lambda o: o.__dict__))
                    except Exception as e:
                        print(f"Error in audio generation pipeline: {e}")
                        traceback.print_exc()
                        error_response = Response(
                            type=ResponseType.ERR,
                            contextId=request.contextId,
                            data=None,
                            metadata=None,
                            timestamps=None,
                            error=f"Audio generation error: {str(e)}"
                        )
                        await websocket.send(json.dumps(error_response, default=lambda o: o.__dict__))
                else:
                    print("Received empty text request, ignoring")
        except Exception as e:
            print(f"Error processing message: {e}")
            traceback.print_exc()
            try:
                context_id = getattr(request, 'contextId', None) if request else None
                response = Response(
                    type=ResponseType.ERR,
                    contextId=context_id,
                    data=None,
                    metadata=None,
                    timestamps=None,
                    error=f"Server error: {str(e)}"
                )
                await websocket.send(json.dumps(response, default=lambda o: o.__dict__))
            except Exception as err:
                print(f"Could not send error response: {err}")
                traceback.print_exc()

async def main():
    global audioGenerator
    
    print("Starting CSM WebSocket Server...")
    # Check CUDA availability
    device = check_cuda_availability()
    print(f"\nUsing device: {device}\n")
    
    try:
        # Initialize generator with the selected device
        audioGenerator = Generator(device=device)

        # get port from environment variable
        port = int(os.getenv("PORT", 8765))
        host = os.getenv("HOST", "localhost")

        async with websockets.serve(echo, host, port):
            print(f"WebSocket server started on ws://{host}:{port}")
            await asyncio.Future()  # run forever
    except Exception as e:
        print(f"Fatal error in main: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())