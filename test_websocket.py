#!/usr/bin/env python3
"""
Simple test script for the CSM WebSocket server.
This helps verify if the server is functioning correctly without using the HTML interface.
"""

import asyncio
import websockets
import json
import base64
import os
import argparse
import uuid

async def test_csm_websocket():
    """Test the CSM WebSocket server with a simple text request"""
    # Get server details
    host = os.getenv("HOST", "localhost")
    port = int(os.getenv("PORT", 8765))
    
    # Test message
    test_text = "Hello, this is a test message to check if the CSM system is working properly."
    context_id = f"test-{uuid.uuid4()}"
    
    print(f"Connecting to WebSocket server at ws://{host}:{port}...")
    
    try:
        async with websockets.connect(f"ws://{host}:{port}") as websocket:
            print(f"Connected! Sending test message with context ID: {context_id}")
            
            # Send text message
            message = {
                "text": test_text,
                "contextId": context_id
            }
            await websocket.send(json.dumps(message))
            print(f"Sent message: {json.dumps(message)}")
            
            # Wait for response
            print("Waiting for response...")
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=30)
                response_data = json.loads(response)
                
                print("\nReceived response:")
                print(f"  Type: {response_data.get('type')}")
                print(f"  Context ID: {response_data.get('contextId')}")
                print(f"  Error: {response_data.get('error')}")
                
                # Check if we got audio data
                if response_data.get('data'):
                    audio_bytes = base64.b64decode(response_data['data'])
                    print(f"  Audio data: {len(audio_bytes)} bytes")
                    print("✅ Successfully received audio data!")
                
                    # Send EOS
                    eos_message = {
                        "operation": "eos",
                        "contextId": context_id
                    }
                    await websocket.send(json.dumps(eos_message))
                    print(f"Sent EOS message: {json.dumps(eos_message)}")
                    
                    # Wait for EOS acknowledgment
                    eos_response = await asyncio.wait_for(websocket.recv(), timeout=5)
                    eos_data = json.loads(eos_response)
                    print(f"EOS response: {eos_data.get('type')}")
                
                else:
                    print("❌ No audio data received!")
                
            except asyncio.TimeoutError:
                print("❌ Timeout waiting for response!")
            
            print("\nTest completed.")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")

if __name__ == "__main__":
    print("CSM WebSocket Test Client")
    print("=========================\n")
    
    # Create event loop
    asyncio.run(test_csm_websocket())