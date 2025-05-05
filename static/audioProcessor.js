// This is the AudioWorklet processor
class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
      super();
      this.bufferSize = 4096;
      this.buffer = new Float32Array(this.bufferSize);
      this.bufferIndex = 0;
    }
  
    process(inputs, outputs, parameters) {
      // Get input data from first input, first channel
      const input = inputs[0][0];
      
      if (!input) return true;
  
      // Copy input data to our buffer
      for (let i = 0; i < input.length; i++) {
        this.buffer[this.bufferIndex++] = input[i];
        
        // If buffer is full, send it and reset
        if (this.bufferIndex >= this.bufferSize) {
          // Clone the buffer and send it
          const audioData = this.buffer.slice(0);
          this.port.postMessage({ audioData });
          
          // Reset buffer index
          this.bufferIndex = 0;
        }
      }
      
      // Return true to keep the processor running
      return true;
    }
  }
  
  // Register the processor
  registerProcessor('audio-processor', AudioProcessor);