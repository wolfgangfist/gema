let ws;
let micAnalyser, micContext, micSource, micStream;
let outputAnalyser, outputAudioCtx;
let lastConfig = null;

document.addEventListener('DOMContentLoaded', async () => {
  await populateAudioDevices();

  ws = new WebSocket(`ws://${window.location.host}/ws`);

  ws.onopen = () => {
    console.log("WebSocket connected, requesting saved config...");
    ws.send(JSON.stringify({ type: "request_saved_config" }));
  };

  ws.onmessage = async (event) => {
    const data = JSON.parse(event.data);

    if (data.type === "saved_config" && data.config) {
      document.getElementById('systemPrompt').value = data.config.system_prompt || "";
      document.getElementById('modelPath').value = data.config.model_path || "";
      document.getElementById('llmPath').value = data.config.llm_path || "";
      document.getElementById('referenceAudio').value = data.config.reference_audio_path || "";
      document.getElementById('referenceText').value = data.config.reference_text || "";

      setTimeout(() => {
        if (data.config.mic_id) document.getElementById('micSelect').value = data.config.mic_id;
        if (data.config.output_id) document.getElementById('outputSelect').value = data.config.output_id;
      }, 500);
    }

    if (data.type === "status" && data.message.includes("Models initialized")) {
      console.log("Model initialization confirmed. Redirecting...");
    
      // Save config again just to be safe
      localStorage.setItem('ai_config', JSON.stringify(lastConfig));
    
      // Close WebSocket before navigating
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    
      // Wait briefly to let server clean up, then redirect
      setTimeout(() => {
        window.location.href = "/chat";
      }, 100);
    }
    
  };

  document.getElementById('testMicBtn').addEventListener('click', async () => {
    const micId = getSelectedMic();
    micStream = await navigator.mediaDevices.getUserMedia({ audio: { deviceId: micId } });

    micContext = new AudioContext();
    micSource = micContext.createMediaStreamSource(micStream);
    micAnalyser = micContext.createAnalyser();
    micSource.connect(micAnalyser);
    visualizeMic(micAnalyser, 'micCanvas');

    const recorder = new MediaRecorder(micStream);
    const chunks = [];

    recorder.ondataavailable = e => {
      if (e.data.size > 0) chunks.push(e.data);
    };

    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: 'audio/webm' });
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audio.play();

      micStream.getTracks().forEach(track => track.stop());
      micContext.close();
    };

    recorder.start();
    setTimeout(() => recorder.stop(), 3000);
  });

  document.getElementById('testOutputBtn').addEventListener('click', () => {
    const audio = new Audio('/static/test.mp3');
    audio.setSinkId(getSelectedOutput()).then(() => {
      outputAudioCtx = new AudioContext();
      const outputSource = outputAudioCtx.createMediaElementSource(audio);
      outputAnalyser = outputAudioCtx.createAnalyser();
      outputSource.connect(outputAnalyser);
      outputAnalyser.connect(outputAudioCtx.destination);
      visualizeMic(outputAnalyser, 'outputCanvas');
      audio.play();
    }).catch(err => {
      console.warn("Failed to route output:", err);
    });
  });

  document.getElementById('saveAndStart').addEventListener('click', () => {
    lastConfig = {
      system_prompt: document.getElementById('systemPrompt').value,
      model_path: document.getElementById('modelPath').value,
      llm_path: document.getElementById('llmPath').value,
      reference_audio_path: document.getElementById('referenceAudio').value,
      reference_text: document.getElementById('referenceText').value,
      mic_id: getSelectedMic(),
      output_id: getSelectedOutput(),
    };
    console.log("Sending config to backend...");
    ws.send(JSON.stringify({ type: "config", config: lastConfig }));
    // we wait for the backend to reply with model status before navigating
  });
});

function getSelectedMic() {
  return document.getElementById('micSelect').value;
}

function getSelectedOutput() {
  return document.getElementById('outputSelect').value;
}

async function populateAudioDevices() {
  try {
    await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    console.warn("Microphone permission denied or not granted.");
    return;
  }

  const devices = await navigator.mediaDevices.enumerateDevices();
  const micSelect = document.getElementById('micSelect');
  const outputSelect = document.getElementById('outputSelect');

  micSelect.innerHTML = '';
  outputSelect.innerHTML = '';

  devices.forEach(device => {
    const option = new Option(device.label || `${device.kind}`, device.deviceId);
    if (device.kind === 'audioinput') micSelect.add(option.cloneNode(true));
    if (device.kind === 'audiooutput') {
      outputSelect.add(option.cloneNode(true));
    }
  });

  if (micSelect.options.length === 0) {
    micSelect.add(new Option("No mic devices found", ""));
  }
  if (outputSelect.options.length === 0) {
    outputSelect.add(new Option("Default Output", "default"));
  }
}

function visualizeMic(analyser, canvasId) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d");
  analyser.fftSize = 256;
  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  function draw() {
    requestAnimationFrame(draw);
    analyser.getByteFrequencyData(dataArray);
    ctx.fillStyle = "#1f2937";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const barWidth = canvas.width / bufferLength;
    for (let i = 0; i < bufferLength; i++) {
      const barHeight = dataArray[i];
      ctx.fillStyle = "#4ade80";
      ctx.fillRect(i * barWidth, canvas.height - barHeight / 2, barWidth - 1, barHeight / 2);
    }
  }
  draw();
}
