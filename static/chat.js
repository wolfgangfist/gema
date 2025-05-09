let ws;
let sessionStartTime   = null;
let messageCount       = 0;
let audioLevelsChart   = null;
let isRecording        = false;
let isAudioCurrentlyPlaying = false;
let configSaved        = false;

const SESSION_ID       = "default";
console.log("chat.js loaded");

let audioContext;
let micStream;
let selectedMicId      = null;
let selectedOutputId   = null;

let audioPlaybackQueue = [];
let audioDataHistory   = [];       
let micAnalyser, micContext;

function createPermanentVoiceCircle() {
  if (document.getElementById('voice-circle')) return;         
  const style = document.createElement('style');
  style.textContent = `
    #voice-circle{
      position:fixed;top:50%;left:50%;
      width:180px;height:180px;border-radius:50%;
      background:rgba(99,102,241,.20);
      transform:translate(-50%,-50%) scale(var(--dynamic-scale,1));
      pointer-events:none;z-index:50;
      transition:background-color .35s ease;
    }
    #voice-circle.active{
      animation:pulse-circle 2s infinite alternate ease-in-out;
    }
    @keyframes pulse-circle{
      0%{background:rgba(99,102,241,.55)}
      100%{background:rgba(99,102,241,.20)}
    }`;
  document.head.appendChild(style);

  const c = document.createElement('div');
  c.id='voice-circle';
  document.body.appendChild(c);
  console.log("Created permanent voice circle");
}
function showVoiceCircle() {
  const c=document.getElementById('voice-circle')||createPermanentVoiceCircle();
  c.classList.add('active');
}
function hideVoiceCircle() {
  const c=document.getElementById('voice-circle');
  if (c){ c.classList.remove('active'); c.style.setProperty('--dynamic-scale',1); }
}

function showNotification(msg, type='info'){
  const n=document.createElement('div');
  n.className=`fixed bottom-4 right-4 px-4 py-3 rounded-lg shadow-lg z-50
               ${type==='success'?'bg-green-600':
                 type==='error'  ?'bg-red-600':'bg-indigo-600'}`;
  n.textContent=msg;
  document.body.appendChild(n);
  setTimeout(()=>{n.classList.add('opacity-0');
                  setTimeout(()=>n.remove(),500)},3000);
}

function addMessageToConversation(sender,text){
  const pane=document.getElementById('conversationHistory');
  if(!pane) return;
  const box=document.createElement('div');
  box.className=`p-3 mb-3 rounded-lg text-sm ${
            sender==='user'?'bg-gray-800 ml-2':'bg-indigo-900 mr-2'}`;
  box.innerHTML=`
      <div class="flex items-start mb-2">
        <div class="w-6 h-6 rounded-full flex items-center justify-center
             ${sender==='user'?'bg-gray-300 text-gray-800':'bg-indigo-500 text-white'}">
             ${sender==='user'?'U':'AI'}
        </div>
        <span class="text-xs text-gray-400 ml-2">${new Date().toLocaleTimeString()}</span>
      </div>
      <div class="text-white mt-1 text-sm">${text
            .replace(/&/g,'&amp;').replace(/</g,'&lt;')
            .replace(/\*\*(.*?)\*\*/g,'<strong>$1</strong>')
            .replace(/\*(.*?)\*/g,'<em>$1</em>')
            .replace(/```([^`]+)```/g,'<pre><code>$1</code></pre>')
            .replace(/`([^`]+)`/g,'<code>$1</code>')
            .replace(/\n/g,'<br>')}</div>`;
  pane.appendChild(box);
  pane.scrollTop=pane.scrollHeight;
}

function connectWebSocket(){
  if(ws) ws.close();
  const proto=location.protocol==='https:'?'wss:':'ws:';
  ws=new WebSocket(`${proto}//${location.host}/ws`);
  window.ws=ws;

  const connLbl=document.getElementById('connectionStatus');
  connLbl.textContent='Connecting…'; connLbl.className='text-yellow-500';

  ws.onopen=()=>{
    connLbl.textContent='Connected'; connLbl.className='text-green-500';
    ws.send(JSON.stringify({type:'request_saved_config'}));
    addMessageToConversation('ai','WebSocket connected. Ready for voice or text.');
  };
  ws.onclose = ()=>{connLbl.textContent='Disconnected'; connLbl.className='text-red-500';
                    setTimeout(connectWebSocket,5000);};
  ws.onerror = ()=>{connLbl.textContent='Error'; connLbl.className='text-red-500';};

  ws.onmessage = e=>{
    try{handleWebSocketMessage(JSON.parse(e.data));}catch(err){console.error(err);}
  };
}

function sendTextMessage(txt){
  if(!txt.trim()) return;
  if(!ws || ws.readyState!==WebSocket.OPEN){
      showNotification("Not connected","error"); return;
  }
  showVoiceCircle();
  ws.send(JSON.stringify({type:'text_message',text:txt,session_id:SESSION_ID}));
  const cnt=document.getElementById('messageCount');
  if(cnt) cnt.textContent=++messageCount;
  document.getElementById('textInput').value='';
}

function handleWebSocketMessage(d){
  switch(d.type){
    case 'transcription':
      addMessageToConversation('user',d.text);
      showVoiceCircle();
      break;
    case 'response':
      addMessageToConversation('ai',d.text);
      showVoiceCircle();
      break;
    case 'audio_chunk':
      queueAudioForPlayback(d.audio,d.sample_rate);
      showVoiceCircle();
      break;
    case 'audio_status':
      if (d.status === 'generating') {
          showVoiceCircle();
      } else if (d.status === 'complete' || d.status === 'interrupted') {
          if (!isAudioCurrentlyPlaying) {
              hideVoiceCircle();
          }
      }
      break;
    case 'status':
      if(d.message==='Thinking...') showVoiceCircle();
      break;
    case 'error':
      showNotification(d.message,'error');
      hideVoiceCircle();
      break;
  }
}

async function startRecording() {
  if (isRecording) return;
  try {
    const constraints = {
      audio: selectedMicId ? {deviceId:{exact:selectedMicId}} : true
    };
    micStream = await navigator.mediaDevices.getUserMedia(constraints);

    if (!audioContext) audioContext = new (AudioContext||webkitAudioContext)();
    const src   = audioContext.createMediaStreamSource(micStream);
    const proc  = audioContext.createScriptProcessor(4096,1,1);
    src.connect(proc); proc.connect(audioContext.destination);

    proc.onaudioprocess = e => {
      const samples = Array.from(e.inputBuffer.getChannelData(0));
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          type:'audio',
          audio:samples,
          sample_rate:audioContext.sampleRate,
          session_id:SESSION_ID
        }));
      }
    };

    window._micProcessor = proc;        
    isRecording = true;
    document.getElementById('micStatus').textContent = 'Listening…';
    showVoiceCircle();
  } catch (err) {
    console.error(err);
    showNotification('Microphone access denied','error');
  }
}

function stopRecording() {
  if (!isRecording) return;
  try {
    if (window._micProcessor) {
      window._micProcessor.disconnect();
      window._micProcessor = null;
    }
    if (micStream) {
      micStream.getTracks().forEach(t => t.stop());
      micStream = null;
    }
  } catch(e){console.warn(e);}
  isRecording = false;
  document.getElementById('micStatus').textContent = 'Click to speak';
  hideVoiceCircle();
}

function queueAudioForPlayback(arr,sr){
  audioPlaybackQueue.push({arr,sr});
  if(!isAudioCurrentlyPlaying) processAudioPlaybackQueue();
}
function processAudioPlaybackQueue(){
  if(!audioPlaybackQueue.length){isAudioCurrentlyPlaying=false;hideVoiceCircle();return;}
  isAudioCurrentlyPlaying=true;
  const {arr,sr}=audioPlaybackQueue.shift();
  playAudioChunk(arr,sr).then(processAudioPlaybackQueue);
}
async function playAudioChunk(audioArr,sampleRate){
  await (window.audioContext?.resume?.()||Promise.resolve());
  if(!window.audioContext) window.audioContext=new (AudioContext||webkitAudioContext)();
  const buf=window.audioContext.createBuffer(1,audioArr.length,sampleRate);
  buf.copyToChannel(new Float32Array(audioArr),0);
  const src=window.audioContext.createBufferSource();src.buffer=buf;
  const an = window.audioContext.createAnalyser(); an.fftSize=256;
  src.connect(an); an.connect(window.audioContext.destination); src.start();

  const arr=new Uint8Array(an.frequencyBinCount);
  const circle=document.getElementById('voice-circle');
  (function pump(){
    an.getByteFrequencyData(arr);
    const avg=arr.reduce((a,b)=>a+b,0)/arr.length;
    circle?.style.setProperty('--dynamic-scale',(1+avg/255*1.5).toFixed(3));
    if(src.playbackState!==src.FINISHED_STATE) requestAnimationFrame(pump);
  })();
  return new Promise(res=>src.onended=res);
}


async function setupChatUI(){

  document.documentElement.classList.add('bg-gray-950');
  document.documentElement.style.backgroundColor='#030712';

  createPermanentVoiceCircle();
  connectWebSocket();

  initAudioLevelsChart();

  const txt=document.getElementById('textInput');
  const btn=document.getElementById('sendTextBtn');
  btn.onclick=()=>sendTextMessage(txt.value);
  txt.addEventListener('keydown',e=>{
      if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendTextMessage(txt.value);}
  });
  const micBtn = document.getElementById('micToggleBtn');
  if (micBtn) {
      micBtn.addEventListener('click', () => {
          if (isRecording) stopRecording();
          else             startRecording();
      });
  }
  window.audioContext=new (AudioContext||webkitAudioContext)();
  ['click','touchstart','keydown'].forEach(ev=>
    document.addEventListener(ev,function unlock(){window.audioContext.resume();
       document.removeEventListener(ev,unlock);}));

  console.log("Chat UI ready");
}

if(document.readyState==='loading'){
  document.addEventListener('DOMContentLoaded',setupChatUI);
}else{
  setupChatUI();
}


function initAudioLevelsChart(){
  const ctx=document.getElementById('audioLevels');
  if(!ctx) return;
  if(audioLevelsChart) audioLevelsChart.destroy();
  const grad=ctx.getContext('2d').createLinearGradient(0,0,0,100);
  grad.addColorStop(0,'rgba(79,70,229,.6)');
  grad.addColorStop(1,'rgba(79,70,229,.1)');
  audioLevelsChart=new Chart(ctx,{type:'line',
    data:{labels:Array(30).fill(''),datasets:[{data:Array(30).fill(0),
      backgroundColor:grad,borderColor:'rgba(99,102,241,1)',borderWidth:2,
      tension:.4,fill:true,pointRadius:0}]},
    options:{animation:false,responsive:true,scales:{y:{beginAtZero:true,max:100,
      ticks:{display:false},grid:{color:'rgba(255,255,255,.1)'}},
      x:{display:false,grid:{display:false}}},plugins:{legend:{display:false},
      tooltip:{enabled:false}},elements:{point:{radius:0}}}});
}

