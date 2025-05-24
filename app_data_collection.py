import os
import uuid
import json # Added for metadata
import wave   # Added for metadata
from datetime import datetime
from flask import Flask, render_template, request, jsonify

# Initialize Flask application
app = Flask(__name__)

# Define the upload directory path
UPLOAD_FOLDER = 'collected_audio'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
METADATA_FILE = os.path.join(UPLOAD_FOLDER, 'metadata.jsonl') # Define metadata file path

# Define route for the root URL
@app.route('/')
def index():
    return render_template('recorder.html')

# Define the /upload_audio endpoint
@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'audio_file' not in request.files:
        return jsonify({'success': False, 'message': 'No audio file part in the request'}), 400
    
    file = request.files['audio_file']
    
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'}), 400
        
    if file:
        try:
            # Generate a unique filename. Saving as .wav as per instructions.
            filename = str(uuid.uuid4()) + '.wav'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file.save(filepath)
            
            # --- Add Metadata ---
            duration = 0.0
            framerate = 0
            n_channels = 0
            sampwidth = 0
            
            try:
                with wave.open(filepath, 'rb') as wf:
                    n_frames = wf.getnframes()
                    framerate = wf.getframerate()
                    n_channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    duration = n_frames / float(framerate)
            except wave.Error as e:
                app.logger.error(f"Error processing WAV file {filename}: {e}. File might not be a valid WAV or is corrupted.")
                # Keep default values for audio properties
            except Exception as e: # Catch other potential errors
                app.logger.error(f"Unexpected error processing WAV file {filename}: {e}")
                # Keep default values for audio properties

            metadata = {
                'recording_id': str(uuid.uuid4()), # Unique ID for this metadata entry
                'audio_filename': filename,
                'timestamp': datetime.utcnow().isoformat() + 'Z', # UTC timestamp
                'duration_seconds': duration,
                'sample_rate_hz': framerate,
                'channels': n_channels,
                'sample_width_bytes': sampwidth,
                'file_format': 'wav' # Assuming WAV as per save operation
            }
            
            try:
                with open(METADATA_FILE, 'a') as mf:
                    mf.write(json.dumps(metadata) + '\n')
            except Exception as e:
                app.logger.error(f"Error writing metadata for {filename}: {e}")
                # Note: Upload is still considered successful if audio is saved,
                # but metadata writing failure is logged.
            # --- End Metadata ---
            
            return jsonify({'success': True, 'message': 'Audio uploaded successfully', 'filename': filename})
        except Exception as e:
            # Log the exception for server-side debugging
            app.logger.error(f"Error saving file or processing metadata: {e}") 
            return jsonify({'success': False, 'message': 'Error saving file on server'}), 500
    
    # Fallback for any other unexpected case.
    return jsonify({'success': False, 'message': 'Unknown error during file upload'}), 500

# Standard Flask development server startup
if __name__ == '__main__':
    # Ensure UPLOAD_FOLDER exists at runtime
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
        print(f"Created directory: {UPLOAD_FOLDER}")
    app.run(debug=True)
