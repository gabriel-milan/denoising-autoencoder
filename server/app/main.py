import io
import json
import librosa
import requests
import numpy as np
from flask_cors import CORS
from flask_limiter import Limiter
from scipy.io.wavfile import write
from flask_limiter.util import get_remote_address
from flask import Flask, render_template, request, make_response

# Configuration
WINDOW_SIZE = 320


# Scaler
class BitScaler:
    def __init__(self, bits):
        self._offset: int = 2 ** (bits - 1)
        self._scale: int = 2 ** (-bits)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x + self._offset) * self._scale

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x / self._scale - self._offset


scaler = BitScaler(16)

# Create a Flask app
app = Flask(__name__)

# Enable CORS for server
cors = CORS(app, resources={r"/*": {"origins": "*"}})

# Adding rate limits for requests
# so we won't get spammed and suffer
# from DDoS
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["1000 per day", "100 per hour"]
)


# Homepage
@app.route('/')
def index():
    return render_template('index.html')


# Here we do our audio handling
@app.route('/audio', methods=['POST'])
@limiter.limit("4 per minute")
def audio():
    # Parse input audio to array
    data, rate = librosa.load(io.BytesIO(request.data), sr=16000)
    # Fix data type
    data = (data * (2 ** 16)).astype(np.int16)
    # Trim array
    data = data[:data.shape[0] - data.shape[0] % WINDOW_SIZE]
    # Reshape array for prediction
    data = data.reshape(data.shape[0] // WINDOW_SIZE, WINDOW_SIZE)
    # Scale data
    data = scaler.transform(data)
    # Request to model deployment
    req_data = json.dumps({"signature_name": "serving_default",
                           "instances": data.tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post(
        'http://model_deployment:8501/v1/models/test:predict', data=req_data, headers=headers)
    predictions = np.array(json.loads(json_response.text)['predictions'])
    # Inverse scale predictions
    predictions = scaler.inverse_transform(predictions)
    # Flatten output
    predictions = predictions.flatten().astype(np.int16)
    # Write to IO stream
    bytes_wav = bytes()
    byte_io = io.BytesIO(bytes_wav)
    write(byte_io, rate, predictions)
    response = make_response(byte_io.read())
    response.headers['Content-Type'] = 'audio/wav'
    response.headers['Content-Disposition'] = 'attachment; filename=output.wav'
    return response


if __name__ == "__main__":
    app.run(debug=True)
