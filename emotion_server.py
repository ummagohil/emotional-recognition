# emotion_server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from deepface import DeepFace  # Requires pandas internally [4]
import base64

app = Flask(__name__)
CORS(app)

@app.route('/analyze', methods=['POST'])
def analyze_emotion():
    try:
        img_data = request.json['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        dominant_emotion = results[0]['dominant_emotion']
        
        return jsonify({
            'emotion': dominant_emotion,
            'confidence': float(results[0]['emotion'][dominant_emotion])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
