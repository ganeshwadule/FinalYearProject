from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector

app = Flask(__name__)
CORS(app)

# Initialize the Hand Detector
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1,
                       detectionCon=0.7, minTrackCon=0.5)

# Store canvas state (for drawing)
canvas = np.zeros((600, 800, 3), np.uint8)
canvas.fill(255)  # White background

# Store previous hand position for smooth drawing
prev_position = None

@app.route('/')
def index():
    return "Welcome to the Hand Gesture Drawing Board!"

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global canvas, prev_position
    
    try:
        if 'frame' not in request.form:
            return jsonify({"error": "No frame received"}), 400

        frame_data = request.form['frame']
        
        if not frame_data:
            return jsonify({"error": "Empty frame data"}), 400

        # Remove the data URL prefix if present
        if frame_data.startswith('data:image/jpeg;base64,'):
            frame_data = frame_data.split(',')[1]

        # Decode the base64 frame data
        image_data = base64.b64decode(frame_data)
        image = Image.open(BytesIO(image_data))
        
        # Convert the image to an OpenCV format
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Process the frame for hand gestures
        hands, _ = detector.findHands(img, draw=False, flipType=True)

        if hands:
            hand = hands[0]
            fingers = detector.fingersUp(hand)
            lmList = hand["lmList"]
            
            # Index finger for drawing (more precise point)
            if fingers == [0, 1, 0, 0, 0]:
                current_pos = (
                    int(lmList[8][0] * canvas.shape[1] / img.shape[1]),
                    int(lmList[8][1] * canvas.shape[0] / img.shape[0])
                )
                
                if prev_position:
                    cv2.line(canvas, prev_position, current_pos, (0, 0, 0), 2)
                prev_position = current_pos
            
            # Thumb up for eraser
            elif fingers == [1, 0, 0, 0, 0]:
                current_pos = (
                    int(lmList[4][0] * canvas.shape[1] / img.shape[1]),
                    int(lmList[4][1] * canvas.shape[0] / img.shape[0])
                )
                cv2.circle(canvas, current_pos, 20, (255, 255, 255), -1)
                prev_position = None
            
            # Open palm for clearing canvas
            elif sum(fingers) >= 4:
                canvas.fill(255)
                prev_position = None
            
            else:
                prev_position = None
        else:
            prev_position = None

        # Convert the canvas to a base64 image
        _, buffer = cv2.imencode('.png', canvas)
        drawing_data = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({"drawing": f"data:image/png;base64,{drawing_data}"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)