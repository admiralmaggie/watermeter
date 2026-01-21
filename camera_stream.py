#!/usr/bin/env python3
"""
Camera Streaming Microservice for Water Meter Troubleshooting
Streams the Raspberry Pi camera to a web browser for setup and debugging.
"""

import os
# Set environment variable to disable OpenCV GUI for headless operation
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

from flask import Flask, render_template, Response
import cv2
import numpy as np
from camera_config import WaterMeterCamera, CAMERA_AVAILABLE
import time
import sys

app = Flask(__name__)

# Global camera instance
camera = None

def get_camera():
    """Get or initialize the camera instance."""
    global camera
    if camera is None:
        if not CAMERA_AVAILABLE:
            print("ERROR: picamera2 not available. Cannot start streaming.")
            sys.exit(1)
        camera = WaterMeterCamera()
        camera.initialize()
    return camera

def generate_frames():
    """Generator function that yields frames in MJPEG format."""
    cam = get_camera()
    
    while True:
        try:
            # Capture frame from camera
            frame = cam.capture_image()
            
            # Convert from RGB to BGR for OpenCV encoding
            if frame.shape[2] == 3 and len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Rotate image 90 degrees clockwise
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
            # Apply 3-degree fine rotation for alignment
            height, width = frame.shape[:2]
            center = (width / 2, height / 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, 3, 1.0)
            frame = cv2.warpAffine(frame, rotation_matrix, (width, height), 
                                    flags=cv2.INTER_LINEAR, 
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(255, 255, 255))
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            if not ret:
                print("ERROR: Failed to encode frame")
                continue
            
            # Convert to bytes
            frame_bytes = buffer.tobytes()
            
            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Small delay to control frame rate (~10 FPS)
            time.sleep(0.1)
            
        except Exception as e:
            print(f"ERROR in frame generation: {e}")
            time.sleep(1)

@app.route('/')
def index():
    """Render the main page with video stream."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/health')
def health():
    """Health check endpoint."""
    return {'status': 'ok', 'camera_available': CAMERA_AVAILABLE}, 200

def cleanup():
    """Clean up camera resources."""
    global camera
    if camera is not None:
        camera.close()
        camera = None

if __name__ == '__main__':
    try:
        print("=" * 60)
        print("Water Meter Camera Stream - Troubleshooting Tool")
        print("=" * 60)
        print("\nStarting camera stream microservice...")
        print("Access the stream at: http://0.0.0.0:5000")
        print("From another device: http://<raspberry-pi-ip>:5000")
        print("\nPress Ctrl+C to stop\n")
        
        # Run Flask app
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        cleanup()
        print("Camera stream stopped.")
