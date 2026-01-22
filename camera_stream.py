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
import time
import sys
from dotenv import load_dotenv
from camera_config import WaterMeterCamera, CAMERA_AVAILABLE
from read_meter import (
    MotionDetector, get_frame_from_camera, read_meter_from_frame,
    MOTION_ZONE1_X, MOTION_ZONE1_Y, MOTION_ZONE1_R,
    MOTION_ZONE2_X, MOTION_ZONE2_Y, MOTION_ZONE2_R,
    MOTION_THRESHOLD, MOTION_FRAME_SKIP,
    MOTION_DETECTION_ENABLED
)

# Load environment variables from .env file
load_dotenv()

# Import detection settings
STREAM_SHOW_DETECTION = os.getenv('STREAM_SHOW_DETECTION', 'false').lower() == 'true'

app = Flask(__name__)

# Global state
camera_instance = None
motion_detector = None

class StreamArgs:
    """Mock arguments for get_frame_from_camera."""
    def __init__(self):
        self.bracket = False  # Don't use bracketing for live stream (too slow)
        self.exposures = ""

def get_camera():
    """Get or initialize the camera instance."""
    global camera_instance
    if camera_instance is None:
        if not CAMERA_AVAILABLE:
            print("ERROR: picamera2 not available. Cannot start streaming.")
            sys.exit(1)
        camera_instance = WaterMeterCamera()
        camera_instance.initialize()
    return camera_instance

def generate_frames():
    """Generator function that yields frames in MJPEG format."""
    global motion_detector
    
    cam = get_camera()
    args = StreamArgs()
    
    # Initialize motion detector if enabled
    if MOTION_DETECTION_ENABLED and motion_detector is None:
        zone1 = (MOTION_ZONE1_X, MOTION_ZONE1_Y, MOTION_ZONE1_R)
        zone2 = (MOTION_ZONE2_X, MOTION_ZONE2_Y, MOTION_ZONE2_R)
        motion_detector = MotionDetector(zone1, zone2, MOTION_THRESHOLD, MOTION_FRAME_SKIP)
        print("Motion detection enabled for camera stream")
    
    while True:
        try:
            # 1. Capture and preprocess frame using shared logic
            frame = get_frame_from_camera(cam, args)
            
            # 2. Apply motion detection and meter reading overlay if enabled
            if STREAM_SHOW_DETECTION:
                # read_meter_from_frame handles motion detection and annotation
                _, _, frame = read_meter_from_frame(frame, motion_detector, show_gui=True)
            elif MOTION_DETECTION_ENABLED and motion_detector is not None:
                # Just motion detection
                motion_detector.process_frame(frame)
                frame = motion_detector.draw_zones(frame)
            
            # 3. Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret: continue
            
            # 4. Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # Small delay to maintain ~15 FPS
            time.sleep(0.06)
            
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
    global camera_instance
    if camera_instance is not None:
        camera_instance.close()
        camera_instance = None

if __name__ == '__main__':
    try:
        print("=" * 60)
        print("Water Meter Camera Stream - Troubleshooting Tool")
        print("=" * 60)
        print(f"  Detection overlay: {'ENABLED' if STREAM_SHOW_DETECTION else 'DISABLED'}")
        print(f"  Motion detection: {'ENABLED' if MOTION_DETECTION_ENABLED else 'DISABLED'}")
        print("\nAccess the stream at: http://0.0.0.0:5000")
        print("\nPress Ctrl+C to stop\n")
        
        # Run Flask app
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cleanup()
