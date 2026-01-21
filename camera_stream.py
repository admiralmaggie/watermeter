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
from dotenv import load_dotenv
from camera_config import WaterMeterCamera, CAMERA_AVAILABLE
import time
import sys

# Load environment variables from .env file
load_dotenv()

# Import detection functions from read_meter if detection overlay enabled
STREAM_SHOW_DETECTION = os.getenv('STREAM_SHOW_DETECTION', 'false').lower() == 'true'
if STREAM_SHOW_DETECTION:
    try:
        from read_meter import (
            parse_manual_circles, select_best_circles, find_needle, 
            read_value, rotate_image, _resize_for_hough,
            USE_MANUAL_CIRCLES, MANUAL_CIRCLES_STR, DIALS_COUNT,
            HOUGH_DP, HOUGH_PARAM1, HOUGH_PARAM2, RADIUS_MIN_FRAC, RADIUS_MAX_FRAC,
            HOUGH_TARGET_WIDTH
        )
        print("Detection overlay enabled for camera stream")
    except ImportError as e:
        print(f"Warning: Could not import detection functions: {e}")
        STREAM_SHOW_DETECTION = False

app = Flask(__name__)

# Fine rotation adjustment (in degrees, negative = clockwise) - loaded from .env
FINE_ROTATION_ANGLE = int(os.getenv('FINE_ROTATION_ANGLE', '-4'))

# Motion detection settings
MOTION_DETECTION_ENABLED = os.getenv('MOTION_DETECTION_ENABLED', 'false').lower() == 'true'
MOTION_ZONE1_X = int(os.getenv('MOTION_ZONE1_X', '200'))
MOTION_ZONE1_Y = int(os.getenv('MOTION_ZONE1_Y', '400'))
MOTION_ZONE1_R = int(os.getenv('MOTION_ZONE1_R', '100'))
MOTION_ZONE2_X = int(os.getenv('MOTION_ZONE2_X', '600'))
MOTION_ZONE2_Y = int(os.getenv('MOTION_ZONE2_Y', '400'))
MOTION_ZONE2_R = int(os.getenv('MOTION_ZONE2_R', '100'))
MOTION_THRESHOLD = float(os.getenv('MOTION_THRESHOLD', '2.0'))
MOTION_FRAME_SKIP = int(os.getenv('MOTION_FRAME_SKIP', '3'))

# Colors for detection overlay
COLOR_GREEN = (0, 255, 0)
COLOR_MAGENTA = (255, 0, 255)
COLOR_ORANGE = (0, 128, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_CYAN = (255, 255, 0)

# Global camera instance
camera = None

# Motion detection state
motion_detector = None


class MotionDetector:
    """Detects motion in two circular zones of the video stream."""
    
    def __init__(self, zone1, zone2, threshold=2.0, frame_skip=3):
        """
        Initialize motion detector with two zones.
        
        Args:
            zone1: Tuple of (x, y, radius) for first detection zone
            zone2: Tuple of (x, y, radius) for second detection zone
            threshold: Minimum percentage of pixels that must change (0-100)
            frame_skip: Check motion every N frames for efficiency
        """
        self.zone1 = zone1
        self.zone2 = zone2
        self.threshold = threshold
        self.frame_skip = frame_skip
        
        self.prev_frame = None
        self.frame_count = 0
        self.motion_detected_zone1 = False
        self.motion_detected_zone2 = False
        self.last_motion_time_zone1 = None
        self.last_motion_time_zone2 = None
        
        print(f"Motion detector initialized:")
        print(f"  Zone 1: center=({zone1[0]}, {zone1[1]}), radius={zone1[2]}")
        print(f"  Zone 2: center=({zone2[0]}, {zone2[1]}), radius={zone2[2]}")
        print(f"  Threshold: {threshold}%, Frame skip: {frame_skip}")
    
    def create_circular_mask(self, frame_shape, center, radius):
        """Create a circular mask for a specific zone."""
        height, width = frame_shape[:2]
        y, x = np.ogrid[:height, :width]
        cx, cy = center
        
        # Create circular mask
        mask = ((x - cx)**2 + (y - cy)**2 <= radius**2).astype(np.uint8) * 255
        return mask
    
    def detect_motion_in_zone(self, current_gray, prev_gray, mask):
        """
        Detect motion within a masked zone.
        
        Returns:
            Tuple of (motion_detected, change_percentage)
        """
        # Calculate absolute difference
        frame_diff = cv2.absdiff(current_gray, prev_gray)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        
        # Apply mask to only consider the zone
        thresh_masked = cv2.bitwise_and(thresh, thresh, mask=mask)
        
        # Count non-zero pixels in the zone
        changed_pixels = cv2.countNonZero(thresh_masked)
        total_pixels = cv2.countNonZero(mask)
        
        if total_pixels == 0:
            return False, 0.0
        
        # Calculate percentage of changed pixels
        change_percentage = (changed_pixels / total_pixels) * 100
        
        # Motion detected if change exceeds threshold
        motion_detected = change_percentage >= self.threshold
        
        return motion_detected, change_percentage
    
    def process_frame(self, frame):
        """
        Process a frame and detect motion in both zones.
        
        Args:
            frame: BGR image frame
            
        Returns:
            Tuple of (zone1_motion, zone2_motion, zone1_percent, zone2_percent)
        """
        self.frame_count += 1
        
        # Skip frames for efficiency
        if self.frame_count % self.frame_skip != 0:
            return (self.motion_detected_zone1, self.motion_detected_zone2, 0.0, 0.0)
        
        # Convert to grayscale and blur to reduce noise
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.GaussianBlur(current_gray, (21, 21), 0)
        
        # Initialize on first frame
        if self.prev_frame is None:
            self.prev_frame = current_gray
            return (False, False, 0.0, 0.0)
        
        # Create masks for both zones
        mask1 = self.create_circular_mask(frame.shape, 
                                          (self.zone1[0], self.zone1[1]), 
                                          self.zone1[2])
        mask2 = self.create_circular_mask(frame.shape, 
                                          (self.zone2[0], self.zone2[1]), 
                                          self.zone2[2])
        
        # Detect motion in both zones
        motion1, percent1 = self.detect_motion_in_zone(current_gray, self.prev_frame, mask1)
        motion2, percent2 = self.detect_motion_in_zone(current_gray, self.prev_frame, mask2)
        
        # Update state
        self.motion_detected_zone1 = motion1
        self.motion_detected_zone2 = motion2
        
        if motion1:
            self.last_motion_time_zone1 = time.time()
        if motion2:
            self.last_motion_time_zone2 = time.time()
        
        # Update previous frame
        self.prev_frame = current_gray
        
        return (motion1, motion2, percent1, percent2)
    
    def draw_zones(self, frame):
        """Draw motion detection zones and status on frame."""
        # Draw zone 1
        color1 = COLOR_RED if self.motion_detected_zone1 else COLOR_GREEN
        cv2.circle(frame, (self.zone1[0], self.zone1[1]), self.zone1[2], color1, 2)
        cv2.putText(frame, "Zone 1", 
                   (self.zone1[0] - 30, self.zone1[1] - self.zone1[2] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color1, 2)
        
        # Draw zone 2
        color2 = COLOR_RED if self.motion_detected_zone2 else COLOR_GREEN
        cv2.circle(frame, (self.zone2[0], self.zone2[1]), self.zone2[2], color2, 2)
        cv2.putText(frame, "Zone 2", 
                   (self.zone2[0] - 30, self.zone2[1] - self.zone2[2] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color2, 2)
        
        # Draw motion status indicator
        status_y = 30
        if self.motion_detected_zone1 or self.motion_detected_zone2:
            status_text = "MOTION DETECTED: "
            zones = []
            if self.motion_detected_zone1:
                zones.append("Zone 1")
            if self.motion_detected_zone2:
                zones.append("Zone 2")
            status_text += ", ".join(zones)
            cv2.putText(frame, status_text, (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_RED, 2)
        else:
            cv2.putText(frame, "No Motion", (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_GREEN, 2)
        
        return frame


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

def apply_detection_overlay(frame):
    """Apply circle and needle detection overlay to frame."""
    if not STREAM_SHOW_DETECTION:
        return frame
    
    try:
        output = frame.copy()
        
        # Check if using manual circles
        if USE_MANUAL_CIRCLES:
            circles = parse_manual_circles(MANUAL_CIRCLES_STR)
            if len(circles) > 0:
                circles = sorted(circles, key=lambda c: c[0])
        else:
            # Dynamic circle detection
            hough_frame, scale = _resize_for_hough(frame)
            gray = cv2.cvtColor(hough_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 1.5)
            
            mind = min(gray.shape[0], gray.shape[1])
            min_radius = max(10, int(mind * RADIUS_MIN_FRAC))
            max_radius = max(min_radius + 1, int(mind * RADIUS_MAX_FRAC))
            
            circles_raw = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, dp=HOUGH_DP,
                minDist=max(10, min_radius),
                param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
                minRadius=min_radius, maxRadius=max_radius
            )
            
            if circles_raw is None:
                cv2.putText(output, "No circles detected", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return output
            
            selected = select_best_circles(circles_raw, gray, DIALS_COUNT)
            circles = []
            for (x, y, r) in selected:
                if scale != 1.0:
                    x = int(round(x / scale))
                    y = int(round(y / scale))
                    r = int(round(r / scale))
                circles.append((x, y, r))
            circles = sorted(circles, key=lambda c: c[0])
        
        if len(circles) != DIALS_COUNT:
            cv2.putText(output, f"Found {len(circles)}/{DIALS_COUNT} dials",
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            # Still draw what we found
            for (x, y, r) in circles:
                cv2.circle(output, (x, y), r, COLOR_ORANGE, 3)
            return output
        
        # Draw circles and detect needles
        values = []
        readout_conventions = ["CW", "CW", "CW", "CW", "CW"]
        
        for i, ((x, y, r), convention) in enumerate(zip(circles, readout_conventions)):
            value, tip = find_needle(output, x, y, r)
            actual_value = read_value(value, convention)
            values.append(actual_value)
            
            # Draw circle
            cv2.circle(output, (x, y), r, COLOR_GREEN, 3)
            # Draw needle line
            cv2.line(output, (x, y), tip, COLOR_MAGENTA, thickness=2)
            # Draw value
            cv2.putText(output, f"{actual_value:.2f}", (x - 30, y + r + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_BLUE, 2)
        
        # Show final reading
        if len(values) == DIALS_COUNT:
            from read_meter import process_values
            reading = process_values(values)
            cv2.putText(output, f"Reading: {reading}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_BLUE, 3)
        
        return output
        
    except Exception as e:
        print(f"Detection overlay error: {e}")
        # Return frame with error message
        cv2.putText(frame, f"Detection error", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

def generate_frames():
    """Generator function that yields frames in MJPEG format."""
    global motion_detector
    
    cam = get_camera()
    
    # Initialize motion detector if enabled
    if MOTION_DETECTION_ENABLED and motion_detector is None:
        zone1 = (MOTION_ZONE1_X, MOTION_ZONE1_Y, MOTION_ZONE1_R)
        zone2 = (MOTION_ZONE2_X, MOTION_ZONE2_Y, MOTION_ZONE2_R)
        motion_detector = MotionDetector(zone1, zone2, MOTION_THRESHOLD, MOTION_FRAME_SKIP)
        print("Motion detection enabled for camera stream")
    
    while True:
        try:
            # Capture frame from camera
            frame = cam.capture_image()
            
            # Convert from RGB to BGR for OpenCV encoding
            if frame.shape[2] == 3 and len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Rotate image 90 degrees clockwise
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
            # Apply fine rotation for alignment (if enabled)
            if FINE_ROTATION_ANGLE != 0:
                height, width = frame.shape[:2]
                center = (width / 2, height / 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, FINE_ROTATION_ANGLE, 1.0)
                frame = cv2.warpAffine(frame, rotation_matrix, (width, height), 
                                        flags=cv2.INTER_LINEAR, 
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(255, 255, 255))
            
            # Process motion detection if enabled
            if MOTION_DETECTION_ENABLED and motion_detector is not None:
                motion1, motion2, percent1, percent2 = motion_detector.process_frame(frame)
                # Draw motion zones on frame
                frame = motion_detector.draw_zones(frame)
            
            # Apply detection overlay if enabled
            if STREAM_SHOW_DETECTION:
                frame = apply_detection_overlay(frame)
            
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
        print(f"  Detection overlay: {'ENABLED' if STREAM_SHOW_DETECTION else 'DISABLED'}")
        print(f"  Motion detection: {'ENABLED' if MOTION_DETECTION_ENABLED else 'DISABLED'}")
        if MOTION_DETECTION_ENABLED:
            print(f"    Zone 1: ({MOTION_ZONE1_X}, {MOTION_ZONE1_Y}, r={MOTION_ZONE1_R})")
            print(f"    Zone 2: ({MOTION_ZONE2_X}, {MOTION_ZONE2_Y}, r={MOTION_ZONE2_R})")
            print(f"    Threshold: {MOTION_THRESHOLD}%")
        print("\nAccess the stream at: http://0.0.0.0:5000")
        print("From another device: http://<raspberry-pi-ip>:5000")
        print("\nPress Ctrl+C to stop\n")
        
        # Run Flask app
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        cleanup()
        print("Camera stream stopped.")
