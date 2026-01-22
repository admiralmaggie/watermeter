import os
import time
import requests
import json
from dotenv import load_dotenv
import cv2
import read_meter
from read_meter import (
    WaterMeterCamera, 
    MotionDetector, 
    get_frame_from_camera, 
    read_meter_from_frame,
    MOTION_ZONE1_X, MOTION_ZONE1_Y, MOTION_ZONE1_R,
    MOTION_ZONE2_X, MOTION_ZONE2_Y, MOTION_ZONE2_R,
    MOTION_THRESHOLD, MOTION_FRAME_SKIP,
    CAMERA_AVAILABLE
)

# Load environment variables
load_dotenv()

# Monitoring Settings
UPLOAD_INTERVAL_MIN = int(os.getenv('UPLOAD_INTERVAL_MINUTES', '15'))
UPLOAD_INTERVAL_SEC = UPLOAD_INTERVAL_MIN * 60

THINGSBOARD_TOKEN = os.getenv('THINGSBOARD_TOKEN', 'MISSING_TOKEN')
THINGSBOARD_URL = f"https://thingsboard.cloud/api/v1/{THINGSBOARD_TOKEN}/telemetry"

LEAK_THRESHOLD_MIN = int(os.getenv('LEAK_DETECTION_THRESHOLD_MINUTES', '60'))
LEAK_THRESHOLD_SEC = LEAK_THRESHOLD_MIN * 60

# Pushover alert settings
PUSHOVER_TOKEN = os.getenv('PUSHOVER_TOKEN', 'MISSING_TOKEN')
PUSHOVER_USER = os.getenv('PUSHOVER_USER', '')
PUSHOVER_URL = "https://api.pushover.net/1/messages.json"

class MonitorArgs:
    """Mock arguments for get_frame_from_camera."""
    def __init__(self):
        self.bracket = os.getenv('BRACKET', 'false').lower() == 'true'
        self.exposures = os.getenv('EXPOSURES', '-1.0,0.0,1.0')

def send_telemetry(usage):
    """Upload water usage to ThingsBoard."""
    try:
        # Use standard JSON format
        data = {"water_usage": usage}
        headers = {"Content-Type": "application/json"}
        response = requests.post(THINGSBOARD_URL, json=data, headers=headers, timeout=10)
        response.raise_for_status()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Telemetry sent: {usage} gallons")
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error sending telemetry: {e}")

def send_leak_alert():
    """Send alert via Pushover."""
    try:
        payload = {
            "token": PUSHOVER_TOKEN,
            "user": PUSHOVER_USER,
            "message": "*** Water leak detected!"
        }
        response = requests.post(PUSHOVER_URL, data=payload, timeout=10)
        response.raise_for_status()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Leak alert sent to Pushover")
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error sending leak alert: {e}")

def main():
    if not CAMERA_AVAILABLE:
        print("ERROR: Camera hardware or picamera2 library not available.")
        return

    print("Water Meter Monitor started.")
    print(f"Upload interval: {UPLOAD_INTERVAL_MIN} minutes")
    print(f"Leak detection threshold: {LEAK_THRESHOLD_MIN} minutes")
    print(f"ThingsBoard Token: {THINGSBOARD_TOKEN}")
    print("Press Ctrl+C to stop.")

    # Initialize components
    zone1 = (MOTION_ZONE1_X, MOTION_ZONE1_Y, MOTION_ZONE1_R)
    zone2 = (MOTION_ZONE2_X, MOTION_ZONE2_Y, MOTION_ZONE2_R)
    motion_det = MotionDetector(zone1, zone2, MOTION_THRESHOLD, MOTION_FRAME_SKIP)
    
    args = MonitorArgs()
    last_upload_time = time.time()
    motion_start_time = None
    leak_alert_sent = False
    
    last_absolute_reading = None
    accumulated_usage = 0.0

    # Disable image saving by default in monitor mode unless forced
    read_meter.SAVE_IMAGE = os.getenv('SAVE_MONITOR_IMAGES', 'false').lower() == 'true'

    with WaterMeterCamera() as camera:
        while True:
            try:
                # 1. Capture and preprocess frame
                frame = get_frame_from_camera(camera, args)
                if frame is None:
                    time.sleep(5)
                    continue

                # 2. Process frame for reading and motion
                reading, motion_detected = read_meter_from_frame(frame, motion_det, show_gui=False)
                
                now = time.time()

                # 3. Handle Leak Detection Logic
                if motion_detected:
                    if motion_start_time is None:
                        motion_start_time = now
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Constant motion detected - leak monitoring active")
                    elif now - motion_start_time >= LEAK_THRESHOLD_SEC:
                        if not leak_alert_sent:
                            send_leak_alert()
                            leak_alert_sent = True
                else:
                    if motion_start_time is not None:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Motion stopped - leak timer reset")
                    motion_start_time = None
                    leak_alert_sent = False

                # 4. Handle Usage Calculation and Telemetry Upload
                if reading is not None:
                    try:
                        # Convert reading (e.g. "02.0") to float
                        current_reading = float(reading)
                        
                        if last_absolute_reading is not None:
                            # Calculate delta, handling rollover at 100.0
                            # % 100 works for positive and negative deltas (rollover)
                            delta = (current_reading - last_absolute_reading) % 100
                            
                            # Sanity check: if delta is very large, it might be a reading error
                            # (e.g. 99.9 gallons used in 5 seconds is unlikely)
                            if delta < 50.0: 
                                accumulated_usage += delta
                            else:
                                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Warning: Ignored large delta {delta:.1f}")
                        
                        last_absolute_reading = current_reading

                        # Time to upload?
                        if now - last_upload_time >= UPLOAD_INTERVAL_SEC:
                            # Round to 1 decimal point
                            send_telemetry(round(accumulated_usage, 1))
                            accumulated_usage = 0.0
                            last_upload_time = now
                    except ValueError:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Warning: Could not parse reading '{reading}'")

                # Check for motion more frequently than we upload
                # 5 seconds provides good balance between responsiveness and CPU load
                time.sleep(5)

            except KeyboardInterrupt:
                print("\nStopping Water Meter Monitor...")
                break
            except Exception as e:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Unexpected error in loop: {e}")
                time.sleep(10)

if __name__ == "__main__":
    main()
