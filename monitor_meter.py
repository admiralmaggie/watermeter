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
                reading, motion_detected, _ = read_meter_from_frame(frame, motion_det, show_gui=False)
                
                now = time.time()
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

                # 3. Handle Leak Detection Logic
                if motion_detected:
                    if motion_start_time is None:
                        motion_start_time = now
                        print(f"[{timestamp}] MOTION: Constant motion started")
                    elif now - motion_start_time >= LEAK_THRESHOLD_SEC:
                        if not leak_alert_sent:
                            send_leak_alert()
                            leak_alert_sent = True
                else:
                    if motion_start_time is not None:
                        duration = now - motion_start_time
                        print(f"[{timestamp}] MOTION: Motion stopped (lasted {duration:.0f}s)")
                    motion_start_time = None
                    leak_alert_sent = False

                # 4. Handle Usage Calculation and Telemetry Upload
                if reading is not None:
                    try:
                        current_reading = float(reading)
                        
                        if last_absolute_reading is not None:
                            # Calculate delta handling the 100-gallon rollover
                            delta = (current_reading - last_absolute_reading) % 100
                            
                            # Filter Jitter and Reading Errors:
                            # A water meter should only move forward.
                            if delta < 5.0: 
                                # Case 1: Valid increase or rollover
                                if delta > 0:
                                    # Explicitly log rollovers for verification
                                    if current_reading < last_absolute_reading:
                                        print(f"[{timestamp}] ROLLOVER DETECTED: {last_absolute_reading} -> {current_reading}")
                                    
                                    print(f"[{timestamp}] READING: {reading} (Delta: +{delta:.1f} gal, Total: {accumulated_usage+delta:.1f} gal)")
                                    accumulated_usage += delta
                                else:
                                    print(f"[{timestamp}] READING: {reading} (No change)")
                                
                                # Update our reference reading
                                last_absolute_reading = current_reading
                            elif delta > 95.0:
                                # Case 2: Jitter (small negative change like 56.9 -> 56.5)
                                # We ignore this and don't update last_absolute_reading
                                print(f"[{timestamp}] JITTER: Ignored {reading} (Previous: {last_absolute_reading})")
                            else:
                                # Case 3: Large jump or suspicious reading
                                print(f"[{timestamp}] WARNING: Ignored suspicious reading {reading} (Delta: {delta:.1f})")
                                # We also don't update reference here to wait for consistency
                        else:
                            print(f"[{timestamp}] INITIAL READING: {reading}")
                            last_absolute_reading = current_reading

                        # Time to upload?
                        time_since_upload = now - last_upload_time
                        if time_since_upload >= UPLOAD_INTERVAL_SEC:
                            send_telemetry(round(accumulated_usage, 1))
                            accumulated_usage = 0.0
                            last_upload_time = now
                        else:
                            remaining = UPLOAD_INTERVAL_SEC - time_since_upload
                            if int(now) % 60 < 5: # Print status roughly every minute
                                print(f"[{timestamp}] STATUS: {accumulated_usage:.1f} gal accumulated. Next upload in {remaining/60:.1f} min")

                    except ValueError:
                        print(f"[{timestamp}] WARNING: Could not parse reading '{reading}'")
                else:
                    print(f"[{timestamp}] WARNING: Dials not detected in this frame")

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
