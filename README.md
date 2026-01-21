# Water Meter Reader with Raspberry Pi Camera Support

This project reads analog water meter dials using computer vision. It now supports both static image files and live capture from a Raspberry Pi camera.

## Features

- **Dial Detection**: Automatically detects circular meter dials using Hough Circle Transform
- **Needle Reading**: Uses HSV color space to isolate red needles and calculate their angle
- **Multiple Modes**:
  - Static image file analysis
  - Single capture from Raspberry Pi camera
  - Continuous monitoring with automatic readings
- **Debug Visualization**: Visual overlays showing detected circles, needles, and readings

## Installation

### On Raspberry Pi

```bash
# Install required Python packages using requirements.txt
pip install -r requirements.txt

# For Raspberry Pi OS Bullseye or newer, picamera2 should be pre-installed
# If not, install it:
sudo apt update
sudo apt install -y python3-picamera2
```

### On Desktop/Development Machine (Windows/Mac/Linux)

```bash
# Install basic requirements (camera features won't be available)
pip install opencv-python numpy matplotlib Flask

# Or use requirements.txt (picamera2 will be skipped automatically on non-ARM platforms)
pip install -r requirements.txt

# Note: picamera2 only works on Raspberry Pi hardware
```

## Usage

### 1. Camera Stream for Troubleshooting (NEW!)

The camera streaming microservice provides a live view of the camera feed in your web browser. This is perfect for:
- Positioning the camera to capture all meter dials
- Adjusting lighting and focus
- Verifying the camera setup before running the meter reader

```bash
# Start the streaming service on Raspberry Pi
python camera_stream.py

# Then open a web browser and navigate to:
# - On the Pi: http://localhost:5000
# - From another device: http://<raspberry-pi-ip>:5000
```

The stream provides a clean, minimalistic interface that displays the live camera feed at ~10 FPS. Once you've positioned everything correctly, press Ctrl+C to stop the stream and proceed with meter reading.

### 2. Analyze a Static Image File

```bash
# Use default image (test.jpeg)
python test.py --file test.jpeg

# Or specify a different image
python test.py --file path/to/your/image.jpg
```

### 3. Single Capture from Raspberry Pi Camera

```bash
# Capture one image and analyze it
python read_meter.py --camera
```

### 4. Continuous Monitoring Mode

```bash
# Take readings every 5 seconds (default)
python read_meter.py --camera --continuous

# Custom interval (e.g., every 30 seconds)
python read_meter.py --camera --continuous --interval 30
```

### 5. Save Images

```bash
# Save captured/processed images to 'data' folder
python read_meter.py --camera --save

# Continuous mode with saving
python read_meter.py --camera --continuous --save --interval 60
```

## Command-Line Options

### read_meter.py

| Option | Description |
|--------|-------------|
| `--camera` | Use Raspberry Pi camera instead of image file |
| `--file <path>` | Path to image file (default: test.jpeg) |
| `--continuous` | Continuous capture mode (camera only) |
| `--save` | Save captured/processed images |
| `--interval <seconds>` | Seconds between captures in continuous mode (default: 5) |

### camera_stream.py

No command-line options. Simply run `python camera_stream.py` and access the stream via web browser at `http://<raspberry-pi-ip>:5000`

## Configuration

Edit these constants in `read_meter.py` to tune the detection:

```python
DIALS_COUNT = 3              # Number of dials to detect
DEBUG_NEEDLE = True          # Show debug visualization windows
HOUGH_PARAM2 = 45           # Circle detection sensitivity
RADIUS_MIN_FRAC = 0.04      # Minimum dial radius (fraction of image)
RADIUS_MAX_FRAC = 0.12      # Maximum dial radius (fraction of image)
```

### Camera Configuration

Edit `camera_config.py` to adjust camera settings:

```python
# Resolution settings in WaterMeterCamera.initialize()
main={"size": (1920, 1080)}  # High-res still capture
lores={"size": (640, 480)}   # Preview resolution
```

## How It Works

1. **Circle Detection**: Uses Hough Circle Transform to locate circular dials
2. **Needle Isolation**: Converts image to HSV color space and isolates red pixels
3. **Angle Calculation**: Fits a line through needle pixels and calculates angle
4. **Value Conversion**: Converts angle (0-360°) to dial reading (0-10)
5. **Final Reading**: Combines all dial values into a complete meter reading

## Troubleshooting

### Camera Not Found

```
ERROR: Camera mode requested but picamera2 is not available.
```

**Solution**: Install picamera2 on your Raspberry Pi:
```bash
sudo apt install -y python3-picamera2
```

### No Circles Detected

```
DEBUG: No circles found by HoughCircles.
```

**Solutions**:
- Ensure good lighting on the meter
- Adjust `HOUGH_PARAM2` (lower value = more sensitive)
- Check that dials are clearly visible in the image
- Verify dial sizes match `RADIUS_MIN_FRAC` and `RADIUS_MAX_FRAC`

### Wrong Number of Circles

```
DEBUG: Found 5 selected circles, but expected 3.
```

**Solutions**:
- Increase `HOUGH_PARAM2` to reduce false detections
- Adjust `RADIUS_MIN_FRAC` and `RADIUS_MAX_FRAC` to match your dial sizes
- Ensure meter is centered in the frame

### Needle Not Detected

```
DEBUG: No dark pixels found in needle ring.
```

**Solutions**:
- Adjust HSV thresholds in `find_needle()` function
- Ensure needles are red colored
- Improve lighting conditions
- Check that needle is within the dial area

## Camera Setup Tips

1. **Mounting**: Mount the camera at a stable position directly in front of the meter
2. **Distance**: Adjust distance so all dials are clearly visible and well-lit
3. **Lighting**: Ensure even lighting without glare or shadows on the dials
4. **Angle**: Position camera perpendicular to meter face (avoid angles)
5. **Focus**: Ensure camera is focused (Pi Camera v2/v3 has fixed focus)

## Project Structure

```
waterMeter/
├── read_meter.py        # Main script with dial detection logic
├── camera_config.py     # Raspberry Pi camera wrapper
├── camera_stream.py     # Flask microservice for live camera streaming
├── templates/
│   └── index.html       # Web interface for camera stream
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── test.jpeg            # Sample test image
├── test1.jpeg           # Sample test image
└── data/                # Output folder for saved images (created automatically)
```

## Examples

### Setup and Troubleshooting Workflow
```bash
# Step 1: Start the camera stream to position your camera
python camera_stream.py
# Open http://<raspberry-pi-ip>:5000 in a browser
# Adjust camera position and lighting until all dials are clearly visible
# Press Ctrl+C when satisfied

# Step 2: Test with a single reading
python read_meter.py --camera

# Step 3: If successful, start continuous monitoring
python read_meter.py --camera --continuous --interval 60 --save
```

### Basic Desktop Testing
```bash
# Test with your own image
python read_meter.py --file my_meter.jpg
```

### Raspberry Pi Monitoring
```bash
# Run continuous monitoring every minute
python read_meter.py --camera --continuous --interval 60 --save
```

### Scheduled Readings (with cron)
```bash
# Add to crontab for hourly readings
0 * * * * cd /home/pi/waterMeter && python read_meter.py --camera --save >> readings.log 2>&1
```

## License

This project is for educational and personal use.
