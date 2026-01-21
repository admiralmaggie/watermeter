# Camera Streaming Microservice - Quick Guide

## Overview
A lightweight Flask-based microservice that streams your Raspberry Pi camera to a web browser for easy troubleshooting and setup.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Stream
```bash
python camera_stream.py
```

### 3. Access the Stream
Open a web browser and navigate to:
- **On the Raspberry Pi:** http://localhost:5000
- **From another device:** http://YOUR_PI_IP:5000

Example: `http://192.168.1.100:5000`

## Features

✅ **Minimalistic Interface** - Clean, dark-themed web UI  
✅ **Real-time Streaming** - ~10 FPS MJPEG stream  
✅ **Mobile Responsive** - Works on phones and tablets  
✅ **Health Check Endpoint** - Available at `/health`  
✅ **Auto Camera Management** - Handles initialization and cleanup  

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main streaming page (HTML) |
| `/video_feed` | GET | MJPEG video stream |
| `/health` | GET | Health check (JSON) |

## Usage Tips

### Finding Your Raspberry Pi's IP Address
```bash
hostname -I
```

### Running as a Background Service
```bash
# Start in background
nohup python camera_stream.py > stream.log 2>&1 &

# Stop the service
pkill -f camera_stream.py
```

### Testing Without Camera (Development)
The service will fail to start if `picamera2` is not available or no camera is detected. This is expected on development machines.

### Headless Mode
The streaming service is designed to work in headless mode (no display). Unlike `read_meter.py` which uses `cv2.imshow()`, the streaming service only uses OpenCV for image encoding and serves the video feed through Flask, so it works perfectly on headless Raspberry Pi systems.

## Troubleshooting

### Port Already in Use
If port 5000 is already in use, modify the port in `camera_stream.py`:
```python
app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
```

### Camera Not Detected
```
ERROR: picamera2 not available. Cannot start streaming.
```
**Solution:** Ensure you're running on a Raspberry Pi with the camera enabled:
```bash
sudo raspi-config
# Interface Options → Camera → Enable
```

### Qt/XCB Display Error
If you encounter Qt/XCB display errors, this is typically a `read_meter.py` issue, not the streaming service. However, if you do see such errors:
```bash
# Use opencv-python-headless instead
pip uninstall opencv-python
pip install opencv-python-headless
```

### Slow Streaming
If the stream is slow or laggy:
1. Check your network connection
2. Adjust the frame rate in `camera_stream.py` (increase the `time.sleep()` value)
3. Reduce image quality by lowering the JPEG quality parameter

### Firewall Issues
If you can't access from another device:
```bash
# Allow port 5000 through firewall (if enabled)
sudo ufw allow 5000/tcp
```

## Integration with Water Meter Reader

### Recommended Workflow
1. **Position Camera:** Use the stream to position your camera
2. **Adjust Lighting:** Ensure all dials are clearly visible
3. **Stop Stream:** Press Ctrl+C to stop the streaming service
4. **Run Reader:** Execute `python read_meter.py --camera` to analyze

### Why Not Stream and Read Simultaneously?
The camera can only be accessed by one process at a time. Stop the streaming service before running the meter reader.

## Advanced Configuration

### Change Resolution
Edit `camera_config.py`:
```python
main={"size": (1920, 1080)}  # Change to (1280, 720) for lower res
```

### Adjust Frame Rate
Edit `camera_stream.py`:
```python
time.sleep(0.1)  # Change to 0.2 for 5 FPS, 0.05 for 20 FPS
```

### Change JPEG Quality
Edit `camera_stream.py`:
```python
cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])  # 0-100
```

## Security Note
⚠️ This service is designed for local network use only. The stream is not encrypted and has no authentication. Do not expose it to the public internet without proper security measures.

## Performance Notes
- **CPU Usage:** ~5-15% on Raspberry Pi 4
- **Memory:** ~100-150 MB
- **Bandwidth:** ~1-3 Mbps depending on quality settings
- **Latency:** <500ms on local network

## Support
For issues or questions, refer to the main README.md or check the camera_config.py documentation.
