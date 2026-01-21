# Hardware Crop Configuration Guide

## Overview
Hardware cropping uses picamera2's `ScalerCrop` control to crop images at the sensor/ISP level **before** data transfer and rotation. This is the most efficient cropping method.

## Quick Start

### 1. Enable Cropping
Edit `camera_config.py`:

```python
# Enable hardware crop
CROP_ENABLED = True

# Define crop region (x, y, width, height) in sensor coordinates
CROP_REGION = (1152, 648, 2304, 1296)  # Center 50% crop
```

### 2. Run Your Scripts
The crop will automatically apply to:
- `camera_stream.py` (web streaming)
- `read_meter.py --camera` (meter reading)

## Understanding Coordinates

### Sensor Space (Before Rotation)
- Full sensor: **4608 x 2592** (landscape)
- Coordinates are in **sensor space** (before 90° clockwise rotation)
- Format: `(x, y, width, height)` where (x,y) is top-left corner

### After 90° Clockwise Rotation
- A crop of (x=1152, y=648, w=2304, h=1296) becomes **1296 x 2304** (portrait)
- Width becomes height, height becomes width

## Preset Crop Regions

### Full Sensor (No Crop)
```python
CROP_ENABLED = False
CROP_REGION = (0, 0, 4608, 2592)
```
**Result after rotation:** 2592 x 4608

### Center 75%
```python
CROP_ENABLED = True
CROP_REGION = (576, 324, 3456, 1944)
```
**Result after rotation:** 1944 x 3456

### Center 50%
```python
CROP_ENABLED = True
CROP_REGION = (1152, 648, 2304, 1296)
```
**Result after rotation:** 1296 x 2304

### Top Half (Before Rotation)
```python
CROP_ENABLED = True
CROP_REGION = (0, 0, 4608, 1296)
```
**Result after rotation:** 1296 x 4608

### Bottom Half (Before Rotation)
```python
CROP_ENABLED = True
CROP_REGION = (0, 1296, 4608, 1296)
```
**Result after rotation:** 1296 x 4608

### Left Third (Before Rotation)
```python
CROP_ENABLED = True
CROP_REGION = (0, 0, 1536, 2592)
```
**Result after rotation:** 2592 x 1536

### Custom Crop
```python
CROP_ENABLED = True
CROP_REGION = (x, y, width, height)  # Your coordinates
```

## Finding the Right Crop Region

### Method 1: Use the Web Stream
1. Set `CROP_ENABLED = False` temporarily
2. Run `python camera_stream.py`
3. View the stream in your browser
4. Note where the meter dials appear in the frame
5. Calculate approximate crop coordinates
6. Update `CROP_REGION` and enable cropping
7. Restart the stream to verify

### Method 2: Trial and Error
1. Start with a conservative crop (e.g., center 75%)
2. Run `python camera_stream.py` to preview
3. Adjust coordinates as needed
4. Restart to apply changes

### Method 3: Calculate from Center
If your meter is roughly centered:
```python
# Formula for center crop
center_x = 4608 // 2  # 2304
center_y = 2592 // 2  # 1296

# For 50% crop (2304x1296 region)
x = center_x - (2304 // 2)  # 1152
y = center_y - (1296 // 2)  # 648
CROP_REGION = (x, y, 2304, 1296)
```

## Dynamic Crop Control

You can also control cropping programmatically:

```python
from camera_config import WaterMeterCamera

# Initialize with custom crop
camera = WaterMeterCamera(crop_enabled=True, crop_region=(1152, 648, 2304, 1296))
camera.initialize()

# Or update crop region
camera.set_crop_region(1152, 648, 2304, 1296)
# Note: Requires camera restart to apply

# Check current crop settings
info = camera.get_crop_info()
print(f"Crop enabled: {info['enabled']}")
print(f"Crop region: {info['region']}")
```

## Benefits of Hardware Cropping

✅ **Performance:** Crops at ISP level, reduces data transfer  
✅ **Efficiency:** Lower memory usage, less CPU processing  
✅ **Speed:** Faster capture and processing  
✅ **Quality:** No quality loss (not interpolated)  

## Troubleshooting

### Crop Not Applied
- Ensure `CROP_ENABLED = True`
- Verify coordinates are within sensor bounds (0-4608, 0-2592)
- Check console output for error messages
- Try restarting the script

### Image Looks Wrong
- Remember coordinates are **before** 90° rotation
- Verify width and height are positive
- Check that x+width ≤ 4608 and y+height ≤ 2592

### Want to Disable Temporarily
```python
CROP_ENABLED = False  # Quick toggle
```

### Test Different Crops Quickly
```bash
# In Python
from camera_config import WaterMeterCamera, CROP_ENABLED, CROP_REGION
camera = WaterMeterCamera(crop_enabled=True, crop_region=(x, y, w, h))
```

## Example Workflow

### Step 1: Identify Meter Location
```bash
# Disable crop to see full frame
# Edit camera_config.py: CROP_ENABLED = False
python camera_stream.py
# View stream and note meter position
```

### Step 2: Calculate Crop
```python
# Based on observations, meter is roughly:
# - 1000px from left
# - 500px from top  
# - 3000px wide
# - 1500px tall
CROP_REGION = (1000, 500, 3000, 1500)
```

### Step 3: Apply and Test
```python
CROP_ENABLED = True
CROP_REGION = (1000, 500, 3000, 1500)
```
```bash
python camera_stream.py  # Verify crop looks good
python read_meter.py --camera  # Test meter reading
```

### Step 4: Fine-tune
Adjust coordinates as needed until optimal.

## Tips

1. **Start Wide:** Begin with a larger crop region, then narrow
2. **Include Margin:** Leave some space around meter dials
3. **Check Rotation:** Remember the 90° clockwise rotation impact
4. **Use Stream:** The streaming service is perfect for preview
5. **Document:** Note your final crop coordinates in comments

## Advanced: Crop Visualization (Future Enhancement)

Could add to streaming service:
- Grid overlay showing crop boundaries
- Interactive crop selector
- Preset crop buttons
- Save/load crop profiles

---

**Note:** Changes to `CROP_ENABLED` or `CROP_REGION` require restarting the camera/script to take effect.
