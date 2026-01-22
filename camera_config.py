# Camera configuration for Raspberry Pi
# This module handles camera initialization and image capture

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    from picamera2 import Picamera2
    import time
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    print("WARNING: picamera2 not available. Camera features disabled.")

# Hardware Crop Configuration (loaded from .env)
# Coordinates are in sensor space (4608x2592 landscape before 90° clockwise rotation)
# Format: (x, y, width, height) where (x,y) is top-left corner
CROP_ENABLED = os.getenv('CROP_ENABLED', 'false').lower() == 'true'
CROP_REGION = (
    int(os.getenv('CROP_X', '0')),
    int(os.getenv('CROP_Y', '0')),
    int(os.getenv('CROP_WIDTH', '4608')),
    int(os.getenv('CROP_HEIGHT', '2592'))
)

# Camera resolution settings (loaded from .env)
CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH', '4608'))
CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT', '2592'))
PREVIEW_WIDTH = int(os.getenv('PREVIEW_WIDTH', '640'))
PREVIEW_HEIGHT = int(os.getenv('PREVIEW_HEIGHT', '480'))

# Autofocus settings (loaded from .env)
AUTOFOCUS_ENABLED = os.getenv('AUTOFOCUS_ENABLED', 'true').lower() == 'true'
AUTOFOCUS_MODE = os.getenv('AUTOFOCUS_MODE', 'continuous').lower()  # 'continuous', 'trigger', or 'manual'
LENS_POSITION = float(os.getenv('LENS_POSITION', '1.0'))  # Lens position for manual mode (0.0-10.0)

class WaterMeterCamera:
    """Wrapper for Raspberry Pi camera operations."""
    
    def __init__(self, crop_enabled=None, crop_region=None):
        self.camera = None
        self.is_initialized = False
        # Use provided crop settings or fall back to module defaults
        self.crop_enabled = crop_enabled if crop_enabled is not None else CROP_ENABLED
        self.crop_region = crop_region if crop_region is not None else CROP_REGION
        
    def initialize(self):
        """Initialize the Raspberry Pi camera."""
        if not CAMERA_AVAILABLE:
            raise RuntimeError("picamera2 library not installed. Install with: pip install picamera2")
        
        try:
            self.camera = Picamera2()
            
            # Determine output size based on crop settings
            if self.crop_enabled:
                # Use crop dimensions as output size
                output_width = self.crop_region[2]
                output_height = self.crop_region[3]
                print(f"Hardware crop enabled: {self.crop_region}")
                print(f"  Crop region: x={self.crop_region[0]}, y={self.crop_region[1]}, "
                      f"w={output_width}, h={output_height}")
            else:
                # Use full sensor resolution
                output_width = 4608
                output_height = 2592
                print("Hardware crop disabled - using full sensor")
            
            # Configure camera for still images (using settings from .env)
            # No preview/lores stream for faster performance
            config = self.camera.create_still_configuration(
                main={"size": (output_width, output_height)}
            )
            self.camera.configure(config)
            
            # Apply hardware crop if enabled (BEFORE camera start)
            if self.crop_enabled:
                try:
                    self.camera.set_controls({"ScalerCrop": self.crop_region})
                    print("Hardware ScalerCrop applied successfully")
                except Exception as crop_error:
                    print(f"Warning: Failed to apply ScalerCrop: {crop_error}")
                    print("  Continuing with full sensor...")
            
            # Set autofocus mode (Pi Camera v3)
            # AfMode: 0=Manual, 1=Auto (trigger), 2=Continuous
            if AUTOFOCUS_ENABLED:
                if AUTOFOCUS_MODE == 'continuous':
                    try:
                        self.camera.set_controls({"AfMode": 2})  # Continuous autofocus
                        print("Autofocus: Continuous mode (AfMode=2)")
                    except Exception as af_error:
                        print(f"Autofocus not available or failed: {af_error}")
                elif AUTOFOCUS_MODE == 'trigger':
                    try:
                        self.camera.set_controls({"AfMode": 1})  # Auto/trigger mode
                        print("Autofocus: Trigger mode (AfMode=1)")
                    except Exception as af_error:
                        print(f"Autofocus not available or failed: {af_error}")
                elif AUTOFOCUS_MODE == 'manual':
                    try:
                        self.camera.set_controls({
                            "AfMode": 0,  # Manual mode
                            "LensPosition": LENS_POSITION
                        })
                        print(f"Autofocus: Manual mode (AfMode=0, LensPosition={LENS_POSITION})")
                    except Exception as af_error:
                        print(f"Manual focus not available or failed: {af_error}")
                else:
                    print(f"Warning: Unknown autofocus mode '{AUTOFOCUS_MODE}', using default")
            else:
                try:
                    # When autofocus disabled, set manual mode with default lens position
                    self.camera.set_controls({
                        "AfMode": 0,  # Manual mode
                        "LensPosition": LENS_POSITION
                    })
                    print(f"Autofocus: Disabled (Manual mode, LensPosition={LENS_POSITION})")
                except Exception:
                    print("Autofocus: Disabled")
            
            self.camera.start()
            # Allow camera to warm up, adjust exposure, and autofocus to settle
            time.sleep(3)
            self.is_initialized = True
            
            crop_status = f"with crop {self.crop_region}" if self.crop_enabled else "at full resolution"
            print(f"Camera initialized successfully {crop_status}")
            print(f"  Output size: {output_width}x{output_height}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize camera: {e}")
    
    def capture_image(self, exposure_compensation=0.0):
        """
        Capture an image from the camera and return as numpy array.
        
        Args:
            exposure_compensation: Exposure compensation value in EV (-2.0 to +2.0)
                                   0 = normal exposure
                                   positive = brighter
                                   negative = darker
        """
        if not self.is_initialized:
            raise RuntimeError("Camera not initialized. Call initialize() first.")
        
        try:
            print(f"Capturing image with exposure compensation: {exposure_compensation:+.1f} EV")

            # Set exposure compensation if specified
            if exposure_compensation != 0.0:
                self.camera.set_controls({"ExposureValue": exposure_compensation})
                time.sleep(0.5)  # Allow exposure to adjust
            
            # Trigger autofocus cycle before capture (only in trigger mode)
            if AUTOFOCUS_ENABLED and AUTOFOCUS_MODE == 'trigger':
                try:
                    self.camera.autofocus_cycle()
                    time.sleep(0.3)  # Wait for focus to settle
                except Exception as e:
                    print(f"Warning: Autofocus trigger failed: {e}")
            
            # Capture image as numpy array
            frame = self.camera.capture_array()
            
            # Reset exposure compensation to normal
            if exposure_compensation != 0.0:
                self.camera.set_controls({"ExposureValue": 0.0})
            
            print("Image captured successfully.")
            return frame
        except Exception as e:
            raise RuntimeError(f"Failed to capture image: {e}")
    
    def capture_bracketed(self, exposures=[-1.0, 0.0, 1.0]):
        """
        Capture multiple images at different exposure levels (bracketing).
        
        Args:
            exposures: List of exposure compensation values in EV
                      Default: [-1.0, 0.0, 1.0] (underexposed, normal, overexposed)
        
        Returns:
            List of numpy arrays, one for each exposure level
        """
        if not self.is_initialized:
            raise RuntimeError("Camera not initialized. Call initialize() first.")
        
        frames = []
        print(f"Capturing {len(exposures)} bracketed exposures: {exposures}")
        
        for i, ev in enumerate(exposures):
            try:
                print(f"  Capture {i+1}/{len(exposures)} at EV {ev:+.1f}...")
                frame = self.capture_image(exposure_compensation=ev)
                frames.append(frame)
            except Exception as e:
                print(f"  Warning: Failed to capture at EV {ev}: {e}")
                continue
        
        return frames
    
    def set_crop_region(self, x, y, width, height):
        """
        Update the crop region dynamically (requires camera restart to apply).
        
        Args:
            x, y: Top-left corner in sensor coordinates (before rotation)
            width, height: Crop dimensions in sensor coordinates
        """
        self.crop_region = (x, y, width, height)
        self.crop_enabled = True
        print(f"Crop region updated to: {self.crop_region}")
        print("Note: Camera must be reinitialized for crop to take effect")
    
    def enable_crop(self, enabled=True):
        """Enable or disable hardware cropping."""
        self.crop_enabled = enabled
        status = "enabled" if enabled else "disabled"
        print(f"Hardware crop {status}")
        print("Note: Camera must be reinitialized for change to take effect")
    
    def get_crop_info(self):
        """Return current crop configuration."""
        return {
            'enabled': self.crop_enabled,
            'region': self.crop_region,
            'x': self.crop_region[0],
            'y': self.crop_region[1],
            'width': self.crop_region[2],
            'height': self.crop_region[3]
        }
    
    def close(self):
        """Close the camera and release resources."""
        if self.camera is not None:
            try:
                if self.debug: print("DEBUG: Stopping and closing camera...")
                self.camera.stop()
                self.camera.close()
                self.is_initialized = False
                if self.debug: print("DEBUG: Camera closed successfully")
            except Exception as e:
                print(f"Error closing camera: {e}")
>>>>+++ REPLACE

    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
