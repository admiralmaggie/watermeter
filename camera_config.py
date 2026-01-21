# Camera configuration for Raspberry Pi
# This module handles camera initialization and image capture

try:
    from picamera2 import Picamera2
    import time
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    print("WARNING: picamera2 not available. Camera features disabled.")

class WaterMeterCamera:
    """Wrapper for Raspberry Pi camera operations."""
    
    def __init__(self):
        self.camera = None
        self.is_initialized = False
        
    def initialize(self):
        """Initialize the Raspberry Pi camera."""
        if not CAMERA_AVAILABLE:
            raise RuntimeError("picamera2 library not installed. Install with: pip install picamera2")
        
        try:
            self.camera = Picamera2()
            # Configure camera for maximum resolution still images
            config = self.camera.create_still_configuration(
                main={"size": (4608, 2592)},  # Maximum resolution for Pi Camera v3
                lores={"size": (640, 480)},
                display="lores"
            )
            self.camera.configure(config)
            
            # Set autofocus mode if supported (Pi Camera v3)
            try:
                self.camera.set_controls({"AfMode": 2})  # 2 = Continuous autofocus
                print("Autofocus enabled (continuous mode)")
            except Exception as af_error:
                print(f"Autofocus not available or failed: {af_error}")
            
            self.camera.start()
            # Allow camera to warm up, adjust exposure, and autofocus to settle
            time.sleep(3)
            self.is_initialized = True
            print("Camera initialized successfully at 4608x2592 resolution")
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
            # Set exposure compensation if specified
            if exposure_compensation != 0.0:
                self.camera.set_controls({"ExposureValue": exposure_compensation})
                time.sleep(0.5)  # Allow exposure to adjust
            
            # Trigger autofocus cycle before capture
            self.camera.autofocus_cycle()
            time.sleep(0.3)  # Wait for focus to settle
            
            # Capture image as numpy array
            frame = self.camera.capture_array()
            
            # Reset exposure compensation to normal
            if exposure_compensation != 0.0:
                self.camera.set_controls({"ExposureValue": 0.0})
            
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
    
    def close(self):
        """Close the camera and release resources."""
        if self.camera is not None:
            try:
                self.camera.stop()
                self.camera.close()
                self.is_initialized = False
                print("Camera closed successfully")
            except Exception as e:
                print(f"Error closing camera: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
