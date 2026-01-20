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
            # Configure camera for high-quality still images
            config = self.camera.create_still_configuration(
                main={"size": (1920, 1080)},  # Adjust resolution as needed
                lores={"size": (640, 480)},
                display="lores"
            )
            self.camera.configure(config)
            self.camera.start()
            # Allow camera to warm up and adjust exposure
            time.sleep(2)
            self.is_initialized = True
            print("Camera initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize camera: {e}")
    
    def capture_image(self):
        """Capture an image from the camera and return as numpy array."""
        if not self.is_initialized:
            raise RuntimeError("Camera not initialized. Call initialize() first.")
        
        try:
            # Capture image as numpy array (BGR format compatible with OpenCV)
            frame = self.camera.capture_array()
            return frame
        except Exception as e:
            raise RuntimeError(f"Failed to capture image: {e}")
    
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
