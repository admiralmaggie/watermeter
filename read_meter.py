import cv2
import numpy as np
import time
import sys
import argparse
import os
from dotenv import load_dotenv
from camera_config import WaterMeterCamera, CAMERA_AVAILABLE

# Load environment variables from .env file
load_dotenv()

# Detect if we're running in headless mode (no display available)
HEADLESS = os.environ.get('DISPLAY', '') == '' or os.environ.get('HEADLESS', '0') == '1'

HORIZONTAL_MAX_DIFF = 1000
COLOR_ORANGE = (0,128,255)
COLOR_MAGENTA = (255,0,255)
COLOR_GREEN = (0,255,0)
COLOR_RED = (0,0,255)
COLOR_BLUE = (255,0,0)

# Configuration from .env
DIALS_COUNT = int(os.getenv('DIALS_COUNT', '3'))
USE_MANUAL_CIRCLES = os.getenv('USE_MANUAL_CIRCLES', 'false').lower() == 'true'
MANUAL_CIRCLES_STR = os.getenv('MANUAL_CIRCLES', '')
SAVE_IMAGE = False

# Fine rotation adjustment (in degrees, negative = clockwise) - from .env
FINE_ROTATION_ANGLE = float(os.getenv('FINE_ROTATION_ANGLE', '-4'))

# Toggle verbose visualization + saving intermediate masks for debugging
# Automatically disabled in headless mode - can be overridden in .env
DEBUG_NEEDLE = os.getenv('DEBUG_NEEDLE', 'true').lower() == 'true' and not HEADLESS

# --- Circle detection tuning (loaded from .env) ---
# HoughCircles is sensitive to contrast/edges. The most robust approach I've found
# here is:
#   1) Run Hough on a *downscaled* grayscale image (faster + fewer noisy edges)
#   2) Restrict radius range to the expected dial size (exclude the big outer rim)
#   3) If extras remain, score candidates by edge-strength around the circumference
#      and keep the best DIALS_COUNT.
HOUGH_TARGET_WIDTH = 1000
HOUGH_DP = float(os.getenv('HOUGH_DP', '1.2'))
HOUGH_PARAM1 = int(os.getenv('HOUGH_PARAM1', '120'))
HOUGH_PARAM2 = int(os.getenv('HOUGH_PARAM2', '45'))  # accumulator threshold; increase to reduce false circles
RADIUS_MIN_FRAC = float(os.getenv('RADIUS_MIN_FRAC', '0.04'))
RADIUS_MAX_FRAC = float(os.getenv('RADIUS_MAX_FRAC', '0.12'))

# Needle detection settings (loaded from .env)
NEEDLE_MIN_PIXELS = int(os.getenv('NEEDLE_MIN_PIXELS', '80'))

# HSV thresholds for red needle detection (loaded from .env)
RED_HUE_LOWER1 = int(os.getenv('RED_HUE_LOWER1', '0'))
RED_SAT_LOWER1 = int(os.getenv('RED_SAT_LOWER1', '80'))
RED_VAL_LOWER1 = int(os.getenv('RED_VAL_LOWER1', '60'))
RED_HUE_UPPER1 = int(os.getenv('RED_HUE_UPPER1', '10'))

RED_HUE_LOWER2 = int(os.getenv('RED_HUE_LOWER2', '170'))
RED_SAT_LOWER2 = int(os.getenv('RED_SAT_LOWER2', '80'))
RED_VAL_LOWER2 = int(os.getenv('RED_VAL_LOWER2', '60'))

# Motion detection settings (loaded from .env)
MOTION_ZONE1_X = int(os.getenv('MOTION_ZONE1_X', '200'))
MOTION_ZONE1_Y = int(os.getenv('MOTION_ZONE1_Y', '400'))
MOTION_ZONE1_R = int(os.getenv('MOTION_ZONE1_R', '100'))
MOTION_ZONE2_X = int(os.getenv('MOTION_ZONE2_X', '600'))
MOTION_ZONE2_Y = int(os.getenv('MOTION_ZONE2_Y', '400'))
MOTION_ZONE2_R = int(os.getenv('MOTION_ZONE2_R', '100'))
MOTION_THRESHOLD = float(os.getenv('MOTION_THRESHOLD', '2.0'))
MOTION_FRAME_SKIP = int(os.getenv('MOTION_FRAME_SKIP', '3'))


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
            print(f"  Motion detected in Zone 1 ({percent1:.2f}% change)")
        if motion2:
            self.last_motion_time_zone2 = time.time()
            print(f"  Motion detected in Zone 2 ({percent2:.2f}% change)")
        
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
        
        return frame


def parse_manual_circles(circles_str):
    """
    Parse manual circle coordinates from string.
    Format: x1,y1,r1;x2,y2,r2;x3,y3,r3
    Returns: List of (x, y, r) tuples
    """
    if not circles_str or circles_str.strip() == '':
        return []
    
    circles = []
    try:
        for circle_str in circles_str.split(';'):
            parts = circle_str.strip().split(',')
            if len(parts) == 3:
                x = int(parts[0].strip())
                y = int(parts[1].strip())
                r = int(parts[2].strip())
                circles.append((x, y, r))
        return circles
    except Exception as e:
        print(f"ERROR: Failed to parse manual circles: {e}")
        return []


def _resize_for_hough(frame: np.ndarray, target_width: int = HOUGH_TARGET_WIDTH):
    """Return (resized_frame, scale) where original = resized / scale."""
    h0, w0 = frame.shape[:2]
    if w0 <= target_width:
        return frame, 1.0
    scale = target_width / float(w0)
    resized = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def _dedupe_circles(circles, center_tol_frac: float = 0.25, radius_tol_frac: float = 0.25):
    """Remove near-duplicate circles (same dial found multiple times)."""
    kept = []
    for (x, y, r) in circles:
        dup = False
        for (kx, ky, kr) in kept:
            tol = center_tol_frac * min(r, kr)
            if (x - kx) ** 2 + (y - ky) ** 2 <= tol ** 2 and abs(r - kr) <= radius_tol_frac * min(r, kr):
                dup = True
                break
        if not dup:
            kept.append((x, y, r))
    return kept


def _edge_strength_score(grad_mag: np.ndarray, x: int, y: int, r: int, samples: int = 72) -> float:
    """Mean gradient magnitude sampled around a circle."""
    h, w = grad_mag.shape[:2]
    angles = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    xs = (x + r * np.cos(angles)).round().astype(np.int32)
    ys = (y + r * np.sin(angles)).round().astype(np.int32)
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)
    return float(np.mean(grad_mag[ys, xs]))


def select_best_circles(circles, gray: np.ndarray, expected_count: int):
    """Given raw Hough circles, return a list of (x,y,r) for the best expected_count."""
    if circles is None:
        return []

    circles = np.round(circles[0, :]).astype(int)
    circles = [(int(x), int(y), int(r)) for (x, y, r) in circles]

    # Dedupe and basic sanity filter
    circles = _dedupe_circles(circles)
    circles = [(x, y, r) for (x, y, r) in circles if r > 0]

    if len(circles) <= expected_count:
        return circles

    # Score by edge strength around the circumference
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)

    scored = []
    for (x, y, r) in circles:
        score = _edge_strength_score(grad_mag, x, y, r)
        scored.append((score, (x, y, r)))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [c for _, c in scored[:expected_count]]

def get_needle_mask(hsv_roi):
    """Isolate red pixels in the ROI and return a cleaned mask."""
    # Red hue wraps around, so we combine two ranges.
    # Thresholds loaded from .env configuration
    lower1 = np.array([RED_HUE_LOWER1, RED_SAT_LOWER1, RED_VAL_LOWER1], dtype=np.uint8)
    upper1 = np.array([RED_HUE_UPPER1, 255, 255], dtype=np.uint8)
    lower2 = np.array([RED_HUE_LOWER2, RED_SAT_LOWER2, RED_VAL_LOWER2], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    
    mask1 = cv2.inRange(hsv_roi, lower1, upper1)
    mask2 = cv2.inRange(hsv_roi, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Clean up small noise / fill gaps
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask

def find_needle(image, cx, cy, radius):
    """
    Finds the needle by isolating red pixels and fitting a line to estimate direction.
    """
    x_min, y_min = max(0, int(cx - radius)), max(0, int(cy - radius))
    x_max, y_max = min(image.shape[1], int(cx + radius)), min(image.shape[0], int(cy + radius))
    roi = image[y_min:y_max, x_min:x_max]
    
    if roi.size == 0:
        return 0, (cx, cy)
        
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = get_needle_mask(hsv)

    if DEBUG_NEEDLE and not HEADLESS:
        cv2.imshow(f"needle_mask_{int(cx)}", mask)

    # Create a mask for the ring where the needle's pointer (triangle) should be
    ring_mask = np.zeros_like(mask)
    center_roi = (cx - x_min, cy - y_min)
    cv2.circle(ring_mask, (int(center_roi[0]), int(center_roi[1])), int(radius * 0.9), 255, -1)
    cv2.circle(ring_mask, (int(center_roi[0]), int(center_roi[1])), int(radius * 0.2), 0, -1)
    
    # Apply the ring mask to our needle mask
    target_mask = cv2.bitwise_and(mask, ring_mask)

    if DEBUG_NEEDLE and not HEADLESS:
        cv2.imshow(f"needle_target_{int(cx)}", target_mask)
    
    # Robust direction estimation:
    # Fit a line through ALL candidate needle pixels and choose the endpoint farthest
    # from the dial center as the needle tip.
    ys, xs = np.where(target_mask > 0)
    if xs.size > NEEDLE_MIN_PIXELS:
        pts = np.stack([xs, ys], axis=1).astype(np.float32)

        # Fit line (vx,vy) through points; (x0,y0) is a point on the line
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = float(vx[0]), float(vy[0]), float(x0[0]), float(y0[0])

        # Project points onto line and take extreme projections to estimate endpoints
        t = (pts[:, 0] - x0) * vx + (pts[:, 1] - y0) * vy
        t_min, t_max = float(t.min()), float(t.max())
        p1 = np.array([x0 + t_min * vx, y0 + t_min * vy], dtype=np.float32)
        p2 = np.array([x0 + t_max * vx, y0 + t_max * vy], dtype=np.float32)

        center = np.array([center_roi[0], center_roi[1]], dtype=np.float32)
        d1 = float(np.linalg.norm(p1 - center))
        d2 = float(np.linalg.norm(p2 - center))
        tip = p1 if d1 > d2 else p2
        tip_x, tip_y = float(tip[0]), float(tip[1])
        
        # Vector from dial center to the furthest point (the tip)
        dx, dy = tip_x - center_roi[0], tip_y - center_roi[1]
        
        # Calculate angle (0 is Top)
        angle_deg = np.degrees(np.arctan2(dy, dx))
        normalized_angle = (angle_deg + 90) % 360
        best_value = 10 * normalized_angle / 360
        needle_tip = (int(x_min + tip_x), int(y_min + tip_y))

        if DEBUG_NEEDLE:
            overlay = roi.copy()
            cv2.line(overlay, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 255, 0), 2)
            cv2.circle(overlay, (int(tip_x), int(tip_y)), 6, (255, 0, 255), -1)
            cv2.circle(overlay, (int(center_roi[0]), int(center_roi[1])), 4, (0, 255, 255), -1)
            
            if not HEADLESS:
                cv2.imshow(f"needle_overlay_{int(cx)}", overlay)

            # Save debug artifacts for offline inspection
            cv2.imwrite(f"_debug_needle_mask_{int(cx)}.png", mask)
            cv2.imwrite(f"_debug_needle_target_{int(cx)}.png", target_mask)
            cv2.imwrite(f"_debug_needle_overlay_{int(cx)}.png", overlay)
        
        print(f"DEBUG: Needle tip found at ({tip_x:.1f}, {tip_y:.1f}), angle: {normalized_angle:.1f} deg, value: {best_value:.2f}")
    else:
        print("DEBUG: No dark pixels found in needle ring.")
        best_value, needle_tip = 0, (cx, cy)

    return best_value, needle_tip

def process_values(values):
    reading = ''
    for i, (v) in enumerate(values):
        whole = int(np.floor(v))
        if i == len(values) - 1:
            reading = reading + str(whole)
            break
        decimals = v - whole
        if decimals < 0.5 and values[i+1] > 5:
            # decimal value low but the next value is high, so need to adjust the reading by -1
            whole = whole-1
        reading = reading + str(whole)

    return reading

def rotate_image(image, angle):
    """Rotate image by specified angle (degrees) around center.
    Positive angle = counter-clockwise, Negative angle = clockwise"""
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                              flags=cv2.INTER_LINEAR, 
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(255, 255, 255))
    return rotated

def get_frame_from_camera(camera, args):
    """Capture and preprocess a frame from the camera."""
    if args.bracket:
        exposures = [float(x.strip()) for x in args.exposures.split(",")]
        frames_raw = camera.capture_bracketed(exposures)
        
        frames = []
        for frame in frames_raw:
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frames.append(frame)
        
        print("Selecting best exposure...")
        frame = select_best_exposure(frames)
    else:
        frame = camera.capture_image()
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    
    if FINE_ROTATION_ANGLE != 0:
        frame = rotate_image(frame, FINE_ROTATION_ANGLE)
        
    return frame

def find_circles(frame, motion_detector=None):
    if frame is None:
        print("DEBUG: Error: Received None frame in find_circles.")
        return

    print(f"DEBUG: find_circles started. Frame shape: {frame.shape}")

    if SAVE_IMAGE:
        filename = time.strftime("data/sample-%Y%m%d-%H%M.jpg")
        cv2.imwrite(filename, frame)

    output = frame.copy()
    
    # Process motion detection if enabled
    if motion_detector is not None:
        motion1, motion2, percent1, percent2 = motion_detector.process_frame(frame)
        output = motion_detector.draw_zones(output)

    # Check if using manual circles
    if USE_MANUAL_CIRCLES:
        print("DEBUG: Using manual circle coordinates from .env")
        circles = parse_manual_circles(MANUAL_CIRCLES_STR)
        
        if len(circles) == 0:
            print("ERROR: Manual circles enabled but no valid coordinates provided.")
            print("      Set MANUAL_CIRCLES in .env file (format: x1,y1,r1;x2,y2,r2;x3,y3,r3)")
            return
        
        if len(circles) != DIALS_COUNT:
            print(f"WARNING: Found {len(circles)} manual circles but expected {DIALS_COUNT}")
        
        print(f"DEBUG: Loaded {len(circles)} manual circles: {circles}")
        
        # Sort by X so dials are left-to-right
        circles = sorted(circles, key=lambda c: c[0])
    else:
        # Dynamic circle detection using Hough Transform
        print("DEBUG: Detecting circles dynamically using HoughCircles...")
        hough_frame, scale = _resize_for_hough(frame)
        gray = cv2.cvtColor(hough_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1.5)

        # Dynamic radius bounds based on image size
        mind = min(gray.shape[0], gray.shape[1])
        min_radius = max(10, int(mind * RADIUS_MIN_FRAC))
        max_radius = max(min_radius + 1, int(mind * RADIUS_MAX_FRAC))

        print(
            f"DEBUG: HoughCircles params: scale={scale:.3f}, minR={min_radius}, "
            f"maxR={max_radius}, param2={HOUGH_PARAM2}"
        )
        
        circles_raw = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=HOUGH_DP,
            minDist=max(10, min_radius),
            param1=HOUGH_PARAM1,
            param2=HOUGH_PARAM2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )

        if circles_raw is None:
            print("DEBUG: No circles found by HoughCircles.")
            return

        # Select the best dials even if Hough finds extras
        selected = select_best_circles(circles_raw, gray, DIALS_COUNT)

        # Map circle coords back to original resolution
        circles = []
        for (x, y, r) in selected:
            if scale != 1.0:
                x = int(round(x / scale))
                y = int(round(y / scale))
                r = int(round(r / scale))
            circles.append((x, y, r))

        # Sort by X so dials are left-to-right
        circles = sorted(circles, key=lambda c: c[0])

    # TODO: move to config. In the provided images, all dials appear to be Clockwise (CW).
    readout_conventions = ["CW"] * DIALS_COUNT

    # DEBUG: show selected circles
    if not HEADLESS:
        debug_output = frame.copy()
        for (x, y, r) in circles:
            cv2.circle(debug_output, (x, y), r, COLOR_GREEN, 3)
        cv2.imshow("selected_circles", debug_output)

    # ignore results if an exact number of dials wasn't found
    if len(circles) != DIALS_COUNT:
        print(
            f"DEBUG: Found {len(circles)} selected circles, but expected {DIALS_COUNT}. "
            "Skipping processing. (Try increasing HOUGH_PARAM2 to reduce false circles.)"
        )
        cv2.imshow("output", output)
        return

    values = []

    # loop over the (x, y) coordinates and radius of the circles
    minx = 0
    miny = 0
    radius = 0
    for i, ((x, y, r), convention) in enumerate(zip(circles, readout_conventions)):
        print(f"DEBUG: Processing circle #{i} at ({x}, {y}) with radius {r}")
        value, tip = find_needle(output, x, y, r)
        actual_value = read_value(value, convention)
        values.append(actual_value)
        print("#%i: (%i, %i) radius: %i - value: %f" % (i, x, y, r, actual_value))

        # draw needle and value
        cv2.line(output, (x, y), tip, COLOR_MAGENTA, thickness=2)
        cv2.putText(output, str(actual_value), (x - 20, y + r + 20), cv2.FONT_HERSHEY_PLAIN, 1, 255)

        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(output, (x, y), r, COLOR_GREEN, 4)
        cv2.rectangle(output, (x - 2, y - 2), (x + 2, y + 2), COLOR_ORANGE, -1)

        if i == 0:
            minx = x
            miny = y
            radius = r

    # TODO: compare to the previous reading? it should never be less than the previous one
    reading = process_values(values)
    print("Final reading: %s" % reading)
    cv2.putText(output, reading, (minx, miny + radius + 100), cv2.FONT_HERSHEY_PLAIN, 2, COLOR_BLUE)

    if SAVE_IMAGE:
        filename = time.strftime("data/sample-%Y%m%d-%H%M-out.jpg")
        cv2.imwrite(filename, output)

    if not HEADLESS:
        cv2.imshow("output", output)

def select_best_exposure(frames):
    """
    Select the best exposed image from a bracketed set based on image quality metrics.
    
    Args:
        frames: List of numpy arrays (BGR images)
    
    Returns:
        The frame with the best exposure (highest contrast/sharpness in mid-tones)
    """
    if len(frames) == 1:
        return frames[0]
    
    best_score = -1
    best_frame = frames[0]
    
    for i, frame in enumerate(frames):
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate sharpness using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Calculate contrast using standard deviation
        contrast = gray.std()
        
        # Calculate histogram distribution (prefer balanced exposures)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # Penalize over/underexposure (too many pixels at extremes)
        underexposed = hist[:30].sum()
        overexposed = hist[226:].sum()
        exposure_penalty = (underexposed + overexposed) * 100
        
        # Combined score (higher is better)
        score = (sharpness * 0.3 + contrast * 0.5) - exposure_penalty
        
        print(f"  Exposure {i+1}: sharpness={sharpness:.1f}, contrast={contrast:.1f}, "
              f"under={underexposed:.3f}, over={overexposed:.3f}, score={score:.1f}")
        
        if score > best_score:
            best_score = score
            best_frame = frame
    
    return best_frame

def read_value(value, convention):
    if convention == "CCW":
        result = 10. - value
    else:
        result = value
    if result == 10:
        result = 0
    return result

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Water Meter Reader - Analyze water meter dials')
    parser.add_argument('--camera', action='store_true', 
                       help='Use Raspberry Pi camera instead of image file')
    parser.add_argument('--file', type=str, default='test.jpeg',
                       help='Path to image file (default: test.jpeg)')
    parser.add_argument('--continuous', action='store_true',
                       help='Continuous capture mode (camera only)')
    parser.add_argument('--save', action='store_true',
                       help='Save captured/processed images')
    parser.add_argument('--interval', type=int, default=5,
                       help='Seconds between captures in continuous mode (default: 5)')
    parser.add_argument('--bracket', action='store_true',
                       help='Use exposure bracketing (capture multiple exposures and select best)')
    parser.add_argument('--exposures', type=str, default='-1.0,0.0,1.0',
                       help='Comma-separated EV values for bracketing (default: -1.0,0.0,1.0)')
    parser.add_argument('--motion', action='store_true',
                       help='Enable motion detection (uses zones from .env)')
    parser.add_argument('--debug', action='store_true',
                       help='Show detailed debugging information')
    
    args = parser.parse_args()
    
    global SAVE_IMAGE
    SAVE_IMAGE = args.save
    
    if args.camera:
        if not CAMERA_AVAILABLE:
            print("ERROR: Camera mode requested but picamera2 is not available.")
            print("Install with: pip install picamera2")
            sys.exit(1)
        
        print("Starting camera mode...")
        
        # Initialize motion detector if requested
        motion_det = None
        if args.motion:
            zone1 = (MOTION_ZONE1_X, MOTION_ZONE1_Y, MOTION_ZONE1_R)
            zone2 = (MOTION_ZONE2_X, MOTION_ZONE2_Y, MOTION_ZONE2_R)
            motion_det = MotionDetector(zone1, zone2, MOTION_THRESHOLD, MOTION_FRAME_SKIP)
        
        with WaterMeterCamera(debug=args.debug) as camera:
            if args.continuous:
                print(f"Continuous capture mode - interval: {args.interval}s. Press Ctrl+C to stop.")
                try:
                    while True:
                        print(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} - Capturing image...")
                        frame = get_frame_from_camera(camera, args)
                        find_circles(frame, motion_det)
                        
                        print(f"Waiting {args.interval} seconds...")
                        time.sleep(args.interval)
                        if not HEADLESS:
                            cv2.destroyAllWindows()
                except KeyboardInterrupt:
                    print("\nStopping continuous capture...")
            else:
                print("Single capture mode...")
                frame = get_frame_from_camera(camera, args)
                find_circles(frame, motion_det)
                
                if not HEADLESS:
                    print("DEBUG: Processing complete. Press any key to exit...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print("DEBUG: Processing complete (headless mode - no display)")
    else:
        # File mode (original behavior)
        image_path = args.file
        print(f"Starting script using image: {image_path}")
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"ERROR: Could not read image from {image_path}")
            sys.exit(1)
        
        # File mode doesn't support motion detection
        find_circles(frame, None)
        
        if not HEADLESS:
            print("DEBUG: Processing complete. Press any key to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("DEBUG: Processing complete (headless mode - no display)")

if __name__ == "__main__":
    main()
