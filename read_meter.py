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
HOUGH_TARGET_WIDTH = 1000
HOUGH_DP = float(os.getenv('HOUGH_DP', '1.2'))
HOUGH_PARAM1 = int(os.getenv('HOUGH_PARAM1', '120'))
HOUGH_PARAM2 = int(os.getenv('HOUGH_PARAM2', '45'))
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
MOTION_DETECTION_ENABLED = os.getenv('MOTION_DETECTION_ENABLED', 'false').lower() == 'true'
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
    
    def create_circular_mask(self, frame_shape, center, radius):
        height, width = frame_shape[:2]
        y, x = np.ogrid[:height, :width]
        cx, cy = center
        mask = ((x - cx)**2 + (y - cy)**2 <= radius**2).astype(np.uint8) * 255
        return mask
    
    def detect_motion_in_zone(self, current_gray, prev_gray, mask):
        frame_diff = cv2.absdiff(current_gray, prev_gray)
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        thresh_masked = cv2.bitwise_and(thresh, thresh, mask=mask)
        changed_pixels = cv2.countNonZero(thresh_masked)
        total_pixels = cv2.countNonZero(mask)
        if total_pixels == 0: return False, 0.0
        change_percentage = (changed_pixels / total_pixels) * 100
        return change_percentage >= self.threshold, change_percentage
    
    def process_frame(self, frame):
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return (self.motion_detected_zone1, self.motion_detected_zone2, 0.0, 0.0)
        
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.GaussianBlur(current_gray, (21, 21), 0)
        
        if self.prev_frame is None:
            self.prev_frame = current_gray
            return (False, False, 0.0, 0.0)
        
        mask1 = self.create_circular_mask(frame.shape, (self.zone1[0], self.zone1[1]), self.zone1[2])
        mask2 = self.create_circular_mask(frame.shape, (self.zone2[0], self.zone2[1]), self.zone2[2])
        
        motion1, percent1 = self.detect_motion_in_zone(current_gray, self.prev_frame, mask1)
        motion2, percent2 = self.detect_motion_in_zone(current_gray, self.prev_frame, mask2)
        
        self.motion_detected_zone1 = motion1
        self.motion_detected_zone2 = motion2
        
        if motion1: self.last_motion_time_zone1 = time.time()
        if motion2: self.last_motion_time_zone2 = time.time()
        
        self.prev_frame = current_gray
        return (motion1, motion2, percent1, percent2)
    
    def draw_zones(self, frame):
        color1 = COLOR_RED if self.motion_detected_zone1 else COLOR_GREEN
        cv2.circle(frame, (self.zone1[0], self.zone1[1]), self.zone1[2], color1, 2)
        cv2.putText(frame, "Zone 1", (self.zone1[0] - 30, self.zone1[1] - self.zone1[2] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color1, 2)
        
        color2 = COLOR_RED if self.motion_detected_zone2 else COLOR_GREEN
        cv2.circle(frame, (self.zone2[0], self.zone2[1]), self.zone2[2], color2, 2)
        cv2.putText(frame, "Zone 2", (self.zone2[0] - 30, self.zone2[1] - self.zone2[2] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color2, 2)
        return frame


def parse_manual_circles(circles_str):
    if not circles_str: return []
    circles = []
    try:
        for s in circles_str.split(';'):
            parts = s.strip().split(',')
            if len(parts) == 3:
                circles.append((int(parts[0]), int(parts[1]), int(parts[2])))
        return circles
    except: return []


def _resize_for_hough(frame, target_width=HOUGH_TARGET_WIDTH):
    h0, w0 = frame.shape[:2]
    if w0 <= target_width: return frame, 1.0
    scale = target_width / float(w0)
    resized = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def _dedupe_circles(circles, center_tol_frac=0.25, radius_tol_frac=0.25):
    kept = []
    for (x, y, r) in circles:
        dup = False
        for (kx, ky, kr) in kept:
            tol = center_tol_frac * min(r, kr)
            if (x - kx) ** 2 + (y - ky) ** 2 <= tol ** 2 and abs(r - kr) <= radius_tol_frac * min(r, kr):
                dup = True; break
        if not dup: kept.append((x, y, r))
    return kept


def _edge_strength_score(grad_mag, x, y, r, samples=72):
    h, w = grad_mag.shape[:2]
    angles = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    xs = np.clip((x + r * np.cos(angles)).round().astype(np.int32), 0, w - 1)
    ys = np.clip((y + r * np.sin(angles)).round().astype(np.int32), 0, h - 1)
    return float(np.mean(grad_mag[ys, xs]))


def select_best_circles(circles_raw, gray, expected_count):
    if circles_raw is None: return []
    circles = np.round(circles_raw[0, :]).astype(int)
    circles = _dedupe_circles([(int(x), int(y), int(r)) for (x, y, r) in circles if r > 0])
    if len(circles) <= expected_count: return circles
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)
    scored = sorted([(_edge_strength_score(grad_mag, x, y, r), (x, y, r)) for (x, y, r) in circles], 
                    key=lambda t: t[0], reverse=True)
    return [c for _, c in scored[:expected_count]]


def get_needle_mask(hsv_roi):
    lower1 = np.array([RED_HUE_LOWER1, RED_SAT_LOWER1, RED_VAL_LOWER1], dtype=np.uint8)
    upper1 = np.array([RED_HUE_UPPER1, 255, 255], dtype=np.uint8)
    lower2 = np.array([RED_HUE_LOWER2, RED_SAT_LOWER2, RED_VAL_LOWER2], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    mask = cv2.bitwise_or(cv2.inRange(hsv_roi, lower1, upper1), cv2.inRange(hsv_roi, lower2, upper2))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def find_needle(image, cx, cy, radius):
    x_min, y_min = max(0, int(cx - radius)), max(0, int(cy - radius))
    x_max, y_max = min(image.shape[1], int(cx + radius)), min(image.shape[0], int(cy + radius))
    roi = image[y_min:y_max, x_min:x_max]
    if roi.size == 0: return 0, (cx, cy)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = get_needle_mask(hsv)
    ring_mask = np.zeros_like(mask)
    center_roi = (cx - x_min, cy - y_min)
    cv2.circle(ring_mask, (int(center_roi[0]), int(center_roi[1])), int(radius * 0.9), 255, -1)
    cv2.circle(ring_mask, (int(center_roi[0]), int(center_roi[1])), int(radius * 0.2), 0, -1)
    target_mask = cv2.bitwise_and(mask, ring_mask)
    ys, xs = np.where(target_mask > 0)
    if xs.size > NEEDLE_MIN_PIXELS:
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = float(vx[0]), float(vy[0]), float(x0[0]), float(y0[0])
        t = (pts[:, 0] - x0) * vx + (pts[:, 1] - y0) * vy
        p1 = np.array([x0 + t.min() * vx, y0 + t.min() * vy])
        p2 = np.array([x0 + t.max() * vx, y0 + t.max() * vy])
        center = np.array([center_roi[0], center_roi[1]])
        tip = p1 if np.linalg.norm(p1 - center) > np.linalg.norm(p2 - center) else p2
        angle_deg = np.degrees(np.arctan2(tip[1] - center_roi[1], tip[0] - center_roi[0]))
        best_value = 10 * ((angle_deg + 90) % 360) / 360
        return best_value, (int(x_min + tip[0]), int(y_min + tip[1]))
    return 0, (cx, cy)


def process_values(values):
    values_rev = values[::-1]
    digits = []
    for i, v in enumerate(values_rev):
        whole = int(np.floor(v))
        if i < len(values_rev) - 1:
            if (v - whole) < 0.5 and values_rev[i+1] > 5: whole -= 1
        digits.append(str(whole % 10))
    if len(digits) >= 3: return "".join(digits[:-1]) + "." + digits[-1]
    return "".join(digits)


def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, 
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def get_frame_from_camera(camera, args):
    if hasattr(args, 'bracket') and args.bracket:
        exposures = [float(x.strip()) for x in args.exposures.split(",")]
        frames = [cv2.rotate(cv2.cvtColor(f, cv2.COLOR_RGB2BGR), cv2.ROTATE_90_CLOCKWISE) 
                  for f in camera.capture_bracketed(exposures)]
        frame = select_best_exposure(frames)
    else:
        frame = cv2.rotate(cv2.cvtColor(camera.capture_image(), cv2.COLOR_RGB2BGR), cv2.ROTATE_90_CLOCKWISE)
    if FINE_ROTATION_ANGLE != 0: frame = rotate_image(frame, FINE_ROTATION_ANGLE)
    return frame


def read_meter_from_frame(frame, motion_detector=None, show_gui=False):
    if frame is None: return None, False, None
    output = frame.copy()
    motion_detected = False
    if motion_detector:
        m1, m2, _, _ = motion_detector.process_frame(frame)
        motion_detected = m1 or m2
        if show_gui: output = motion_detector.draw_zones(output)
    
    if USE_MANUAL_CIRCLES:
        circles = sorted(parse_manual_circles(MANUAL_CIRCLES_STR), key=lambda c: c[0])
    else:
        h_frame, scale = _resize_for_hough(frame)
        gray = cv2.GaussianBlur(cv2.cvtColor(h_frame, cv2.COLOR_BGR2GRAY), (5, 5), 1.5)
        mind = min(gray.shape[:2])
        circles_raw = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=HOUGH_DP, 
                                      minDist=max(10, int(mind * RADIUS_MIN_FRAC)), 
                                      param1=HOUGH_PARAM1, param2=HOUGH_PARAM2, 
                                      minRadius=int(mind * RADIUS_MIN_FRAC), 
                                      maxRadius=int(mind * RADIUS_MAX_FRAC))
        circles = []
        for (x, y, r) in select_best_circles(circles_raw, gray, DIALS_COUNT):
            circles.append((int(round(x/scale)), int(round(y/scale)), int(round(r/scale))))
        circles = sorted(circles, key=lambda c: c[0])

    if len(circles) != DIALS_COUNT:
        for (x, y, r) in circles: cv2.circle(output, (x, y), r, COLOR_ORANGE, 3)
        return None, motion_detected, output

    values = []
    for i, (x, y, r) in enumerate(circles):
        val, tip = find_needle(output, x, y, r)
        values.append(val)
        if show_gui:
            cv2.circle(output, (x, y), r, COLOR_GREEN, 3)
            cv2.line(output, (x, y), tip, COLOR_MAGENTA, 2)
            cv2.putText(output, f"{val:.1f}", (x-20, y+r+20), cv2.FONT_HERSHEY_PLAIN, 1, COLOR_BLUE)
    
    reading = process_values(values)
    if show_gui:
        cv2.putText(output, f"Reading: {reading}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_BLUE, 2)
    return reading, motion_detected, output


def find_circles(frame, motion_detector=None):
    r, _, _ = read_meter_from_frame(frame, motion_detector, not HEADLESS)
    return r


def select_best_exposure(frames):
    if len(frames) == 1: return frames[0]
    scored = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten() / gray.size
        score = (cv2.Laplacian(gray, cv2.CV_64F).var() * 0.3 + gray.std() * 0.5) - (hist[:30].sum() + hist[226:].sum()) * 100
        scored.append((score, f))
    return max(scored, key=lambda t: t[0])[1]


def read_value(value, convention):
    return (10. - value) % 10 if convention == "CCW" else value % 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', action='store_true')
    parser.add_argument('--file', type=str, default='test.jpeg')
    parser.add_argument('--continuous', action='store_true')
    parser.add_argument('--interval', type=int, default=5)
    args = parser.parse_args()
    
    if args.camera:
        with WaterMeterCamera() as cam:
            try:
                while True:
                    frame = get_frame_from_camera(cam, args)
                    read_meter_from_frame(frame, show_gui=True)
                    if not args.continuous: break
                    time.sleep(args.interval)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
            except KeyboardInterrupt: pass
    else:
        frame = cv2.imread(args.file)
        if frame is not None: read_meter_from_frame(frame, show_gui=True); cv2.waitKey(0)

if __name__ == "__main__":
    main()
