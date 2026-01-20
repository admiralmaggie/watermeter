import cv2
import numpy as np
import time
import sys
import argparse
from matplotlib import pyplot as plt
from camera_config import WaterMeterCamera, CAMERA_AVAILABLE

HORIZONTAL_MAX_DIFF = 1000
COLOR_ORANGE = (0,128,255)
COLOR_MAGENTA = (255,0,255)
COLOR_GREEN = (0,255,0)
COLOR_RED = (0,0,255)
COLOR_BLUE = (255,0,0)

# Allow overriding the image from the command line:
#   python test.py test1.jpeg
IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else 'test.jpeg'
DIALS_COUNT = 3
SAVE_IMAGE = False
fig, ax = plt.subplots(figsize=(6, 6))

# Toggle verbose visualization + saving intermediate masks for debugging
DEBUG_NEEDLE = True

# --- Circle detection tuning ---
# HoughCircles is sensitive to contrast/edges. The most robust approach I've found
# here is:
#   1) Run Hough on a *downscaled* grayscale image (faster + fewer noisy edges)
#   2) Restrict radius range to the expected dial size (exclude the big outer rim)
#   3) If extras remain, score candidates by edge-strength around the circumference
#      and keep the best DIALS_COUNT.
HOUGH_TARGET_WIDTH = 1000
HOUGH_DP = 1.2
HOUGH_PARAM1 = 120
HOUGH_PARAM2 = 45  # accumulator threshold; increase to reduce false circles
RADIUS_MIN_FRAC = 0.04
RADIUS_MAX_FRAC = 0.12


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

def filter_circles(circles):
    print(f"DEBUG: filter_circles received {len(circles[0]) if circles is not None else 0} raw circles")
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
    # sort by X-axis
    circles = sorted(circles, key=lambda x: x[0])

    print(f"DEBUG: Circles after sorting by X-axis: {circles}")

    # remove circles with Y-axis deviating too much from the rest
    valid_circles = []
    min_y = None
    for c in circles:
        y = c[1]
        if min_y == None:
            min_y = y
        if y < min_y:
            min_y = y

    for c in circles:
        x = c[0]
        y = c[1]
        r = c[2]
        if abs(y-min_y) < HORIZONTAL_MAX_DIFF:
            valid_circles.append((x, y, r))
        else:
            print(f"DEBUG: Circle at ({x}, {y}) filtered out due to Y-axis deviation (min_y={min_y})")

    print("Found #%i circles after filtering:" % len(valid_circles))
    return valid_circles

def find_needle(image, cx, cy, radius):
    """
    Finds the needle by scanning radial lines and finding the one with the most dark pixels.
    """
    # Use a slightly smaller radius to stay within the dial
    scan_radius = radius * 0.9
    # Start slightly away from the center to avoid the hub
    start_offset = radius * 0.15
    
    slices = 100 # Increased resolution (3.6 degrees per slice)
    factor = 360 / slices
    
    best_value = 0
    max_darkness_score = -1
    needle_tip = (cx, cy)

    # Pre-calculate grayscale for the ROI to speed up
    x_min = max(0, int(cx - radius))
    y_min = max(0, int(cy - radius))
    x_max = min(image.shape[1], int(cx + radius))
    y_max = min(image.shape[0], int(cy + radius))
    roi = image[y_min:y_max, x_min:x_max]
    
    if roi.size == 0:
        return 0, (cx, cy)
        
    # Isolate RED pixels.
    # In warm lighting the BGR-difference method can incorrectly classify much of the
    # dial background as "red". HSV thresholding is typically more robust.
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Red hue wraps around, so we combine two ranges.
    # Tune these thresholds per camera/lighting.
    lower1 = np.array([0, 80, 60], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 80, 60], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Clean up small noise / fill gaps
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    if DEBUG_NEEDLE:
        cv2.imshow(f"needle_mask_{int(cx)}", mask)

    # Create a mask for the ring where the needle's pointer (triangle) should be
    # This ignores the hub (center circle) and the dial edge
    ring_mask = np.zeros_like(mask)
    h, w = mask.shape
    center_roi = (cx - x_min, cy - y_min)
    cv2.circle(ring_mask, (int(center_roi[0]), int(center_roi[1])), int(radius * 0.9), 255, -1)
    cv2.circle(ring_mask, (int(center_roi[0]), int(center_roi[1])), int(radius * 0.2), 0, -1)
    
    # Apply the ring mask to our needle mask
    target_mask = cv2.bitwise_and(mask, ring_mask)

    if DEBUG_NEEDLE:
        cv2.imshow(f"needle_target_{int(cx)}", target_mask)
    
    # Find the largest contour in the ring (which should be the needle)
    contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    needle_contour = None
    if contours:
        needle_contour = max(contours, key=cv2.contourArea)
        
    # Robust direction estimation:
    # Fit a line through ALL candidate needle pixels and choose the endpoint farthest
    # from the dial center as the needle tip.
    ys, xs = np.where(target_mask > 0)
    if xs.size > 80:
        pts = np.stack([xs, ys], axis=1).astype(np.float32)

        # Fit line (vx,vy) through points; (x0,y0) is a point on the line
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        # fitLine returns 1x1 arrays; extract scalars explicitly
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
        dx = tip_x - center_roi[0]
        dy = tip_y - center_roi[1]
        
        # Calculate angle (0 is Top)
        # atan2(y, x) gives angle from Right. 
        # We want 0 at Top, so we use atan2(dx, -dy) or adjust atan2(dy, dx)
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        
        # Normalize to 0-360 starting from Top (-90 deg in standard coord)
        # Standard: Right=0, Down=90, Left=180, Up=270
        # We want: Up=0, Right=90, Down=180, Left=270
        normalized_angle = (angle_deg + 90) % 360
        best_value = 10 * normalized_angle / 360
        
        # Calculate needle tip for visualization
        # Use the detected tip position (mapped back to image coords) for visualization
        needle_tip = (int(x_min + tip_x), int(y_min + tip_y))

        if DEBUG_NEEDLE:
            overlay = roi.copy()
            if needle_contour is not None:
                cv2.drawContours(overlay, [needle_contour], -1, (0, 255, 0), 2)
            cv2.line(overlay, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 255, 0), 2)
            cv2.circle(overlay, (int(tip_x), int(tip_y)), 6, (255, 0, 255), -1)
            cv2.circle(overlay, (int(center_roi[0]), int(center_roi[1])), 4, (0, 255, 255), -1)
            cv2.imshow(f"needle_overlay_{int(cx)}", overlay)

            # Save debug artifacts for offline inspection
            cv2.imwrite(f"_debug_needle_mask_{int(cx)}.png", mask)
            cv2.imwrite(f"_debug_needle_target_{int(cx)}.png", target_mask)
            cv2.imwrite(f"_debug_needle_overlay_{int(cx)}.png", overlay)
        
        print(f"DEBUG: Needle tip found at ({tip_x:.1f}, {tip_y:.1f}), angle: {normalized_angle:.1f} deg, value: {best_value:.2f}")
    else:
        print("DEBUG: No dark pixels found in needle ring.")
        best_value = 0
        needle_tip = (cx, cy)

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

def find_circles(frame):
    if frame is None:
        print(f"DEBUG: Error: Could not read image from {IMAGE_PATH}.")
        return

    print(f"DEBUG: find_circles started. Frame shape: {frame.shape}")

    if SAVE_IMAGE:
        filename = time.strftime("data/sample-%Y%m%d-%H%M.jpg")
        cv2.imwrite(filename, frame)

    # NOTE:
    # HoughCircles is very sensitive to contrast changes. Aggressively scaling the
    # grayscale image (e.g. alpha=2.5, beta=-300) can create many false edges and
    # cause HoughCircles to return hundreds of circles.
    #
    # We keep circle detection on a stable grayscale (+ optional blur), and do any
    # color emphasis (e.g., RED needle isolation) inside find_needle().
    print("DEBUG: Preparing image for circle detection...")
    hough_frame, scale = _resize_for_hough(frame)
    gray = cv2.cvtColor(hough_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.5)
    output = frame.copy()

    # Dynamic radius bounds based on image size (works across different resolutions)
    mind = min(gray.shape[0], gray.shape[1])
    min_radius = max(10, int(mind * RADIUS_MIN_FRAC))
    max_radius = max(min_radius + 1, int(mind * RADIUS_MAX_FRAC))

    print(
        "DEBUG: Searching for circles using HoughCircles... "
        f"(scale={scale:.3f}, minR={min_radius}, maxR={max_radius}, param2={HOUGH_PARAM2})"
    )
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=max(10, min_radius),
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    # TODO: move to config. In the provided images, all dials appear to be Clockwise (CW).
    readout_conventions = ["CW", "CW", "CW", "CW", "CW"]

    if circles is None:
        print("DEBUG: No circles found by HoughCircles.")
        return

    # Select the best dials even if Hough finds extras
    selected = select_best_circles(circles, gray, DIALS_COUNT)

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

    # DEBUG: show selected circles
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

    cv2.imshow("output", output)

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
    
    args = parser.parse_args()
    
    global SAVE_IMAGE
    SAVE_IMAGE = args.save
    
    if args.camera:
        if not CAMERA_AVAILABLE:
            print("ERROR: Camera mode requested but picamera2 is not available.")
            print("Install with: pip install picamera2")
            sys.exit(1)
        
        print("Starting camera mode...")
        
        if args.continuous:
            # Continuous capture mode
            print(f"Continuous capture mode - interval: {args.interval}s. Press Ctrl+C to stop.")
            with WaterMeterCamera() as camera:
                try:
                    while True:
                        print(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} - Capturing image...")
                        frame = camera.capture_image()
                        
                        # Convert from RGB to BGR for OpenCV
                        if frame.shape[2] == 3:
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        
                        find_circles(frame)
                        
                        print(f"Waiting {args.interval} seconds...")
                        time.sleep(args.interval)
                        cv2.destroyAllWindows()
                except KeyboardInterrupt:
                    print("\nStopping continuous capture...")
        else:
            # Single capture mode
            print("Single capture mode...")
            with WaterMeterCamera() as camera:
                frame = camera.capture_image()
                
                # Convert from RGB to BGR for OpenCV
                if frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                find_circles(frame)
                
                print("DEBUG: Processing complete. Press any key to exit...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    else:
        # File mode (original behavior)
        image_path = args.file
        print(f"Starting script using image: {image_path}")
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"ERROR: Could not read image from {image_path}")
            sys.exit(1)
        
        find_circles(frame)
        
        print("DEBUG: Processing complete. Press any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
