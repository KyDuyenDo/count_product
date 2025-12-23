
import cv2
import numpy as np
import time
from collections import deque
from enum import Enum, auto
from ultralytics import YOLO
from pyzbar.pyzbar import decode
import pytesseract
import re

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # Camera
    CAMERA_ID = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FOCUS_VAL = 30    # 0-255, 30 is usually good for macro/nearby
    AUTOFOCUS_VAL = 0 # 0=Off, 1=On
    
    # Optical Flow (Farneback)
    FLOW_DOWNSCALE = 0.5  # Calculate flow on half resolution for speed
    FLOW_PYR_SCALE = 0.5
    FLOW_LEVELS = 3
    FLOW_WINSZ = 15
    FLOW_ITERATIONS = 3
    FLOW_POLY_N = 5
    FLOW_POLY_SIGMA = 1.2
    
    # Motion Thresholds (Pixels per frame, adjusted for downscale)
    THRESH_ENTRY_X = 0.5      # Significant horizontal motion (Lowered for sensitivity)
    THRESH_EXIT_Y = -3.0      # Significant UPWARD vertical motion (negative Y)
    THRESH_STABLE = 1.0       # Low motion threshold
    
    # Motion Processing
    MOTION_ROI_Y_MIN = 0.2    # Ignore top 20% (Background)
    MOTION_ROI_Y_MAX = 0.8    # Ignore bottom 20%
    DEBOUNCE_FRAMES = 3       # Frames of consistent motion to trigger ENTERING
    
    # Logic
    STABILITY_FRAMES = 3      # Frames of low motion to trigger SCANNING (Faster entry)
    RESET_FRAMES = 10         # Frames of low motion to trigger COUNTING (after exit)
    BARCODE_PERSISTENCE = 10  # Frames to keep barcode after detection loss
    OCR_THROTTLE_FRAMES = 1   # Run OCR every frame in SCANNING state (Max reliability)
    
    # Work Zone (Percent of Frame Width)
    WORK_ZONE_X_MIN = 0.25    # Left boundary (25%)
    WORK_ZONE_X_MAX = 0.75    # Right boundary (75%)
    
    # YOLO
    YOLO_MODEL = "YOLOV8s_Barcode_Detection.pt" # Or "yolov8n.pt" if custom not found
    CONF_THRESH = 0.4

# =============================================================================
# STATES
# =============================================================================

class State(Enum):
    IDLE = auto()
    ENTERING = auto()
    SCANNING = auto() # Equivalent to STABLE but actively scanning
    EXITING = auto()
    COUNTING = auto()

# =============================================================================
# CORE SYSTEM
# =============================================================================

class BoxFlowAnalyzer:
    def __init__(self):
        # Hardware
        self.cap = cv2.VideoCapture(Config.CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        # Focus Settings
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, Config.AUTOFOCUS_VAL)
        self.cap.set(cv2.CAP_PROP_FOCUS, Config.FOCUS_VAL)
        
        # AI Model - REMOVED YOLO as per request
        self.model = None 
            
        # OCR
        print(">>> Using Tesseract OCR (PyTesseract)")
        # No init needed for pytesseract, assuming binary in PATH
        self.use_ocr = True

        # State Machine
        self.state = State.IDLE
        self.total_count = 0
        self.current_barcode = None
        self.current_po = None
        self.last_barcode_rect = None # For visualization
        self.last_po_rect = None      # For visualization
        self.state_frame_counter = 0 # To track duration in a state (e.g. for stability)
        self.debounce_counter = 0    # For debouncing transitions
        self.barcode_missing_frames = 0 # Track frames since last barcode detection
        self.frame_count_total = 0 # Global frame counter for throttling logic
        
        # Motion Analysis
        self.prev_gray = None
        self.avg_vx = 0.0
        self.avg_vy = 0.0
        self.last_flow = None
        
        # Smoothing Checkpoint
        self.rect_history = deque(maxlen=5)

    def get_flow_vectors(self, gray):
        # 1. Blur to reduce noise (sensor noise, lighting flicker)
        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.prev_gray is None:
            self.prev_gray = gray_blurred
            return 0.0, 0.0
            
        # 2. Downscale for performance
        small_gray = cv2.resize(gray_blurred, None, fx=Config.FLOW_DOWNSCALE, fy=Config.FLOW_DOWNSCALE)
        small_prev = cv2.resize(self.prev_gray, None, fx=Config.FLOW_DOWNSCALE, fy=Config.FLOW_DOWNSCALE)
        
        # 3. Crop to ROI (Ignore background top/bottom)
        h, w = small_gray.shape
        y_min = int(h * Config.MOTION_ROI_Y_MIN)
        y_max = int(h * Config.MOTION_ROI_Y_MAX)
        
        # Ensure ROI is valid
        if y_max <= y_min:
             y_min, y_max = 0, h
        
        roi_gray = small_gray[y_min:y_max, :]
        roi_prev = small_prev[y_min:y_max, :]
        
        # 4. Optical Flow on ROI only
        flow = cv2.calcOpticalFlowFarneback(
            roi_prev, roi_gray, None,
            Config.FLOW_PYR_SCALE, Config.FLOW_LEVELS, Config.FLOW_WINSZ,
            Config.FLOW_ITERATIONS, Config.FLOW_POLY_N, Config.FLOW_POLY_SIGMA,
            0
        )
        
        # 5. Average Motion
        avg_vx = np.mean(flow[:, :, 0])
        avg_vy = np.mean(flow[:, :, 1])
        
        self.prev_gray = gray_blurred
        self.last_flow = flow 
        return avg_vx, avg_vy

    def get_object_centroid(self, frame):
        """
        Finds the center (x, y) of the largest object in the frame.
        Assumes dark object on light background or vice-versa.
        Returns (cx, cy) or None.
        """
        # Convert to grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Simple thresholding (Adjust 100 based on lighting)
        _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Get largest contour
        c = max(contours, key=cv2.contourArea)
        
        # Ignore small noise
        if cv2.contourArea(c) < 3000:
            return None
            
        # Compute Center
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy
        
        return None

    def visualize_flow(self):
        # Adjusted visualization to show ROI flow
        if not hasattr(self, 'last_flow') or self.last_flow is None:
            return np.zeros((Config.FRAME_HEIGHT, Config.FRAME_WIDTH, 3), dtype=np.uint8)
            
        flow = self.last_flow
        h, w = flow.shape[:2]
        
        # Create HSV image
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 1] = 255
        
        # Convert to polar
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Map to HSV
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        # Convert to BGR
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Resize to matching display size (Just stretch it to fill frame for debug)
        return cv2.resize(bgr, (Config.FRAME_WIDTH, Config.FRAME_HEIGHT))
        
        # Resize to matching display size
        return cv2.resize(bgr, (Config.FRAME_WIDTH, Config.FRAME_HEIGHT))

    def preprocess_for_ocr(self, img):
        """
        Advanced Preprocessing (Refactor v4):
        1. 2x Upscale
        2. Grayscale
        3. Bilateral Filter (Noise Removal, Edge Preservation)
        4. Adaptive Thresholding (Handle varying lighting)
        """
        if img is None or img.size == 0:
            return None
            
        # 1. Upscale
        scale = 2.0
        upscaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        
        # 2. Grayscale
        if len(upscaled.shape) == 3:
            gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        else:
            gray = upscaled
            
        # 3. Bilateral Filter
        # d=9: Diameter of pixel neighborhood
        # sigmaColor=75: Filter sigma in color space
        # sigmaSpace=75: Filter sigma in coordinate space
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
    # 4. Adaptive Thresholding
        # ADAPTIVE_THRESH_GAUSSIAN_C: Threshold value is weighted sum of neighbourhood values
        # THRESH_BINARY: Maxval is 255
        # Block Size: 11
        # C: 2 (Constant subtracted from mean)
        binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
                                     
        # 5. Morphological Closing (Connect broken characters)
        # Useful for dot-matrix or noisy prints
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                                     
        # Convert to BGR for display
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def scan_barcode_and_po(self, frame):
        found = False
        h_img, w_img = frame.shape[:2]
        
        # 1. Barcode Detection & Reading (Pyzbar)
        decoded_objects = decode(frame)
        
        if decoded_objects:
            # Reset missing counter
            self.barcode_missing_frames = 0
            
            # Take the first detected barcode
            obj = decoded_objects[0]
            new_barcode = obj.data.decode("utf-8")
            
            # CHECK: Has barcode changed?
            if new_barcode != self.current_barcode:
                self.current_barcode = new_barcode
                self.current_po = None # Reset PO for new box
                self.rect_history.clear() # Reset smoothing history
            
            # Localization from Pyzbar
            raw_x, raw_y, raw_w, raw_h = obj.rect.left, obj.rect.top, obj.rect.width, obj.rect.height
            
            # SMOOTHING: Moving Average
            self.rect_history.append((raw_x, raw_y, raw_w, raw_h))
            avg_rect = np.mean(self.rect_history, axis=0).astype(int)
            x, y, w, h = avg_rect
            
            self.last_barcode_rect = (x, y, w, h)
            found = True
            
            # OPTIMIZATION: If we already found the PO for this barcode, SKIP OCR!
            if self.current_po is not None:
                # We already have the data, just draw the rects (handled in draw_hud via last_*)
                # We do need to update the PO rect location relative to the moving barcode though (logic below)
                pass 
            
            # 2. PO Region Extraction & OCR (Only if PO missing or need to update rect)
            # We always calculate rect to visualize it correctly even if OCR skipped
            
            # Rule: 10-15px below Barcode box
            gap = 12 
            po_x = x
            po_y = y + h + gap
            po_w = w
            po_h = h // 2 # Half height of barcode
            
            # Boundary check
            if po_x >= 0 and po_y >= 0 and (po_x + po_w) <= w_img and (po_y + po_h) <= h_img:
                # Update visualization box every frame for aiming, even if we stop OCR
                self.last_po_rect = (po_x, po_y, po_w, po_h)
                
                # OPTIMIZATION: Stop all processing if PO is already found
                if self.current_po is None:
                    # Extract crop
                    po_crop = frame[int(po_y):int(po_y+po_h), int(po_x):int(po_x+po_w)]
                    
                    # Debug: Show the crop in a window for real-time feedback
                    if po_crop.size > 0:
                        # Apply Robust Preprocessing
                        po_processed = self.preprocess_for_ocr(po_crop)
                        
                        if po_processed is not None:
                            # Visual Confirmation: Label the image
                            cv2.putText(po_processed, "OCR INPUT", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            
                            cv2.imshow("Debug PO Input", po_processed)
                            
                            # Smart Throttling: Scan faster if SCANNING (Stable)
                            interval = 5 if self.state == State.SCANNING else 15

                            # Run OCR only if match throttle
                            self.frame_count_total += 1
                            if self.use_ocr and (self.frame_count_total % interval == 0):
                                try:
                                    # OCR the processed image using Tesseract 
                                    full_text = pytesseract.image_to_string(po_processed)
                                    full_text = full_text.strip()
                                    print(f"OCR Raw: {full_text}")
                                    
                                    # Regex: Relaxed - find any sequence of 5 or more digits
                                    # User specified no strict length limit and no required prefix
                                    candidates = re.findall(r'\d{5,}', full_text)
                                    
                                    if candidates:
                                        # Heuristic: The longest number sequence is likely the PO
                                        # (Avoids picking up small stray numbers)
                                        best_cand = max(candidates, key=len)
                                        
                                        self.current_po = best_cand
                                        print(f">>> FOUND PO (Relaxed): {self.current_po}")
                                            
                                except Exception as e:
                                    # print(f"OCR Error: {e}")
                                    pass # Silence errors
                else:
                    pass # PO already found, skip logic
            else:
                 pass # Silence PO bounds error

                

        else:
             # No barcode found
             self.barcode_missing_frames += 1
             if self.barcode_missing_frames > Config.BARCODE_PERSISTENCE:
                 if self.current_barcode is not None:
                     self.current_barcode = None # Clear if missing for too long
                     self.current_po = None # Clear PO too
            
        return found

    def update_logic(self, frame):
        # 1. Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Motion Analysis
        vx, vy = self.get_flow_vectors(gray)
        self.avg_vx = vx * 0.7 + self.avg_vx * 0.3 # Moving average for smoothness
        self.avg_vy = vy * 0.7 + self.avg_vy * 0.3
        
        # Magnitude
        mag = np.sqrt(self.avg_vx**2 + self.avg_vy**2)
        
        # 3. FSM Transitions (Refactor v6 - Work Zone)
        
        if self.state == State.IDLE:
            # Transition: Significant horizontal motion with DEBOUNCING
            if abs(self.avg_vx) > Config.THRESH_ENTRY_X:
                self.debounce_counter += 1
            else:
                self.debounce_counter = 0
                
            if self.debounce_counter > Config.DEBOUNCE_FRAMES:
                self.state = State.ENTERING
                self.state_frame_counter = 0
                self.debounce_counter = 0
                # Reset tracking data
                self.current_barcode = None 
                self.current_po = None
                self.last_barcode_rect = None
                self.last_po_rect = None
                self.barcode_missing_frames = 0
                
        elif self.state == State.ENTERING:
            # Requirement: Box must stabilize IN THE WORK ZONE to start scanning
            if mag < Config.THRESH_STABLE:
                self.state_frame_counter += 1
            else:
                self.state_frame_counter = 0
                
            # Check stability duration
            if self.state_frame_counter >= Config.STABILITY_FRAMES:
                # Check WORK ZONE Position
                centroid = self.get_object_centroid(frame)
                if centroid:
                    cx, cy = centroid
                    frame_w = Config.FRAME_WIDTH
                    
                    # Verify X center is in [25% - 75%] zone
                    valid_zone = (frame_w * Config.WORK_ZONE_X_MIN) < cx < (frame_w * Config.WORK_ZONE_X_MAX)
                    
                    if valid_zone:
                         self.state = State.SCANNING
                         self.state_frame_counter = 0
                    else:
                         # Still stable, but not centered. Wait.
                         pass
                else:
                    # No object found? Maybe unstable lighting. Wait.
                    pass
                
        elif self.state == State.SCANNING:
            # Action: Run Barcode and PO Detection (Throttled)
            self.frame_count_total += 1
            if self.frame_count_total % Config.OCR_THROTTLE_FRAMES == 0:
                 self.scan_barcode_and_po(frame)
                    
            # GRACE PERIOD & PO EXTENSION: 
            # Force stay in scanning for at least 30 frames (approx 1 sec)
            # OR if we have a barcode but NO PO yet (give it more time)
            
            extend_scan = False
            if self.current_barcode and not self.current_po:
                # If we have barcode but no PO, give it extra time (up to 60 frames)
                if self.state_frame_counter < 60:
                    extend_scan = True
            
            self.state_frame_counter += 1
            
            if self.state_frame_counter < 10 or extend_scan:
                pass # Force stay
            else:
                # Transition: Uppward vertical motion -> EXITING
                # Horizontal motion is NOW IGNORED (Refactor v10) - stay scanning!
                is_moving_vert = self.avg_vy < Config.THRESH_EXIT_Y
                # is_moving_horz = abs(self.avg_vx) > Config.THRESH_ENTRY_X
                
                if is_moving_vert:
                    self.state = State.EXITING
                    self.state_frame_counter = 0
                
        elif self.state == State.EXITING:
            # Animation: Move the visualization boxes with the flow
            if self.last_barcode_rect:
                x, y, w, h = self.last_barcode_rect
                x = int(x + self.avg_vx)
                y = int(y + self.avg_vy)
                self.last_barcode_rect = (x, y, w, h)
                
            if self.last_po_rect:
                px, py, pw, ph = self.last_po_rect
                px = int(px + self.avg_vx)
                py = int(py + self.avg_vy)
                self.last_po_rect = (px, py, pw, ph)

            # Transition: Motion stops (scene cleared) -> COUNTING
            if mag < Config.THRESH_STABLE:
                self.state_frame_counter += 1
            else:
                self.state_frame_counter = 0
                
            if self.state_frame_counter >= Config.RESET_FRAMES:
                self.state = State.COUNTING
        
        elif self.state == State.COUNTING:
            # Condition: Only count if we actually found a barcode
            if self.current_barcode:
                self.total_count += 1
                print(f">>> Box Counted! Total: {self.total_count} | BC: {self.current_barcode} | PO: {self.current_po}")
                
            else:
                 print(">>> Movement finished but no barcode found. Ignoring count (Shake/False Trigger).")
            
            # CRITICAL: Clear all data to prevent ghosting
            self.current_barcode = None
            self.current_po = None
            self.last_barcode_rect = None
            self.last_po_rect = None
            
            self.state = State.IDLE

    def draw_hud(self, frame):
        h, w = frame.shape[:2]
        
        # 0. Draw Work Zone (Vertical Lines)
        min_x = int(w * Config.WORK_ZONE_X_MIN)
        max_x = int(w * Config.WORK_ZONE_X_MAX)
        
        cv2.line(frame, (min_x, 0), (min_x, h), (255, 255, 0), 2) # Cyan Line Left
        cv2.line(frame, (max_x, 0), (max_x, h), (255, 255, 0), 2) # Cyan Line Right
        
        cv2.putText(frame, "WORK ZONE", (min_x + 10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        # 1. State Visualization (Top Left)
        cv2.putText(frame, f"STATE: {self.state.name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 2. Motion Vectors
        # Draw an arrow from center
        cx, cy = w // 2, h // 2
        # Scale vectors for visibility
        end_x = int(cx + self.avg_vx * 20)
        end_y = int(cy + self.avg_vy * 20)
        cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), (0, 0, 255), 3)
        
        cv2.putText(frame, f"Vx: {self.avg_vx:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, f"Vy: {self.avg_vy:.1f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # 3. Barcode & PO Visualization (Attached to Object)
        if self.last_barcode_rect:
             lx, ly, lw, lh = self.last_barcode_rect
             
             # Color based on detection status
             color = (0, 255, 255) # Yellow (Barcode only)
             if self.current_po:
                 color = (0, 255, 0) # Green (Complete)
             elif self.state == State.EXITING:
                 color = (200, 200, 200) # Grey/Fading (Exiting)
                 
             cv2.rectangle(frame, (lx, ly), (lx+lw, ly+lh), color, 2)
             
             # Draw Barcode Text ABOVE box
             if self.current_barcode:
                 label = f"BC: {self.current_barcode}"
                 cv2.putText(frame, label, (lx, ly - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
             else:
                 cv2.putText(frame, "Scanning...", (lx, ly - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

             # Draw PO Text BELOW box
             if self.current_po:
                 label_po = f"PO: {self.current_po}"
                 cv2.putText(frame, label_po, (lx, ly + lh + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
                 
        if self.last_po_rect:
             px, py, pw, ph = self.last_po_rect
             # Only draw PO rect if debugging or not found yet? 
             # Let's draw it faint blue to show where we looked
             cv2.rectangle(frame, (px, py), (px+pw, py+ph), (255, 0, 0), 1)
             
        # 4. Total Count
        cv2.putText(frame, f"COUNT: {self.total_count}", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    def run(self):
        print(">>> Starting BoxFlowAnalyzer...")
        print("Press 'q' to quit.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Camera frame lost. Attempting to reconnect...")
                self.cap.release()
                time.sleep(1)
                try:
                    self.cap = cv2.VideoCapture(Config.CAMERA_ID)
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
                    self.cap.set(cv2.CAP_PROP_AUTOFOCUS, Config.AUTOFOCUS_VAL)
                    self.cap.set(cv2.CAP_PROP_FOCUS, Config.FOCUS_VAL)
                    if not self.cap.isOpened():
                         raise Exception("Could not open camera")
                    print(">>> Camera Reconnected!")
                except Exception as e:
                    print(f"Reconnect failed: {e}")
                    time.sleep(2)
                continue
                
            self.update_logic(frame)
            self.draw_hud(frame)
            
            # Visualize Flow
            flow_vis = self.visualize_flow()
            cv2.imshow("Optical Flow", flow_vis)
            
            cv2.imshow("BoxFlowAnalyzer", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = BoxFlowAnalyzer()
    app.run()
