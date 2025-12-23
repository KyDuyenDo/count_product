
import cv2
import numpy as np
import time
import threading
import queue
from collections import deque
from enum import Enum, auto
from ultralytics import YOLO
from pyzbar.pyzbar import decode
import pytesseract
import re

# New Imports
from config import Config
from data_provider import DataProvider

# =============================================================================
# STATES
# =============================================================================

class State(Enum):
    IDLE = auto()
    ENTERING = auto()
    SCANNING = auto() 
    VALIDATING = auto() # New State: Waiting for DB
    VALIDATED = auto()  # New State: DB Success, locked
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
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, Config.AUTOFOCUS_VAL)
        self.cap.set(cv2.CAP_PROP_FOCUS, Config.FOCUS_VAL)
        
        # Tools
        self.model = None 
        self.data_provider = DataProvider()
        
        # OCR
        print(">>> Using Tesseract OCR (PyTesseract)")
        self.use_ocr = True

        # State Machine
        self.state = State.IDLE
        self.total_count = 0
        
        # Tracking Data
        self.current_barcode = None
        self.current_po = None
        self.last_barcode_rect = None
        self.last_po_rect = None
        
        # Validation Data
        self.validation_result = None # Stores DB result
        self.failed_pos = set()       # Anti-Spam: Set of POs that failed recently
        self.last_api_time = 0        # Anti-Spam: Timestamp of last API call
        
        # Counters
        self.state_frame_counter = 0 
        self.debounce_counter = 0    
        self.barcode_missing_frames = 0
        self.frame_count_total = 0 
        
        # Motion Analysis
        self.prev_gray = None
        self.avg_vx = 0.0
        self.avg_vy = 0.0
        self.last_flow = None
        
        # Threading
        self.validation_queue = queue.Queue()
        self.is_validating_thread = False

        # Performance Timing
        self.scan_start_time = None
        self.processing_time = None

        # Smoothing Checkpoint
        self.rect_history = deque(maxlen=5)

    def get_flow_vectors(self, gray):
        # 1. Blur to reduce noise
        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.prev_gray is None:
            self.prev_gray = gray_blurred
            return 0.0, 0.0
            
        # 2. Downscale
        small_gray = cv2.resize(gray_blurred, None, fx=Config.FLOW_DOWNSCALE, fy=Config.FLOW_DOWNSCALE)
        small_prev = cv2.resize(self.prev_gray, None, fx=Config.FLOW_DOWNSCALE, fy=Config.FLOW_DOWNSCALE)
        
        # 3. Crop to ROI
        h, w = small_gray.shape
        y_min = int(h * Config.MOTION_ROI_Y_MIN)
        y_max = int(h * Config.MOTION_ROI_Y_MAX)
        if y_max <= y_min: y_min, y_max = 0, h
        
        roi_gray = small_gray[y_min:y_max, :]
        roi_prev = small_prev[y_min:y_max, :]
        
        # 4. Optical Flow
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours: return None
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 3000: return None
            
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy
        return None

    def preprocess_for_ocr(self, img):
        if img is None or img.size == 0: return None
        scale = 2.0
        upscaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        adjusted = cv2.convertScaleAbs(upscaled, alpha=1.1, beta=50)
        return adjusted

    def scan_barcode_and_po(self, frame):
        h_img, w_img = frame.shape[:2]
        decoded_objects = decode(frame)
        
        if decoded_objects:
            self.barcode_missing_frames = 0
            obj = decoded_objects[0]
            new_barcode = obj.data.decode("utf-8")
            
            # Reset on new barcode
            if new_barcode != self.current_barcode:
                self.current_barcode = new_barcode
                self.current_po = None 
                self.rect_history.clear()
                self.validation_result = None # Clear validation
            
            # Tracking Rect
            raw_x, raw_y, raw_w, raw_h = obj.rect.left, obj.rect.top, obj.rect.width, obj.rect.height
            self.rect_history.append((raw_x, raw_y, raw_w, raw_h))
            avg_rect = np.mean(self.rect_history, axis=0).astype(int)
            x, y, w, h = avg_rect
            self.last_barcode_rect = (x, y, w, h)
            
            # PO Region 
            gap = 12 
            po_x, po_y = x, y + h + gap
            po_w, po_h = w, h // 2
            
            if po_x >= 0 and po_y >= 0 and (po_x + po_w) <= w_img and (po_y + po_h) <= h_img:
                self.last_po_rect = (po_x, po_y, po_w, po_h)
                
                # Run OCR if PO not found yet
                if self.current_po is None:
                    po_crop = frame[int(po_y):int(po_y+po_h), int(po_x):int(po_x+po_w)]
                    if po_crop.size > 0:
                        po_processed = self.preprocess_for_ocr(po_crop)
                        if po_processed is not None and self.use_ocr:
                            full_text = pytesseract.image_to_string(po_processed).strip()
                            candidates = re.findall(r'\d{5,}', full_text)
                            if candidates:
                                best_cand = max(candidates, key=len)
                                self.current_po = best_cand
                                print(f">>> FOUND PO: {self.current_po}")
        else:
             self.barcode_missing_frames += 1
             if self.barcode_missing_frames > Config.BARCODE_PERSISTENCE:
                 if self.current_barcode is not None:
                     self.current_barcode = None 
                     self.current_po = None
                     self.validation_result = None
                     self.last_barcode_rect = None
                     self.rect_history.clear()

    def trigger_validation(self):
        """
        Spawns a background thread to validate the current Barcode/PO pair.
        Implements Anti-Spam filters.
        """
        # Filter 1: Temporal Throttle
        now = time.time()
        if (now - self.last_api_time) < Config.API_COOLDOWN_SECONDS:
            print("[Anti-Spam] Throttled (Too fast)")
            return

        # Filter 2: Unique Sequence (Repeated Failures)
        if self.current_po in self.failed_pos:
            print(f"[Anti-Spam] Message Suppressed: PO {self.current_po} already failed recently.")
            return

        # Start Thread
        self.is_validating_thread = True
        self.state = State.VALIDATING
        self.last_api_time = now
        
        # Defines the work for the thread
        def validation_worker(po, bc):
            try:
                result = self.data_provider.validate_po_barcode(po, bc)
                self.validation_queue.put(result)
            except Exception as e:
                print(f"Thread Error: {e}")
                self.validation_queue.put(None)

        t = threading.Thread(target=validation_worker, args=(self.current_po, self.current_barcode))
        t.daemon = True
        t.start()

    def update_logic(self, frame):
        # 1. Motion Analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vx, vy = self.get_flow_vectors(gray)
        self.avg_vx = vx * 0.7 + self.avg_vx * 0.3 
        self.avg_vy = vy * 0.7 + self.avg_vy * 0.3
        mag = np.sqrt(self.avg_vx**2 + self.avg_vy**2)
        
        # 2. Physical Reset Trigger (Anti-Spam Reset)
        # If significant motion detected, user might have adjusted the box.
        if mag > 2.0: # Threshold for manual adjustment
            if self.failed_pos:
                print("[Anti-Spam] Physical Movement Detected - Clearing Failure Cache")
                self.failed_pos.clear()

        # 3. State Machine
        
        if self.state == State.IDLE:
            if abs(self.avg_vx) > Config.THRESH_ENTRY_X:
                self.debounce_counter += 1
            else:
                self.debounce_counter = 0
            if self.debounce_counter > Config.DEBOUNCE_FRAMES:
                self.state = State.ENTERING
                self.state_frame_counter = 0
                self.debounce_counter = 0
                self.current_barcode = None 
                self.current_po = None
                self.validation_result = None
                
        elif self.state == State.ENTERING:
            if mag < Config.THRESH_STABLE:
                self.state_frame_counter += 1
            else:
                self.state_frame_counter = 0
            if self.state_frame_counter >= Config.STABILITY_FRAMES:
                centroid = self.get_object_centroid(frame)
                if centroid:
                    cx, _ = centroid
                    w = Config.FRAME_WIDTH
                    if (w * Config.WORK_ZONE_X_MIN) < cx < (w * Config.WORK_ZONE_X_MAX):
                         self.state = State.SCANNING
                         self.state_frame_counter = 0
                         self.scan_start_time = time.time() # Start Timer

        elif self.state == State.SCANNING:
            if not self.get_object_centroid(frame):
                 # Emergency exit (lifted) - Clear ghosts
                 self.last_barcode_rect = None
                 self.rect_history.clear()
                 self.state = State.COUNTING 
                 return

            if self.frame_count_total % Config.OCR_THROTTLE_FRAMES == 0:
                 self.scan_barcode_and_po(frame)
            self.frame_count_total += 1
            
            # Trigger Validation logic
            if self.current_barcode and self.current_po:
                # If we have both, we attempt to validate
                 self.trigger_validation()
            
            # Transition to EXITING?
            is_moving_vert = self.avg_vy < Config.THRESH_EXIT_Y
            if is_moving_vert:
                self.state = State.EXITING
                self.state_frame_counter = 0

        elif self.state == State.VALIDATING:
            # Check for thread result non-blocking
            try:
                result = self.validation_queue.get_nowait()
                self.is_validating_thread = False
                
                if result and result.get("valid"):
                    self.validation_result = result
                    self.state = State.VALIDATED
                    
                    # COUNT IMMEDIATELY ON VALIDATION
                    self.total_count += 1
                    
                    # Calculate Logic Time
                    if self.scan_start_time:
                        self.processing_time = time.time() - self.scan_start_time
                        print(f">>> Processed in {self.processing_time:.2f}s")
                        
                    print(f">>> VALIDATION SUCCESS: {result}")
                    print(f">>> COUNTED! Total: {self.total_count}")
                    
                else:
                    print(">>> VALIDATION FAILED")
                    self.failed_pos.add(self.current_po) # Block this PO until motion
                    self.state = State.SCANNING # Go back to scanning to retry or adjust
                    self.current_po = None 
                    
            except queue.Empty:
                pass # Wait...

        elif self.state == State.VALIDATED:
            # Locked state - show green frame, success info.
            # Wait for exit
            is_moving_vert = self.avg_vy < Config.THRESH_EXIT_Y
            if is_moving_vert:
                self.state = State.EXITING
                self.state_frame_counter = 0

        elif self.state == State.EXITING:
            if self.last_barcode_rect:
                x, y, w, h = self.last_barcode_rect
                self.last_barcode_rect = (int(x+self.avg_vx), int(y+self.avg_vy), w, h)
            
            if mag < Config.THRESH_STABLE:
                self.state_frame_counter += 1
            else:
                self.state_frame_counter = 0
            if self.state_frame_counter >= Config.RESET_FRAMES:
                self.state = State.COUNTING
        
        elif self.state == State.COUNTING:
            # Just reset state, no counting here
            self.current_barcode = None
            self.current_po = None
            self.validation_result = None
            self.failed_pos.clear()
            
            # Clear Ghost Box
            self.last_barcode_rect = None
            self.rect_history.clear()
            
            # Clear Timers
            self.scan_start_time = None
            self.processing_time = None
            
            self.state = State.IDLE

    def draw_hud(self, frame):
        h, w = frame.shape[:2]
        
        # Zones
        min_x = int(w * Config.WORK_ZONE_X_MIN)
        max_x = int(w * Config.WORK_ZONE_X_MAX)
        cv2.line(frame, (min_x, 0), (min_x, h), (255, 255, 0), 2)
        cv2.line(frame, (max_x, 0), (max_x, h), (255, 255, 0), 2)
        
        # State
        state_color = (0, 255, 0)
        if self.state == State.VALIDATING: state_color = (0, 255, 255) # Yellow
        if self.state == State.VALIDATED: state_color = (0, 200, 0)    # Green
        cv2.putText(frame, f"STATE: {self.state.name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, state_color, 2)
        
        # Object Info
        if self.last_barcode_rect:
             lx, ly, lw, lh = self.last_barcode_rect
             
             # Color Logic
             # Green Frame: Data Validated
             # Yellow/Orange Frame: PO Not Found / Invalid
             color = (0, 255, 255) # Default Yellow
             
             if self.state == State.VALIDATED:
                 color = (0, 255, 0)
             elif self.state == State.SCANNING and self.current_po in self.failed_pos:
                 color = (0, 0, 255) # Red/Orange for Invalid
                 
             cv2.rectangle(frame, (lx, ly), (lx+lw, ly+lh), color, 3)
             
             msg = "Scanning..."
             if self.current_barcode: msg = f"BC: {self.current_barcode}"
             cv2.putText(frame, msg, (lx, ly - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
             
             if self.current_po:
                 cv2.putText(frame, f"PO: {self.current_po}", (lx, ly + lh + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
                 
             # Success Details
             if self.validation_result:
                 info = f"{self.validation_result['article']} | Size: {self.validation_result['size']}"
                 cv2.putText(frame, info, (lx, ly + lh + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                 
                 if self.processing_time:
                     time_str = f"Time: {self.processing_time:.2f}s"
                     cv2.putText(frame, time_str, (lx, ly + lh + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
             
             # Failure Details
             if self.state == State.SCANNING and self.current_po in self.failed_pos:
                 cv2.putText(frame, "Invalid Code", (lx, ly + lh + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Motion Vector
        cx, cy = w // 2, h // 2
        end_x = int(cx + self.avg_vx * 20)
        end_y = int(cy + self.avg_vy * 20)
        cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), (0, 0, 255), 3)

        # Total Count
        cv2.putText(frame, f"COUNT: {self.total_count}", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    def run(self):
        print(">>> Starting Box Scanning System...")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(1)
                continue
                
            self.update_logic(frame)
            self.draw_hud(frame)
            cv2.imshow("Box Scanner", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = BoxFlowAnalyzer()
    app.run()
