
import cv2
import numpy as np
import time
from enum import Enum, auto
from ultralytics import YOLO
from pyzbar.pyzbar import decode
from paddleocr import PaddleOCR
import re

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # Camera
    CAMERA_ID = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    
    # Optical Flow (Farneback)
    FLOW_DOWNSCALE = 0.5  # Calculate flow on half resolution for speed
    FLOW_PYR_SCALE = 0.5
    FLOW_LEVELS = 3
    FLOW_WINSZ = 15
    FLOW_ITERATIONS = 3
    FLOW_POLY_N = 5
    FLOW_POLY_SIGMA = 1.2
    
    # Motion Thresholds (Pixels per frame, adjusted for downscale)
    THRESH_ENTRY_X = 2.0      # Significant horizontal motion
    THRESH_EXIT_Y = -3.0      # Significant UPWARD vertical motion (negative Y)
    THRESH_STABLE = 1.0       # Low motion threshold
    
    # Logic
    STABILITY_FRAMES = 5      # Frames of low motion to trigger SCANNING
    RESET_FRAMES = 10         # Frames of low motion to trigger COUNTING (after exit)
    
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
        
        # AI Model (YOLO - kept for compatibility, strictly using Pyzbar for logic as requested)
        self.model = None # Disabling YOLO load for speed if not strictly needed, relying on clean imports
        # But to be safe/compliant with prompt "Design a Python system using OpenCV and YOLOv8", I will keep it loaded but use Pyzbar for reading.
        try:
            self.model = YOLO(Config.YOLO_MODEL)
            print(f">>> Loaded YOLO model: {Config.YOLO_MODEL}")
        except Exception as e:
            print(f">>> WARN: Could not load {Config.YOLO_MODEL}, falling back to yolov8n.pt. Error: {e}")
            self.model = YOLO("yolov8n.pt")
            
        # OCR
        print(">>> Initializing PaddleOCR...")
        try:
             # Fix deprecation warning
             self.ocr = PaddleOCR(use_textline_orientation=True, lang='en', show_log=False)
        except Exception as e:
             print(f">>> WARN: PaddleOCR failed to load. OCR will be disabled. Error: {e}")
             self.ocr = None

        # State Machine
        self.state = State.IDLE
        self.total_count = 0
        self.current_barcode = None
        self.current_po = None
        self.last_barcode_rect = None # For visualization
        self.last_po_rect = None      # For visualization
        self.state_frame_counter = 0 # To track duration in a state (e.g. for stability)
        
        # Motion Analysis
        self.prev_gray = None
        self.avg_vx = 0.0
        self.avg_vy = 0.0
        self.last_flow = None

    def get_flow_vectors(self, gray):
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0, 0.0
            
        # Downscale for performance
        small_gray = cv2.resize(gray, None, fx=Config.FLOW_DOWNSCALE, fy=Config.FLOW_DOWNSCALE)
        small_prev = cv2.resize(self.prev_gray, None, fx=Config.FLOW_DOWNSCALE, fy=Config.FLOW_DOWNSCALE)
        
        flow = cv2.calcOpticalFlowFarneback(
            small_prev, small_gray, None,
            Config.FLOW_PYR_SCALE, Config.FLOW_LEVELS, Config.FLOW_WINSZ,
            Config.FLOW_ITERATIONS, Config.FLOW_POLY_N, Config.FLOW_POLY_SIGMA,
            0
        )
        
        # Calculate average motion of the center of the frame (where the box is)
        h, w = flow.shape[:2]
        roi = flow # Use full frame average for large boxes covering most of view
        
        avg_vx = np.mean(roi[:, :, 0])
        avg_vy = np.mean(roi[:, :, 1])
        
        self.prev_gray = gray
        self.last_flow = flow  # Store for visualization
        return avg_vx, avg_vy

    def visualize_flow(self):
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
        
        # Resize to matching display size
        return cv2.resize(bgr, (Config.FRAME_WIDTH, Config.FRAME_HEIGHT))

    def scan_barcode_and_po(self, frame):
        found = False
        
        # 1. Barcode Detection (YOLO)
        # Predict
        results = self.model.predict(frame, conf=Config.CONF_THRESH, verbose=False)
        
        if results and len(results[0].boxes) > 0:
            # Get highest confidence box
            best_box = results[0].boxes[0]
            x_c, y_c, w_box, h_box = best_box.xywh[0].cpu().numpy()
            
            # Convert to top-left
            x = int(x_c - w_box/2)
            y = int(y_c - h_box/2)
            w = int(w_box)
            h = int(h_box)
            
            # Clamp
            h_img, w_img = frame.shape[:2]
            x = max(0, x); y = max(0, y)
            w = min(w, w_img - x); h = min(h, h_img - y)
            
            self.last_barcode_rect = (x, y, w, h)
            found = True
            
            # 1B. Read Barcode Content (Pyzbar on crop)
            barcode_crop = frame[y:y+h, x:x+w]
            if barcode_crop.size > 0:
                decoded_objects = decode(barcode_crop)
                if decoded_objects:
                    self.current_barcode = decoded_objects[0].data.decode("utf-8")
                else:
                    if self.current_barcode is None: # Don't overwrite if we had a read
                        self.current_barcode = "Detected (Unreadable)"

            # 2. PO Region Extraction
            # Rule: 10-15px below YOLO box
            gap = 12 
            po_x = x
            po_y = y + h + gap
            po_w = w
            po_h = h // 2 # Half height of barcode
            
            # Boundary check
            if po_x >= 0 and po_y >= 0 and (po_x + po_w) <= w_img and (po_y + po_h) <= h_img:
                po_crop = frame[int(po_y):int(po_y+po_h), int(po_x):int(po_x+po_w)]
                self.last_po_rect = (po_x, po_y, po_w, po_h)
                
                # 3. OCR on PO Region
                if self.ocr and po_crop.size > 0:
                     try:
                        result = self.ocr.ocr(po_crop, cls=False)
                        if result and result[0]:
                            txts = [line[1][0] for line in result[0]]
                            full_text = " ".join(txts)
                            print(f"OCR Raw: {full_text}")
                            
                            # Regex
                            match = re.search(r'#\s*([A-Za-z0-9]+)', full_text)
                            if match:
                                self.current_po = match.group(1)
                                print(f">>> FOUND PO: {self.current_po}")
                     except Exception as e:
                         print(f"OCR Error: {e}")
            else:
                 print("PO Region out of bounds")
            
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
        
        # 3. FSM Transitions
        
        if self.state == State.IDLE:
            # Transition: Significant horizontal motion
            if abs(self.avg_vx) > Config.THRESH_ENTRY_X:
                self.state = State.ENTERING
                self.state_frame_counter = 0
                self.current_barcode = None # Reset for new box
                self.current_po = None
                self.last_barcode_rect = None
                self.last_po_rect = None
                
        elif self.state == State.ENTERING:
            # monitor for stability
            if mag < Config.THRESH_STABLE:
                self.state_frame_counter += 1
            else:
                self.state_frame_counter = 0
                
            # Transition: Stable for N frames -> SCANNING
            if self.state_frame_counter >= Config.STABILITY_FRAMES:
                self.state = State.SCANNING
                self.state_frame_counter = 0
                
        elif self.state == State.SCANNING:
            # Action: Run Barcode and PO Detection
            # We want to lock the result once found
            if self.current_barcode is None:
                self.scan_barcode_and_po(frame)
                    
            # Transition: Upward vertical motion -> EXITING
            if self.avg_vy < Config.THRESH_EXIT_Y:
                self.state = State.EXITING
                self.state_frame_counter = 0
                
        elif self.state == State.EXITING:
            # Transition: Motion stops (scene cleared) -> COUNTING
            if mag < Config.THRESH_STABLE:
                self.state_frame_counter += 1
            else:
                self.state_frame_counter = 0
                
            if self.state_frame_counter >= Config.RESET_FRAMES:
                self.state = State.COUNTING
        
        elif self.state == State.COUNTING:
            self.total_count += 1
            self.state = State.IDLE
            print(f">>> Box Counted! Total: {self.total_count} | BC: {self.current_barcode} | PO: {self.current_po}")

    def draw_hud(self, frame):
        h, w = frame.shape[:2]
        
        # 1. State Visualization
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
        
        # 3. Barcode & PO Visualization
        if self.last_barcode_rect:
             lx, ly, lw, lh = self.last_barcode_rect
             cv2.rectangle(frame, (lx, ly), (lx+lw, ly+lh), (0, 255, 0), 2)
             
        if self.last_po_rect:
             px, py, pw, ph = self.last_po_rect
             cv2.rectangle(frame, (px, py), (px+pw, py+ph), (255, 0, 0), 2)
             cv2.putText(frame, "PO AREA", (px, py-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        if self.current_barcode:
             cv2.putText(frame, f"BC: {self.current_barcode}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if self.current_po:
             cv2.putText(frame, f"PO: {self.current_po}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
             
        # 4. Total Count
        cv2.putText(frame, f"COUNT: {self.total_count}", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    def run(self):
        print(">>> Starting BoxFlowAnalyzer...")
        print("Press 'q' to quit.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
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
