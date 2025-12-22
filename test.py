import cv2
import numpy as np
from pyzbar.pyzbar import decode
from paddleocr import PaddleOCR
import re
import time
import math
from enum import Enum, auto
from collections import deque, Counter
import os

# Suppress PaddleOCR logging
os.environ['FLAGS_minloglevel'] = '2'

# =============================================================================
# V5 CONFIGURATION & CONSTANTS
# =============================================================================

ROI_EXIT_TOP_H = 0.15    
# ROI_ENTRY_WIDTH Removed - Entry is anywhere below Exit

# Tracking & Motion
STABILITY_BUFFER = 10
STABILITY_THRESH = 10.0
TRACKING_TIMEOUT_MS = 500
MIN_MOTION_AREA = 3000    

class BoxState(Enum):
    NONE = auto()
    WORKING = auto() # Merged ENTRY/WORKING
    EXIT = auto()
    COUNTED = auto()

class TrackingSource(Enum):
    NONE = auto()
    BARCODE = auto()
    MOTION = auto()
    COASTING = auto()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def rect_center(rect):
    return (rect[0] + rect[2]//2, rect[1] + rect[3]//2)

def rect_overlap(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    return (x1 < x2 + w2 and x1 + w1 > x2 and
            y1 < y2 + h2 and y1 + h1 > y2)

def get_largest_rect(rects):
    if not rects: return None
    return max(rects, key=lambda r: r[2] * r[3])

def detect_motion(frame, bg_subtractor, min_area=MIN_MOTION_AREA):
    mask = bg_subtractor.apply(frame, learningRate=0.005)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            rects.append(cv2.boundingRect(cnt))
    return rects, mask

def extract_po_number(img_array, ocr_engine):
    try:
        if img_array is None or img_array.size == 0: return None
        result = ocr_engine.ocr(img_array, cls=True)
        if not result or result[0] is None: return None
        
        found_text = ""
        for line in result[0]:
            found_text += " " + line[1][0]
            
        match = re.search(r'(?:#|PO:?|NO\.?)\s*(\d+)', found_text, re.IGNORECASE)
        if match: return match.group(1).strip()
        
        match_digits = re.search(r'\b\d{5,10}\b', found_text)
        if match_digits: return match_digits.group(0)
    except Exception as e:
        pass
    return None

def enhance_po_image(po_region):
    if po_region is None or po_region.size == 0: return None
    alpha = 1.3; beta = 10
    enhanced = cv2.convertScaleAbs(po_region, alpha=alpha, beta=beta)
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    sharp = cv2.filter2D(enhanced, -1, kernel)
    zoom = 2.0
    zoomed = cv2.resize(sharp, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    return zoomed

# =============================================================================
# BOX TRACKER
# =============================================================================

class BoxTracker:
    def __init__(self):
        self.active = False
        self.source = TrackingSource.NONE
        self.rect = None
        self.center = None
        self.last_update_time = 0
        self.pos_history = deque(maxlen=STABILITY_BUFFER)
       
    @property
    def is_stable(self):
        if len(self.pos_history) < STABILITY_BUFFER: return False
        xs = [p[0] for p in self.pos_history]
        ys = [p[1] for p in self.pos_history]
        mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
        max_d = max(math.sqrt((p[0]-mx)**2 + (p[1]-my)**2) for p in self.pos_history)
        return max_d < STABILITY_THRESH

    def update(self, rect, source, now):
        self.active = True
        self.source = source
        self.rect = rect
        self.center = rect_center(rect)
        self.last_update_time = now
        self.pos_history.append(self.center)
       
    def predict(self):
        return self.center

    def check_timeout(self, now):
        if self.active and (now - self.last_update_time > TRACKING_TIMEOUT_MS):
            self.active = False
            self.source = TrackingSource.NONE
            self.pos_history.clear()

# =============================================================================
# MAIN SYSTEM
# =============================================================================

class BoxCountingSystem:
    def __init__(self):
        self.state = BoxState.NONE
        self.total_counts = {}
        self.tracker = BoxTracker()
       
        print(">>> Initializing Motion Detection <<<")
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
        
        print(">>> Loading PaddleOCR Model (English) <<<")
        self.ocr = PaddleOCR(use_textline_orientation=True, lang='en')
       
        self.current_barcode = None
        self.current_barcode_rect = None
        self.current_po = None
        self.alert_msg = None
        self.alert_time = 0
       
        self.mask_debug = None
        self.debug_ocr_roi = None
        self.debug_scan_area = None
        self.debug_barcode_roi = None 
       
    def set_alert(self, msg):
        self.alert_msg = msg
        self.alert_time = time.time() * 1000
       
    def get_rois(self, w, h):
        # Simplified: Exit Top, Working Bottom
        exit_h = int(h * ROI_EXIT_TOP_H)
        r_exit = (0, 0, w, exit_h)
        r_working = (0, exit_h, w, h - exit_h)
        return r_exit, r_working

    def determine_state_from_pos(self, rect, w, h):
        cx, cy = rect_center(rect)
        r_exit, r_working = self.get_rois(w, h)
        if rect_overlap(((cx, cy, 1, 1)), r_exit): return BoxState.EXIT
        # If in Working rect (which is everywhere else), it's WORKING
        return BoxState.WORKING

    def update(self, frame):
        h, w = frame.shape[:2]
        now = time.time() * 1000
       
        # 1. MOTION
        motion_rects, self.mask_debug = detect_motion(frame, self.bg_subtractor)
        largest_motion = get_largest_rect(motion_rects)
       
        # 2. BARCODE
        barcodes = decode(frame)
        self.debug_barcode_roi = frame
       
        if not barcodes:
             bright = cv2.convertScaleAbs(frame, alpha=1.5, beta=50)
             self.debug_barcode_roi = bright
             barcodes = decode(bright)
       
        # 3. TRACKING FUSION
        best_rect = None
        best_source = TrackingSource.NONE
        pred = self.tracker.predict()
        self.debug_scan_area = (0, 0, w, h)

        if barcodes:
             if pred:
                  best_bc = min(barcodes, key=lambda b: distance(rect_center(b.rect), pred))
             else:
                  best_bc = max(barcodes, key=lambda b: b.rect[2]*b.rect[3])
             best_rect = best_bc.rect
             best_source = TrackingSource.BARCODE
             self.current_barcode = best_bc.data.decode("utf-8")
             self.current_barcode_rect = best_rect
             
        elif largest_motion:
             m_center = rect_center(largest_motion)
             if pred:
                  dist = distance(m_center, pred)
                  if dist < w * 0.5:
                       best_rect = largest_motion
                       best_source = TrackingSource.MOTION
             else:
                  best_rect = largest_motion
                  best_source = TrackingSource.MOTION
                 
        # Update Tracker
        if best_rect:
            self.tracker.update(best_rect, best_source, now)
        else:
            self.tracker.check_timeout(now)
            if self.tracker.active:
                self.tracker.source = TrackingSource.COASTING

        # 5. FSM LOGIC
        if self.tracker.active and self.tracker.rect:
            region = self.determine_state_from_pos(self.tracker.rect, w, h)
               
            # SIMPLIFIED: Only Transitions are NONE->WORKING, WORKING->EXIT
            
            if self.state == BoxState.NONE:
                if region == BoxState.WORKING:
                    # New Entry from anywhere in working zone
                    self.state = BoxState.WORKING
                    if best_source == TrackingSource.MOTION:
                        self.current_barcode = None
                        self.current_po = None
                        self.current_barcode_rect = None
                elif region == BoxState.EXIT:
                    # Appearing directly in Exit zone? Ignore or count?
                    # V5 logic usually requires seeing it in working first, but user said "entering anywhere except top"
                    # If it appears in TOP, we probably ignore it as it's leaving or noise
                    pass

            elif self.state == BoxState.WORKING:
                if region == BoxState.EXIT:
                    self.state = BoxState.EXIT
                else:
                    # Still Working
                    # Scan Logic
                    if self.tracker.is_stable and not self.current_po:
                        self.attempt_ocr(frame)

            elif self.state == BoxState.EXIT:
                if region == BoxState.WORKING:
                    self.set_alert("DROP BACK")
                    self.state = BoxState.WORKING
                if self.tracker.is_stable and not self.current_po:
                    self.attempt_ocr(frame)
                   
            elif self.state == BoxState.COUNTED:
                if region == BoxState.WORKING:
                    # New object or re-entry? 
                    # Treat as re-entry for now -> reset
                    self.state = BoxState.WORKING
                    self.current_barcode = None
                    self.current_po = None
                    self.current_barcode_rect = None
                   
        else:
            if self.state == BoxState.EXIT:
                self.finish_count()
                self.state = BoxState.COUNTED
            elif self.state not in [BoxState.NONE, BoxState.COUNTED]:
                # Lost track in Working zone
                if self.state == BoxState.WORKING:
                     # self.set_alert("LOST TRACK")
                     pass
                self.state = BoxState.NONE
                self.current_barcode = None
                self.current_po = None

    def attempt_ocr(self, frame):
        h, w = frame.shape[:2]
        if self.current_barcode_rect:
             bx, by, bw, bh = self.current_barcode_rect
        elif self.tracker.rect:
             bx, by, bw, bh = self.tracker.rect
        else:
             return
       
        scan_x = bx
        scan_y = by + bh + 10
        scan_w = bw
        scan_h = 80
       
        sx1 = max(0, scan_x); sy1 = max(0, scan_y)
        sx2 = min(w, scan_x + scan_w); sy2 = min(h, scan_y + scan_h)
       
        if (sx2 - sx1) > 10 and (sy2 - sy1) > 10:
            roi = frame[sy1:sy2, sx1:sx2]
            enhanced = enhance_po_image(roi)
            self.debug_ocr_roi = enhanced
            val = extract_po_number(enhanced, self.ocr)
            if val:
                print(f"[OCR] Found PO: {val}")
                self.current_po = val

    def finish_count(self):
        bc = self.current_barcode
        po = self.current_po or "Unknown"
        if not bc:
            print(">>> IGNORED: NO BARCODE Detected <<<")
            return

        if bc not in self.total_counts: self.total_counts[bc] = {}
        if po not in self.total_counts[bc]: self.total_counts[bc][po] = 0
        self.total_counts[bc][po] += 1
        print(f"\n>>> COUNTED <<< Barcode: {bc} | PO: {po} | Total: {self.total_counts[bc][po]}\n")

    def draw(self, frame):
        h, w = frame.shape[:2]
        r_exit, r_working = self.get_rois(w, h)
        
        # Draw Zones
        # EXIT (Red)
        cv2.rectangle(frame, (r_exit[0], r_exit[1]), (r_exit[0]+r_exit[2], r_exit[1]+r_exit[3]), (0,0,255), 2)
        cv2.putText(frame, "EXIT", (10, r_exit[3]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        
        # WORKING (Cyan)
        # Just draw boundary line
        # cv2.rectangle(frame, (r_working[0], r_working[1]), (r_working[0]+r_working[2], r_working[1]+r_working[3]), (255,255,0), 2)
        cv2.line(frame, (0, r_working[1]), (w, r_working[1]), (255,255,0), 2)
        cv2.putText(frame, "WORKING ZONE", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
       
        if self.tracker.active and self.tracker.rect:
            x, y, bw, bh = self.tracker.rect
            col = (0,0,255)
            if self.tracker.source == TrackingSource.BARCODE: col = (0,255,0)
            elif self.tracker.source == TrackingSource.MOTION: col = (255,100,0)
           
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), col, 2)
            lbl = f"{self.tracker.source.name} [{'STABLE' if self.tracker.is_stable else 'MOV'}]"
            cv2.putText(frame, lbl, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
           
        cv2.rectangle(frame, (0,0), (w, 80), (0,0,0), -1)
        cnt = sum(sum(x.values()) for x in self.total_counts.values())
        cv2.putText(frame, f"STATE: {self.state.name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"COUNT: {cnt}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
       
        bc = self.current_barcode if self.current_barcode else "--"
        po = self.current_po if self.current_po else "--"
        cv2.putText(frame, f"BC: {bc}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cv2.putText(frame, f"PO: {po}", (300, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
       
        if self.alert_msg and (time.time()*1000 - self.alert_time < 3000):
            sz = cv2.getTextSize(self.alert_msg, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            cv2.putText(frame, self.alert_msg, ((w-sz[0])//2, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): return
    
    # Optional: Set higher resolution for better OCR
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    sys = BoxCountingSystem()
    print(">>> SIMPLIFIED MOTION SYSTEM START <<<")
   
    while True:
        ret, frame = cap.read()
        if not ret: break
       
        sys.update(frame)
        sys.draw(frame)
       
        cv2.imshow("Box System", frame)
        if sys.mask_debug is not None:
             cv2.imshow("Motion Mask", sys.mask_debug)
        if sys.debug_ocr_roi is not None:
             cv2.imshow("PO Scan Area", sys.debug_ocr_roi)
             
        if cv2.waitKey(1) & 0xFF == ord('q'): break
       
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
