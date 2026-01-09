import cv2
import numpy as np
import time

# --- MOCK CONFIG (Using values from config.py) ---
class Config:
    CAMERA_ID = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    
    # Sparse Optical Flow Config (Matching Android)
    MAX_TRACKING_POINTS = 50
    MIN_VALID_POINTS = 3
    QUALITY_LEVEL = 0.01
    MIN_DISTANCE = 10
    FLOW_WIN_SIZE = (21, 21) # Size for LK Flow
    
    # Motion Thresholds
    THRESH_ENTRY_X = 0.5
    THRESH_STABLE = 1.0
    
    # Logic
    DEBOUNCE_FRAMES = 3
    STABILITY_FRAMES = 3
    WORK_ZONE_X_MIN = 0.25
    WORK_ZONE_X_MAX = 0.75

class BoxFlowAnalyzerTest:
    def __init__(self):
        self.cap = cv2.VideoCapture(Config.CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        
        # State & Smoothing
        self.state = "IDLE"
        self.debounce_counter = 0
        self.stable_counter = 0
        self.feedback_msg = "READY"
        self.avg_vx = 0.0
        self.avg_vy = 0.0
        
        # Sparse Flow Data
        self.prev_gray = None
        self.prev_points = None

    def get_object_centroid(self, frame):
        # Logic from main.py: Thresholding to find object
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours: return None
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 3000: return None # Min area
            
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy, cv2.boundingRect(c)
        return None

    def run(self):
        print(">>> Starting TEST (SPARSE FLOW)... Press 'q' to exit.")
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # --- SPARSE OPTICAL FLOW LOGIC (Matching Android) ---
            
            # 1. Initialize features if needed
            if self.prev_points is None or len(self.prev_points) < 10:
                self.prev_points = cv2.goodFeaturesToTrack(
                    gray,
                    maxCorners=Config.MAX_TRACKING_POINTS,
                    qualityLevel=Config.QUALITY_LEVEL,
                    minDistance=Config.MIN_DISTANCE
                )
                self.prev_gray = gray.copy()
                # If still no points, just show frame
                if self.prev_points is None:
                    cv2.imshow("Main Logic Test", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                    continue

            # 2. Calculate Flow (Lucas-Kanade)
            next_points, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_points, None,
                winSize=Config.FLOW_WIN_SIZE, maxLevel=2
            )
            
            # 3. Filter Valid Points
            good_new = []
            good_old = []
            
            sum_dx = 0.0
            sum_dy = 0.0
            valid_count = 0
            
            if next_points is not None:
                # Select good points
                good_points_new = next_points[status == 1]
                good_points_old = self.prev_points[status == 1]
                
                for i, (new, old) in enumerate(zip(good_points_new, good_points_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    
                    dx = a - c
                    dy = b - d
                    
                    sum_dx += dx
                    sum_dy += dy
                    valid_count += 1
                    
                    good_new.append(new)
                    good_old.append(old)
                    
                    # Draw Points (Visualization)
                    cv2.circle(frame, (int(a), int(b)), 3, (0, 0, 255), -1)

            # 4. Calculate Average Velocity
            curr_vx = 0.0
            curr_vy = 0.0
            
            if valid_count >= Config.MIN_VALID_POINTS:
                curr_vx = sum_dx / valid_count
                curr_vy = sum_dy / valid_count
            else:
                # Not enough points - assume 0 or reset if starting
                if self.state == "IDLE":
                    self.avg_vx = 0.0
                    self.avg_vy = 0.0
            
            # 5. Apply Smoothing (EMA 0.7 / 0.3)
            self.avg_vx = curr_vx * 0.7 + self.avg_vx * 0.3
            self.avg_vy = curr_vy * 0.7 + self.avg_vy * 0.3
            mag = np.sqrt(self.avg_vx**2 + self.avg_vy**2)

            # Update for next frame
            self.prev_gray = gray.copy()
            if len(good_new) > 0:
                self.prev_points = np.array(good_new).reshape(-1, 1, 2)
            else:
                self.prev_points = None


            # --- STATE MACHINE (Exact same logic as before) ---
            if self.state == "IDLE":
                if abs(self.avg_vx) > Config.THRESH_ENTRY_X:
                    self.debounce_counter += 1
                else:
                    self.debounce_counter = 0
                
                if self.debounce_counter > Config.DEBOUNCE_FRAMES:
                    self.state = "SLIDING" # Renamed to match Android
                    self.stable_counter = 0
                    self.debounce_counter = 0
                    self.feedback_msg = "DETECTED MOTION"

            elif self.state == "SLIDING":
                if mag < Config.THRESH_STABLE:
                    self.stable_counter += 1
                else:
                    self.stable_counter = 0
                
                if self.stable_counter >= Config.STABILITY_FRAMES:
                    res = self.get_object_centroid(frame)
                    if res:
                        cx, cy, rect = res
                        w = Config.FRAME_WIDTH
                        if (w * Config.WORK_ZONE_X_MIN) < cx < (w * Config.WORK_ZONE_X_MAX):
                             self.state = "SCANNING"
                             self.feedback_msg = "SCAN TRIGGERED!"
                             print(">>> [TEST] SCAN TRIGGERED at Centroid X:", cx)
            
            elif self.state == "SCANNING":
                res = self.get_object_centroid(frame)
                if res:
                    cx, cy, rect = res
                    w = Config.FRAME_WIDTH
                    min_x = w * Config.WORK_ZONE_X_MIN
                    max_x = w * Config.WORK_ZONE_X_MAX
                    
                    if min_x < cx < max_x:
                        self.feedback_msg = "SCANNING (LOCKED)"
                    else:
                        self.state = "IDLE" 
                        self.feedback_msg = "RESET (EXITED ZONE)"
                else:
                    if mag < 1.0: pass 
                    else:
                         self.state = "IDLE"
                         self.feedback_msg = "RESET (NO OBJECT)"

            # --- VISUALIZATION ---
            h, w = frame.shape[:2]
            
            min_x = int(w * Config.WORK_ZONE_X_MIN)
            max_x = int(w * Config.WORK_ZONE_X_MAX)
            cv2.line(frame, (min_x, 0), (min_x, h), (255, 255, 0), 2)
            cv2.line(frame, (max_x, 0), (max_x, h), (255, 255, 0), 2)
            
            cx, cy = w // 2, h // 2
            end_x = int(cx + self.avg_vx * 50) 
            end_y = int(cy + self.avg_vy * 50)
            cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), (0, 255, 255), 3)
            
            res = self.get_object_centroid(frame)
            if res:
                cxx, cyy, (bx, by, bw, bh) = res
                cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
                cv2.circle(frame, (cxx, cyy), 5, (0, 0, 255), -1)

            cv2.putText(frame, f"STATE: {self.state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Points: {valid_count} | VX: {self.avg_vx:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(frame, self.feedback_msg, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Main Logic Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test = BoxFlowAnalyzerTest()
    test.run()
