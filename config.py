
# config.py
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
    YOLO_MODEL = "YOLOV8s_Barcode_Detection.pt" 
    CONF_THRESH = 0.4

    # Database & Validation
    USE_MOCK = False         # Toggle: True=Local Dict, False=Real SQL Server
    DB_SERVER = "192.168.30.115"
    DB_NAME = "Adidas_Shoebox"
    DB_USER = "odin"
    DB_PASS = "Perthr0"
    
    # Anti-Spam / Logic
    API_COOLDOWN_SECONDS = 1.5
