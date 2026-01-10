import cv2
import numpy as np
import time
from enum import Enum
from typing import Optional, Tuple

class AppState(Enum):
    IDLE = 0
    MOVING = 1
    STABLE = 2
    SCANNING = 3
    SUCCESS = 4
    ERROR = 5

class Config:
    """Configuration matching Config.kt"""
    # Presence Detection (Morphology)
    GRADIENT_THRESHOLD = 50.0
    MORPH_KERNEL_SIZE = (21, 3)  # Width, Height - narrower to avoid vertical noise
    MIN_BARCODE_AREA_RATIO = 0.03
    MIN_ASPECT_RATIO = 2.5  # Stricter ratio - barcodes are wider than squares
    PRESENCE_CONFIRM_FRAMES = 2
    PRESENCE_LOST_FRAMES = 2
    
    # Texture Validation (Barcode Density)
    # Real barcodes show ~6-7% edge density after Sobel + threshold
    MIN_TEXTURE_DENSITY = 0.05  # 5% edge pixels minimum
    MAX_TEXTURE_DENSITY = 0.20  # 20% edge pixels maximum
    
    # Stationary Detection
    STATIONARY_DIFF_PIXEL_THRESHOLD = 20.0
    STATIONARY_CHANGED_RATIO = 0.01
    STATIONARY_CONFIRM_FRAMES = 3  # Reduced from 5 for faster response
    
    # Blur/Focus Detection
    BLUR_THRESHOLD = 100.0
    
    # ROI & Scanning
    ROI_WIDTH_RATIO = 0.5
    ROI_HEIGHT_RATIO = 0.5
    SCAN_TIMEOUT_MS = 3000
    ERROR_RETRY_COOLDOWN_MS = 1200
    
    # VEPP (Vertical Edge Projection Profile)
    # Increased threshold to tolerate hand shake and minor movements
    BARCODE_STATIONARY_L1_THRESHOLD = 2.5  # Was 0.15, now allows L1 up to 2.5

class OCRFusion:
    """OCR Fusion with majority voting"""
    def __init__(self, max_frames=5, min_agree=3):
        self.max_frames = max_frames
        self.min_agree = min_agree
        self.results = []
    
    def reset(self):
        self.results.clear()
    
    def add(self, raw: str):
        clean = self._normalize(raw)
        if self._is_valid_format(clean):
            self.results.append(clean)
        if len(self.results) > self.max_frames:
            self.results.pop(0)
    
    def is_ready(self) -> bool:
        return len(self.results) >= self.min_agree
    
    def get_fused(self) -> Optional[str]:
        """Majority vote per character"""
        if not self.is_ready():
            return None
        
        # Find most common length
        length_counts = {}
        for r in self.results:
            length_counts[len(r)] = length_counts.get(len(r), 0) + 1
        
        if not length_counts:
            return None
        
        length = max(length_counts.items(), key=lambda x: x[1])[0]
        same_len = [r for r in self.results if len(r) == length]
        
        if len(same_len) < self.min_agree:
            return None
        
        # Vote per character
        result = []
        for i in range(length):
            char_votes = {}
            for s in same_len:
                char_votes[s[i]] = char_votes.get(s[i], 0) + 1
            
            if not char_votes:
                return None
            
            best_char = max(char_votes.items(), key=lambda x: x[1])[0]
            result.append(best_char)
        
        return ''.join(result)
    
    def _normalize(self, s: str) -> str:
        return s.upper().replace(" ", "").replace("\n", "")
    
    def _is_valid_format(self, s: str) -> bool:
        """Validate PO format: P + 5-7 digits"""
        import re
        return bool(re.match(r'^P[0-9]{5,7}$', s))

class BoxProcessor:
    """
    Python port of BoxProcessor.kt
    Handles barcode detection state machine with presence, stationary, and blur detection
    """
    
    def __init__(self):
        # Public state
        self.current_state = AppState.IDLE
        self.feedback_message = "READY"
        self.barcode = None
        self.po = None
        
        # Internal state
        self.state_start_time = 0
        self.error_time = 0
        
        self.presence_frames = 0
        self.lost_frames = 0
        self.stationary_frames = 0
        
        # OCR
        self.ocr_fusion = OCRFusion()
        
        # VEPP (Vertical Edge Projection Profile)
        self.prev_vepp_profile = None
        
        # ROI cache
        self.roi_rect = None
        
        # Morphology kernel
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            Config.MORPH_KERNEL_SIZE
        )
        
        # Metrics for visualization
        self.last_presence = False
        self.last_stationary = False
        self.last_blur_variance = 0.0
        self.last_vepp_l1 = 0.0
    
    def update_logic(self, gray: np.ndarray, ocr_result: Optional[str] = None):
        """
        Main update function - call once per frame
        
        Args:
            gray: Grayscale frame (numpy array)
            ocr_result: Optional OCR result string
        """
        now = int(time.time() * 1000)  # milliseconds
        
        presence = self._detect_presence_robust(gray)
        stationary = self._detect_barcode_stationary_vepp(gray, presence)
        
        # Store for visualization
        self.last_presence = presence
        self.last_stationary = stationary
        
        # State machine
        if self.current_state == AppState.IDLE:
            self.feedback_message = "READY"
            if presence:
                self.presence_frames += 1
                if self.presence_frames >= Config.PRESENCE_CONFIRM_FRAMES:
                    print(f"[BoxProcessor] STATE: IDLE -> MOVING")
                    self.current_state = AppState.MOVING
                    self._reset_counters()
            else:
                self.presence_frames = 0
        
        elif self.current_state == AppState.MOVING:
            self.feedback_message = "MOVING..."
            if not presence:
                self._reset_to_idle()
            elif stationary:
                self.stationary_frames += 1
                if self.stationary_frames >= Config.STATIONARY_CONFIRM_FRAMES:
                    print(f"[BoxProcessor] STATE: MOVING -> STABLE")
                    self.current_state = AppState.STABLE
                    self._reset_counters()
            else:
                self.stationary_frames = 0
        
        elif self.current_state == AppState.STABLE:
            self.feedback_message = "CHECKING FOCUS..."
            if not presence:
                self._reset_to_idle()
            else:
                roi = self._get_center_roi(gray)
                roi_mat = gray[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
                if self._is_image_sharp(roi_mat):
                    print(f"[BoxProcessor] STATE: STABLE -> SCANNING")
                    self.current_state = AppState.SCANNING
                    self.ocr_fusion.reset()
                    self.state_start_time = now
        
        elif self.current_state == AppState.SCANNING:
            self.feedback_message = "SCANNING..."
            elapsed = now - self.state_start_time
            
            if not presence or elapsed > Config.SCAN_TIMEOUT_MS:
                self._mark_error()
            elif ocr_result is not None:
                self.ocr_fusion.add(ocr_result)
                fused = self.ocr_fusion.get_fused()
                if fused is not None:
                    self.po = fused
                    self.feedback_message = "SUCCESS"
                    self.current_state = AppState.SUCCESS
                    print(f"[BoxProcessor] STATE: SCANNING -> SUCCESS (PO: {fused})")
        
        elif self.current_state == AppState.SUCCESS:
            self.feedback_message = "SUCCESS"
            if not presence:
                self.lost_frames += 1
                if self.lost_frames >= Config.PRESENCE_LOST_FRAMES:
                    self._reset_to_idle()
            else:
                self.lost_frames = 0
        
        elif self.current_state == AppState.ERROR:
            self.feedback_message = "ERROR – ADJUST BOX"
            if not presence:
                self._reset_to_idle()
            elif stationary and (now - self.error_time) > Config.ERROR_RETRY_COOLDOWN_MS:
                self.stationary_frames += 1
                if self.stationary_frames >= Config.STATIONARY_CONFIRM_FRAMES:
                    print(f"[BoxProcessor] STATE: ERROR -> STABLE")
                    self.current_state = AppState.STABLE
                    self._reset_counters()
            else:
                self.stationary_frames = 0
    
    def _detect_presence_robust(self, gray: np.ndarray) -> bool:
        """
        Detect barcode presence using morphology with texture validation
        Returns True if barcode-like structure is detected
        """
        try:
            # 1. Sobel gradient (vertical edges detection)
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
            abs_grad_x = cv2.convertScaleAbs(grad_x)
            
            # 2. Threshold
            _, threshold_mat = cv2.threshold(
                abs_grad_x,
                Config.GRADIENT_THRESHOLD,
                255,
                cv2.THRESH_BINARY
            )
            
            # 3. Morphological closing (connect barcode lines horizontally)
            morph_mat = cv2.morphologyEx(
                threshold_mat,
                cv2.MORPH_CLOSE,
                self.morph_kernel
            )
            
            # 4. Find contours
            contours, _ = cv2.findContours(
                morph_mat,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # 5. Filter by area, aspect ratio, and texture
            frame_area = gray.shape[0] * gray.shape[1]
            min_area = frame_area * Config.MIN_BARCODE_AREA_RATIO
            
            candidates = 0
            for i, cnt in enumerate(contours):
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                ratio = w / h if h > 0 else 0
                
                # Basic shape filter: must be wide and large enough
                if area > min_area and ratio > Config.MIN_ASPECT_RATIO:
                    candidates += 1
                    # === CRITICAL: TEXTURE VALIDATION ===
                    # Check ROI from threshold image (before morphology) to verify barcode texture
                    roi_check = threshold_mat[y:y+h, x:x+w]
                    density = self._get_texture_density(roi_check)
                    is_valid = self._validate_barcode_texture(roi_check)
                    
                    print(f"[Presence] Candidate {candidates}: area={area:.0f}, ratio={ratio:.2f}, density={density:.3f} -> {is_valid}")
                    
                    if is_valid:
                        print(f"[Presence] ✓ BARCODE DETECTED")
                        return True
            
            if candidates == 0:
                print(f"[Presence] {len(contours)} contours, 0 candidates (all too small/narrow)")
            else:
                print(f"[Presence] ✗ {candidates} candidates, none valid (density out of range)")
            return False
        
        except Exception as e:
            print(f"[BoxProcessor] Presence error: {e}")
            return False
    
    def _get_texture_density(self, binary_roi: np.ndarray) -> float:
        """
        Calculate edge density of a binary ROI
        
        Args:
            binary_roi: Binary image ROI
            
        Returns:
            Density value (0.0 to 1.0)
        """
        total_pixels = binary_roi.shape[0] * binary_roi.shape[1]
        if total_pixels == 0:
            return 0.0
        
        non_zero = cv2.countNonZero(binary_roi)
        return non_zero / total_pixels
    
    def _validate_barcode_texture(self, binary_roi: np.ndarray) -> bool:
        """
        Validate that the ROI has barcode-like texture (concentrated stripe cluster)
        
        Args:
            binary_roi: Binary image ROI from threshold (before morphology)
            
        Returns:
            True if has barcode-like concentrated vertical edge pattern
        """
        density = self._get_texture_density(binary_roi)
        
        # Step 1: Basic density check
        if not (0.05 <= density <= 0.20):
            return False
        
        h, w = binary_roi.shape
        if h < 5 or w < 10:
            return False
        
        # Step 2: Column-based clustering analysis
        # Real barcodes: Dense cluster of adjacent columns with edges
        # Noise: Isolated spikes scattered across width
        
        # Sum edges per column (vertical profile)
        col_sums = np.sum(binary_roi > 0, axis=0)
        
        # Normalize to 0-1
        if col_sums.max() > 0:
            col_profile = col_sums / col_sums.max()
        else:
            return False
        
        # Find columns with significant activity (>30% of max)
        active_threshold = 0.3
        active_cols = col_profile > active_threshold
        
        # Count consecutive runs of active columns
        # Real barcode: Long consecutive runs (clustered)
        # Noise: Short isolated runs (scattered)
        
        runs = []
        current_run = 0
        for is_active in active_cols:
            if is_active:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                current_run = 0
        if current_run > 0:
            runs.append(current_run)
        
        if not runs:
            print(f"[Texture] Rejected: no active column runs")
            return False
        
        # Get longest run
        max_run = max(runs)
        
        # RULE: Real barcode should have at least one run of 15+ consecutive columns
        # This filters out scattered noise which has runs of 1-5 columns
        min_cluster_width = 15
        has_barcode_cluster = max_run >= min_cluster_width
        
        if not has_barcode_cluster:
            print(f"[Texture] Rejected: max_run={max_run} cols (need >={min_cluster_width})")
        
        return has_barcode_cluster
    
    def _detect_barcode_stationary_vepp(self, gray: np.ndarray, presence: bool) -> bool:
        """
        Detect if barcode is stationary using VEPP (Vertical Edge Projection Profile)
        
        Args:
            gray: Grayscale frame
            presence: Whether barcode is present
            
        Returns:
            True if barcode is stationary
        """
        # No barcode = not stationary
        if not presence:
            self.prev_vepp_profile = None
            print(f"[VEPP] No presence -> not stationary")
            return False
        
        try:
            # 1. Get center ROI
            roi = self._get_center_roi(gray)
            roi_mat = gray[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
            
            # 2. Sobel gradient (horizontal edges)
            vepp_grad_x = cv2.Sobel(roi_mat, cv2.CV_32F, 1, 0)
            vepp_abs_grad_x = cv2.convertScaleAbs(vepp_grad_x)
            
            # 3. Sum columns (vertical edge profile)
            vepp_profile = np.sum(vepp_abs_grad_x, axis=0, dtype=np.float32)
            
            # 4. Normalize to [0, 1]
            if vepp_profile.max() > 0:
                vepp_profile = vepp_profile / vepp_profile.max()
            
            # 5. Compare with previous frame
            if self.prev_vepp_profile is None:
                self.prev_vepp_profile = vepp_profile.copy()
                print(f"[VEPP] First frame -> initializing profile")
                return False
            
            # 6. L1 norm (Manhattan distance)
            l1 = np.sum(np.abs(vepp_profile - self.prev_vepp_profile))
            
            # Store for visualization
            self.last_vepp_l1 = l1
            
            # Update previous
            self.prev_vepp_profile = vepp_profile.copy()
            
            # 7. Check threshold
            stationary = l1 < Config.BARCODE_STATIONARY_L1_THRESHOLD
            
            print(f"[VEPP] L1={l1:.4f} (threshold={Config.BARCODE_STATIONARY_L1_THRESHOLD}) -> stationary={stationary}")
            
            return stationary
        
        except Exception as e:
            print(f"[BoxProcessor] VEPP error: {e}")
            return False
    
    def _is_image_sharp(self, roi: np.ndarray) -> bool:
        """
        Check if image is sharp using Laplacian variance
        
        Args:
            roi: Region of interest (grayscale)
            
        Returns:
            True if image is sharp (in focus)
        """
        try:
            laplacian = cv2.Laplacian(roi, cv2.CV_64F)
            variance = laplacian.var()
            
            # Store for visualization
            self.last_blur_variance = variance
            
            is_sharp = variance > Config.BLUR_THRESHOLD
            print(f"[Blur] Variance={variance:.1f} (threshold={Config.BLUR_THRESHOLD}) -> sharp={is_sharp}")
            
            return is_sharp
        
        except Exception as e:
            print(f"[BoxProcessor] Blur check error: {e}")
            return False
    
    def _get_center_roi(self, gray: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Get center ROI rectangle (x, y, w, h)
        
        Args:
            gray: Grayscale frame
            
        Returns:
            Tuple of (x, y, width, height)
        """
        h, w = gray.shape
        rw = int(w * Config.ROI_WIDTH_RATIO)
        rh = int(h * Config.ROI_HEIGHT_RATIO)
        
        if self.roi_rect is None or self.roi_rect[2] != rw or self.roi_rect[3] != rh:
            self.roi_rect = ((w - rw) // 2, (h - rh) // 2, rw, rh)
        
        return self.roi_rect
    
    def _mark_error(self):
        """Mark state as ERROR"""
        self.current_state = AppState.ERROR
        self.error_time = int(time.time() * 1000)
        self._reset_counters()
        print(f"[BoxProcessor] STATE: -> ERROR")
    
    def _reset_to_idle(self):
        """Reset to IDLE state"""
        self.current_state = AppState.IDLE
        self.feedback_message = "READY"
        self.barcode = None
        self.po = None
        self.prev_vepp_profile = None
        self._reset_counters()
        print(f"[BoxProcessor] STATE: -> IDLE (reset)")
    
    def _reset_counters(self):
        """Reset frame counters"""
        self.presence_frames = 0
        self.lost_frames = 0
        self.stationary_frames = 0
