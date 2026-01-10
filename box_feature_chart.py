import cv2
import numpy as np
from box_processor import BoxProcessor, AppState

class BoxFeatureChart:
    """
    Striped feature chart visualization for BoxProcessor
    Shows detection features as colored horizontal bars/stripes
    """
    
    def __init__(self, width=400, height=600):
        """
        Args:
            width: Chart width in pixels
            height: Chart height in pixels
        """
        self.width = width
        self.height = height
        
        # State colors (matching box_test.py)
        self.state_colors = {
            AppState.IDLE: (128, 128, 128),      # Gray
            AppState.MOVING: (0, 165, 255),      # Orange
            AppState.STABLE: (0, 255, 255),      # Yellow
            AppState.SCANNING: (255, 191, 0),    # Blue
            AppState.SUCCESS: (0, 255, 0),       # Green
            AppState.ERROR: (0, 0, 255),         # Red
        }
        
        # Feature history for sparklines
        self.history_length = 100
        self.presence_history = []
        self.stationary_history = []
        self.blur_history = []
        self.vepp_history = []
    
    def create_chart(self, processor: BoxProcessor, gray_frame=None):
        """
        Create the feature chart visualization
        
        Args:
            processor: BoxProcessor instance
            gray_frame: Optional grayscale frame for VEPP visualization
            
        Returns:
            Chart image (BGR format)
        """
        # Create blank chart
        chart = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        chart[:] = (30, 30, 30)  # Dark background
        
        y_offset = 20
        bar_height = 60
        spacing = 10
        
        # 1. STATE BANNER
        state_color = self.state_colors.get(processor.current_state, (255, 255, 255))
        cv2.rectangle(chart, (10, y_offset), (self.width - 10, y_offset + bar_height), state_color, -1)
        cv2.putText(
            chart,
            f"STATE: {processor.current_state.name}",
            (20, y_offset + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2
        )
        y_offset += bar_height + spacing
        
        # 2. PRESENCE STRIPE
        presence_color = (0, 255, 0) if processor.last_presence else (100, 100, 100)
        self._draw_feature_bar(
            chart,
            y_offset,
            bar_height,
            "PRESENCE",
            processor.last_presence,
            presence_color
        )
        y_offset += bar_height + spacing
        
        # 3. STATIONARY STRIPE
        stationary_color = (0, 255, 255) if processor.last_stationary else (100, 100, 100)
        self._draw_feature_bar(
            chart,
            y_offset,
            bar_height,
            "STATIONARY",
            processor.last_stationary,
            stationary_color
        )
        y_offset += bar_height + spacing
        
        # 4. BLUR VARIANCE STRIPE (with value bar)
        blur_normalized = min(processor.last_blur_variance / 300.0, 1.0)
        blur_color = self._get_gradient_color(blur_normalized, (0, 0, 255), (0, 255, 0))
        self._draw_value_bar(
            chart,
            y_offset,
            bar_height,
            "BLUR VARIANCE",
            processor.last_blur_variance,
            blur_normalized,
            blur_color
        )
        y_offset += bar_height + spacing
        
        # 5. VEPP L1 STRIPE (with value bar)
        vepp_normalized = 1.0 - min(processor.last_vepp_l1 / 0.5, 1.0)  # Invert: lower L1 = more stationary
        vepp_color = self._get_gradient_color(vepp_normalized, (0, 0, 255), (0, 255, 0))
        self._draw_value_bar(
            chart,
            y_offset,
            bar_height,
            "VEPP L1",
            processor.last_vepp_l1,
            vepp_normalized,
            vepp_color
        )
        y_offset += bar_height + spacing
        
        # 6. VEPP PROFILE VISUALIZATION (if gray frame provided)
        if gray_frame is not None and processor.prev_vepp_profile is not None:
            self._draw_vepp_profile(chart, y_offset, processor.prev_vepp_profile)
            y_offset += 80 + spacing
        
        # 7. FEEDBACK MESSAGE
        cv2.putText(
            chart,
            processor.feedback_message,
            (20, y_offset + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        y_offset += 40
        
        # 8. PO DISPLAY (if available)
        if processor.po:
            cv2.rectangle(chart, (10, y_offset), (self.width - 10, y_offset + 50), (0, 255, 0), 2)
            cv2.putText(
                chart,
                f"PO: {processor.po}",
                (20, y_offset + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )
        
        return chart
    
    def _draw_feature_bar(self, chart, y, height, label, value, color):
        """Draw a simple on/off feature bar"""
        # Background
        cv2.rectangle(chart, (10, y), (self.width - 10, y + height), (50, 50, 50), -1)
        
        # Fill if active
        if value:
            cv2.rectangle(chart, (10, y), (self.width - 10, y + height), color, -1)
        
        # Border
        cv2.rectangle(chart, (10, y), (self.width - 10, y + height), (200, 200, 200), 2)
        
        # Label
        text_color = (0, 0, 0) if value else (200, 200, 200)
        cv2.putText(
            chart,
            f"{label}: {'ON' if value else 'OFF'}",
            (20, y + height // 2 + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            2
        )
    
    def _draw_value_bar(self, chart, y, height, label, raw_value, normalized_value, color):
        """Draw a value bar with gradient fill"""
        # Background
        cv2.rectangle(chart, (10, y), (self.width - 10, y + height), (50, 50, 50), -1)
        
        # Fill bar based on normalized value
        fill_width = int((self.width - 20) * normalized_value)
        if fill_width > 0:
            cv2.rectangle(chart, (10, y), (10 + fill_width, y + height), color, -1)
        
        # Border
        cv2.rectangle(chart, (10, y), (self.width - 10, y + height), (200, 200, 200), 2)
        
        # Label and value
        cv2.putText(
            chart,
            label,
            (20, y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        cv2.putText(
            chart,
            f"{raw_value:.2f}",
            (20, y + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        # Percentage
        cv2.putText(
            chart,
            f"{int(normalized_value * 100)}%",
            (self.width - 80, y + height // 2 + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    
    def _draw_vepp_profile(self, chart, y, vepp_profile):
        """Draw VEPP profile as vertical bars (barcode-like visualization)"""
        profile_height = 80
        
        # Background
        cv2.rectangle(chart, (10, y), (self.width - 10, y + profile_height), (0, 0, 0), -1)
        
        # Label
        cv2.putText(
            chart,
            "VEPP PROFILE (Vertical Edge)",
            (20, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (150, 150, 150),
            1
        )
        
        # Draw profile as vertical bars
        profile_len = len(vepp_profile)
        if profile_len > 0:
            bar_width = max(1, (self.width - 20) // profile_len)
            
            for i, val in enumerate(vepp_profile):
                if i * bar_width >= self.width - 20:
                    break
                
                # Height based on value (0-1 normalized)
                bar_height = int(val * (profile_height - 25))
                
                # Color gradient from blue to yellow
                color = self._get_gradient_color(val, (255, 0, 0), (0, 255, 255))
                
                # Draw bar from bottom up
                x1 = 10 + i * bar_width
                y1 = y + profile_height - bar_height
                x2 = x1 + bar_width - 1
                y2 = y + profile_height
                
                cv2.rectangle(chart, (x1, y1), (x2, y2), color, -1)
        
        # Border
        cv2.rectangle(chart, (10, y), (self.width - 10, y + profile_height), (100, 100, 100), 1)
    
    def _get_gradient_color(self, value, color_start, color_end):
        """
        Get gradient color between two colors based on value (0-1)
        
        Args:
            value: 0-1 normalized value
            color_start: BGR tuple for value=0
            color_end: BGR tuple for value=1
            
        Returns:
            BGR tuple
        """
        value = max(0.0, min(1.0, value))
        
        b = int(color_start[0] + (color_end[0] - color_start[0]) * value)
        g = int(color_start[1] + (color_end[1] - color_start[1]) * value)
        r = int(color_start[2] + (color_end[2] - color_start[2]) * value)
        
        return (b, g, r)

if __name__ == "__main__":
    # Test the feature chart
    from box_processor import BoxProcessor
    import time
    
    cap = cv2.VideoCapture(0)
    processor = BoxProcessor()
    chart_viz = BoxFeatureChart(width=400, height=600)
    
    print(">>> Testing Feature Chart... Press 'q' to exit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Update processor
        processor.update_logic(gray)
        
        # Create chart
        chart = chart_viz.create_chart(processor, gray)
        
        # Show both
        cv2.imshow("Camera", frame)
        cv2.imshow("Feature Chart", chart)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
