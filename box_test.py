import cv2
import numpy as np
import time
from box_processor import BoxProcessor, AppState
from box_graph import BoxGraphSimple
from box_feature_chart import BoxFeatureChart

class BoxTest:
    """
    Main test script for BoxProcessor with real-time visualization
    Similar to main_test.py but using BoxProcessor logic
    """
    
    def __init__(self, camera_id=0, width=640, height=480, show_graph=False):
        """
        Args:
            camera_id: Camera device ID
            width: Frame width
            height: Frame height
            show_graph: Whether to show real-time graph
        """
        # Camera setup
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # BoxProcessor
        self.processor = BoxProcessor()
        
        # Graph
        self.show_graph = show_graph
        self.graph = None
        if show_graph:
            self.graph = BoxGraphSimple(window_seconds=60)
        
        # Feature Chart (ALWAYS ENABLED)
        self.chart = BoxFeatureChart(width=400, height=600)
        
        # State colors for visualization
        self.state_colors = {
            AppState.IDLE: (128, 128, 128),      # Gray
            AppState.MOVING: (0, 165, 255),      # Orange
            AppState.STABLE: (0, 255, 255),      # Yellow
            AppState.SCANNING: (255, 191, 0),    # Blue
            AppState.SUCCESS: (0, 255, 0),       # Green
            AppState.ERROR: (0, 0, 255),         # Red
        }
    
    def run(self):
        """Main loop"""
        print(">>> Starting BoxProcessor Test... Press 'q' to exit, 'p' to simulate OCR")
        
        # Create named windows
        cv2.namedWindow("BoxProcessor Test", cv2.WINDOW_NORMAL)
        if self.chart:
            cv2.namedWindow("Feature Chart", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Feature Chart", 400, 600)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Simulate OCR input (press 'p' to trigger)
            ocr_result = None
            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'):
                ocr_result = "P123456"  # Mock PO number
                print(f"[Test] Simulated OCR: {ocr_result}")
            
            # Update BoxProcessor
            self.processor.update_logic(gray, ocr_result)
            
            # Update graph
            if self.graph:
                self.graph.add_data_point(
                    self.processor.current_state.value,
                    self.processor.last_presence,
                    self.processor.last_stationary,
                    self.processor.last_blur_variance,
                    self.processor.last_vepp_l1
                )
                self.graph.update()
            
            # Update feature chart (ALWAYS SHOW)
            chart_img = self.chart.create_chart(self.processor, gray)
            cv2.imshow("Feature Chart", chart_img)
            
            # Visualize on frame
            self._draw_visualization(frame, gray)
            
            # Show frame
            cv2.imshow("BoxProcessor Test", frame)
            
            # Exit on 'q'
            if key == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        if self.graph:
            self.graph.close()
        
        print("Test completed.")
    
    def _draw_visualization(self, frame, gray):
        """Draw visualization overlay on frame"""
        h, w = frame.shape[:2]
        
        # Get current state
        state = self.processor.current_state
        state_color = self.state_colors.get(state, (255, 255, 255))
        
        # Draw state banner at top
        cv2.rectangle(frame, (0, 0), (w, 60), state_color, -1)
        cv2.putText(
            frame,
            f"STATE: {state.name}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 0),
            3
        )
        
        # Draw feedback message
        cv2.putText(
            frame,
            self.processor.feedback_message,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )
        
        # Draw metrics
        y_offset = 120
        metrics = [
            f"Presence: {self.processor.last_presence}",
            f"Stationary: {self.processor.last_stationary}",
            f"Blur Variance: {self.processor.last_blur_variance:.1f}",
            f"VEPP L1: {self.processor.last_vepp_l1:.3f}",
        ]
        
        for metric in metrics:
            cv2.putText(
                frame,
                metric,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )
            y_offset += 25
        
        # Draw ROI rectangle
        roi = self.processor._get_center_roi(gray)
        x, y, rw, rh = roi
        cv2.rectangle(frame, (x, y), (x + rw, y + rh), (0, 255, 0), 2)
        cv2.putText(
            frame,
            "ROI",
            (x + 5, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )
        
        # Draw PO if available
        if self.processor.po:
            cv2.putText(
                frame,
                f"PO: {self.processor.po}",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )
        
        # Draw instructions
        cv2.putText(
            frame,
            "Press 'p' to simulate OCR | 'q' to quit",
            (10, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

if __name__ == "__main__":
    import sys
    
    # Check if --graph flag is provided
    show_graph = "--graph" in sys.argv
    
    if show_graph:
        print("Note: Graph display enabled. May cause threading issues on some systems.")
        print("If you experience crashes, run without --graph flag.")
    
    test = BoxTest(camera_id=0, width=640, height=480, show_graph=show_graph)
    test.run()
