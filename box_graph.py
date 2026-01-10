import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend to avoid threading issues with OpenCV
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import time

class BoxGraphSimple:
    """
    Simplified real-time graph visualization for BoxProcessor metrics
    Uses manual updates instead of animation to avoid threading issues
    """
    
    def __init__(self, window_seconds=60):
        """
        Args:
            window_seconds: Time window to display (seconds)
        """
        self.window_seconds = window_seconds
        
        # Data storage (deques for efficient append/pop)
        self.max_points = window_seconds * 30  # Assume ~30 FPS
        self.times = deque(maxlen=self.max_points)
        self.states = deque(maxlen=self.max_points)
        self.presence = deque(maxlen=self.max_points)
        self.stationary = deque(maxlen=self.max_points)
        self.blur_variance = deque(maxlen=self.max_points)
        self.vepp_l1 = deque(maxlen=self.max_points)
        
        self.start_time = time.time()
        
        # State color mapping
        self.state_colors = {
            0: '#808080',  # IDLE - Gray
            1: '#FFA500',  # MOVING - Orange
            2: '#FFFF00',  # STABLE - Yellow
            3: '#00BFFF',  # SCANNING - Blue
            4: '#00FF00',  # SUCCESS - Green
            5: '#FF0000',  # ERROR - Red
        }
        
        self.state_names = {
            0: 'IDLE',
            1: 'MOVING',
            2: 'STABLE',
            3: 'SCANNING',
            4: 'SUCCESS',
            5: 'ERROR',
        }
        
        # Create figure
        plt.ion()  # Interactive mode
        self.fig, (self.ax_state, self.ax_metrics) = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True
        )
        
        self.fig.suptitle('BoxProcessor Real-Time Metrics', fontsize=14, fontweight='bold')
        
        # Setup state axis
        self.ax_state.set_ylabel('State', fontsize=10)
        self.ax_state.set_ylim(-0.5, 5.5)
        self.ax_state.set_yticks(range(6))
        self.ax_state.set_yticklabels(
            ['IDLE', 'MOVING', 'STABLE', 'SCANNING', 'SUCCESS', 'ERROR']
        )
        self.ax_state.grid(True, alpha=0.3)
        
        # Setup metrics axis
        self.ax_metrics.set_ylabel('Metrics', fontsize=10)
        self.ax_metrics.set_xlabel('Time (seconds)', fontsize=10)
        self.ax_metrics.set_ylim(-0.1, 1.1)
        self.ax_metrics.grid(True, alpha=0.3)
        
        # Initialize line plots
        self.state_line, = self.ax_state.plot([], [], 'o-', markersize=3, linewidth=2, color='blue')
        self.presence_line, = self.ax_metrics.plot([], [], 'o-', label='Presence', markersize=2, color='green')
        self.stationary_line, = self.ax_metrics.plot([], [], 'o-', label='Stationary', markersize=2, color='orange')
        self.blur_line, = self.ax_metrics.plot([], [], '-', label='Blur (norm)', linewidth=1.5, color='purple', alpha=0.7)
        self.vepp_line, = self.ax_metrics.plot([], [], '-', label='VEPP L1 (norm)', linewidth=1.5, color='red', alpha=0.7)
        
        self.ax_metrics.legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        plt.show(block=False)
        
        self.update_counter = 0
    
    def add_data_point(self, state_value, presence, stationary, blur_variance, vepp_l1):
        """
        Add a new data point
        
        Args:
            state_value: State enum value (0-5)
            presence: Boolean (True/False)
            stationary: Boolean (True/False)
            blur_variance: Float (blur variance value)
            vepp_l1: Float (VEPP L1 distance)
        """
        current_time = time.time() - self.start_time
        
        self.times.append(current_time)
        self.states.append(state_value)
        self.presence.append(1.0 if presence else 0.0)
        self.stationary.append(1.0 if stationary else 0.0)
        
        # Normalize blur variance (typical range 0-500, cap at 300)
        normalized_blur = min(blur_variance / 300.0, 1.0)
        self.blur_variance.append(normalized_blur)
        
        # Normalize VEPP L1 (typical range 0-1, already normalized)
        self.vepp_l1.append(min(vepp_l1, 1.0))
    
    def update(self):
        """Update plot manually (call this in main loop)"""
        # Only update every 10 frames to reduce overhead
        self.update_counter += 1
        if self.update_counter < 10:
            return
        
        self.update_counter = 0
        
        if len(self.times) == 0:
            plt.pause(0.001)
            return
        
        # Convert to numpy arrays
        times_arr = np.array(self.times)
        states_arr = np.array(self.states)
        presence_arr = np.array(self.presence)
        stationary_arr = np.array(self.stationary)
        blur_arr = np.array(self.blur_variance)
        vepp_arr = np.array(self.vepp_l1)
        
        # Update state line
        self.state_line.set_data(times_arr, states_arr)
        
        # Update metrics lines
        self.presence_line.set_data(times_arr, presence_arr)
        self.stationary_line.set_data(times_arr, stationary_arr)
        self.blur_line.set_data(times_arr, blur_arr)
        self.vepp_line.set_data(times_arr, vepp_arr)
        
        # Auto-scroll x-axis
        if len(times_arr) > 0:
            current_time = times_arr[-1]
            if current_time > self.window_seconds:
                self.ax_state.set_xlim(current_time - self.window_seconds, current_time)
                self.ax_metrics.set_xlim(current_time - self.window_seconds, current_time)
            else:
                self.ax_state.set_xlim(0, self.window_seconds)
                self.ax_metrics.set_xlim(0, self.window_seconds)
        
        # Redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
    
    def close(self):
        """Close the graph window"""
        plt.close(self.fig)

if __name__ == "__main__":
    # Test the graph
    import random
    
    graph = BoxGraphSimple(window_seconds=30)
    
    print("Testing graph with random data... Press Ctrl+C to stop")
    
    try:
        state = 0
        for i in range(1000):
            # Simulate state transitions
            if i % 50 == 0:
                state = (state + 1) % 6
            
            # Random metrics
            presence = random.choice([True, False])
            stationary = random.choice([True, False])
            blur = random.uniform(0, 300)
            vepp = random.uniform(0, 0.5)
            
            graph.add_data_point(state, presence, stationary, blur, vepp)
            graph.update()
            
            time.sleep(0.033)  # ~30 FPS
    
    except KeyboardInterrupt:
        print("\nStopping test...")
    
    graph.close()
