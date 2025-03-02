import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pyaudio
import wave
import threading
import time
import os
import serial
import struct
from scipy import signal
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

class SpeechRecognitionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech Recognition Interface")
        self.root.geometry("1200x800")
        self.root.configure(bg="#2E3440")  # Nordic dark theme
        
        # Current working directory for saving files
        self.working_dir = os.getcwd()
        
        # Audio recording parameters
        self.is_armed = False
        self.is_recording = False
        self.frames = []
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        self.audio_format = pyaudio.paInt16
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.record_thread = None
        self.record_time = 0
        self.timer_thread = None
        self.stop_timer = False
        self.threshold = 0.1  # Amplitude threshold for auto recording
        self.recording_duration = 1.0  # Auto recording duration in seconds
        self.record_start_time = 0
        
        # UART parameters
        self.uart_port = None
        self.uart_baud = 115200
        self.uart_thread = None
        self.uart_done = False
        self.fpga_response = ""
        self.response_window = None
        
        # Audio data and features
        self.audio_data = np.zeros(self.sample_rate * 5)  # 5 seconds buffer
        self.mel_features = np.zeros((16, 100))  # Initial placeholder
        self.feature_dim = 16  # Number of mel bins
        
        # Create mel filterbank
        self.mel_filter = self._create_mel_filterbank()
        
        # Create the main frames
        self.create_frames()
        
        # Create interface elements
        self.create_control_buttons()
        self.create_timer_display()
        self.create_visualizations()
        
        # Animation update function
        self.animation = FuncAnimation(self.fig, self.update_plots, interval=100, cache_frame_data=False)
        
        # Configure style
        self.style_ui()
        
        # Create response window
        self.create_response_window()
        
    def style_ui(self):
        """Apply a modern dark theme style to UI elements"""
        style = ttk.Style()
        style.theme_use("clam")
        
        # Configure colors
        style.configure("TButton", 
                      background="#4C566A", 
                      foreground="#ECEFF4", 
                      font=("Arial", 12, "bold"),
                      borderwidth=0,
                      padding=10)
        
        style.map("TButton",
                background=[("active", "#5E81AC"), ("pressed", "#81A1C1")],
                foreground=[("active", "#ECEFF4")])
        
        style.configure("Arm.TButton", background="#BF616A")
        style.map("Arm.TButton", background=[("active", "#D08770"), ("pressed", "#EBCB8B")])
        
        style.configure("Stop.TButton", background="#A3BE8C")
        style.map("Stop.TButton", background=[("active", "#B48EAD"), ("pressed", "#EBCB8B")])
        
        style.configure("Reset.TButton", background="#EBCB8B")
        style.map("Reset.TButton", background=[("active", "#D08770"), ("pressed", "#BF616A")])
        
        style.configure("TLabel", 
                      background="#2E3440", 
                      foreground="#ECEFF4",
                      font=("Arial", 12))
        
        style.configure("Timer.TLabel", 
                      font=("Arial", 24, "bold"),
                      foreground="#88C0D0")
        
        style.configure("Status.TLabel", 
                      font=("Arial", 16, "bold"),
                      foreground="#A3BE8C")
        
        # Apply style to frames
        for frame in [self.top_left_frame, self.top_right_frame, self.bottom_frame]:
            frame.configure(style="TFrame")
        
        style.configure("TFrame", background="#2E3440")
        
        # Configure scale
        style.configure("Horizontal.TScale", 
                       background="#2E3440",
                       troughcolor="#4C566A",
                       sliderrelief="flat")
    
    def create_frames(self):
        """Create the main layout frames"""
        # Create top frame (upper half)
        self.top_frame = ttk.Frame(self.root)
        self.top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create top left and right frames
        self.top_left_frame = ttk.Frame(self.top_frame)
        self.top_left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.top_right_frame = ttk.Frame(self.top_frame)
        self.top_right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create bottom frame (lower half)
        self.bottom_frame = ttk.Frame(self.root)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def create_control_buttons(self):
        """Create the control buttons in the top left frame"""
        # Title label
        title_label = ttk.Label(self.top_left_frame, text="Speech Recognition Control", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Button frame
        button_frame = ttk.Frame(self.top_left_frame)
        button_frame.pack(pady=10)
        
        # Arm button (replaces Record)
        self.arm_button = ttk.Button(button_frame, text="Arm", style="Arm.TButton",
                                    command=self.arm_recording)
        self.arm_button.grid(row=0, column=0, padx=10, pady=10)
        
        # Reset button
        self.reset_button = ttk.Button(button_frame, text="Reset", style="Reset.TButton",
                                     command=self.reset_system)
        self.reset_button.grid(row=0, column=1, padx=10, pady=10)
        self.reset_button.state(['disabled'])
        
        # Threshold setting frame
        threshold_frame = ttk.Frame(self.top_left_frame)
        threshold_frame.pack(pady=10, fill=tk.X, padx=20)
        
        # Threshold label
        ttk.Label(threshold_frame, text="Amplitude Threshold:").pack(side=tk.TOP, anchor=tk.W)
        
        # Threshold scale
        self.threshold_var = tk.DoubleVar(value=0.1)
        threshold_scale = ttk.Scale(threshold_frame, from_=0.01, to=0.5, 
                                  variable=self.threshold_var, 
                                  orient=tk.HORIZONTAL,
                                  length=200,
                                  command=self.update_threshold)
        threshold_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        # Threshold value display
        self.threshold_value_label = ttk.Label(threshold_frame, text="0.10")
        self.threshold_value_label.pack(side=tk.RIGHT, padx=5)
        
        # UART settings frame
        uart_frame = ttk.Frame(self.top_left_frame)
        uart_frame.pack(pady=20)
        
        # UART port selection
        ttk.Label(uart_frame, text="UART Port:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        
        # Get available COM ports
        available_ports = self.get_available_ports()
        self.port_var = tk.StringVar()
        if available_ports:
            self.port_var.set(available_ports[0])
        
        port_dropdown = ttk.Combobox(uart_frame, textvariable=self.port_var, values=available_ports, width=15)
        port_dropdown.grid(row=0, column=1, padx=5, pady=5)
        
        # Baud rate selection
        ttk.Label(uart_frame, text="Baud Rate:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        
        baud_rates = ["9600", "19200", "38400", "57600", "115200"]
        self.baud_var = tk.StringVar(value="115200")
        
        baud_dropdown = ttk.Combobox(uart_frame, textvariable=self.baud_var, values=baud_rates, width=15)
        baud_dropdown.grid(row=1, column=1, padx=5, pady=5)
        
        # Save location
        save_frame = ttk.Frame(self.top_left_frame)
        save_frame.pack(pady=10, fill=tk.X, padx=20)
        
        ttk.Label(save_frame, text="Save Location:").pack(side=tk.TOP, anchor=tk.W)
        ttk.Label(save_frame, text=self.working_dir, 
                foreground="#88C0D0", font=("Consolas", 10)).pack(side=tk.TOP, anchor=tk.W, pady=5)
        
        # Output file base name
        output_frame = ttk.Frame(self.top_left_frame)
        output_frame.pack(pady=10, fill=tk.X, padx=20)
        
        ttk.Label(output_frame, text="Output File Base Name:").pack(side=tk.TOP, anchor=tk.W)
        
        self.output_basename = tk.StringVar(value="recording")
        basename_entry = ttk.Entry(output_frame, textvariable=self.output_basename, width=20)
        basename_entry.pack(side=tk.TOP, fill=tk.X, pady=5)
    
    def create_timer_display(self):
        """Create the timer and status display in the top right frame"""
        # Title label
        title_label = ttk.Label(self.top_right_frame, text="Status Monitor", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Status frame
        status_frame = ttk.Frame(self.top_right_frame)
        status_frame.pack(pady=20)
        
        # Recording timer
        timer_label = ttk.Label(status_frame, text="Recording Time:", font=("Arial", 12))
        timer_label.grid(row=0, column=0, padx=10, pady=10, sticky='w')
        
        self.timer_display = ttk.Label(status_frame, text="00:00.0", style="Timer.TLabel")
        self.timer_display.grid(row=0, column=1, padx=10, pady=10)
        
        # System status
        status_label = ttk.Label(status_frame, text="System Status:", font=("Arial", 12))
        status_label.grid(row=1, column=0, padx=10, pady=10, sticky='w')
        
        self.system_status = ttk.Label(status_frame, text="Ready", style="Status.TLabel")
        self.system_status.grid(row=1, column=1, padx=10, pady=10)
        
        # Current amplitude
        amp_label = ttk.Label(status_frame, text="Current Amplitude:", font=("Arial", 12))
        amp_label.grid(row=2, column=0, padx=10, pady=10, sticky='w')
        
        self.amplitude_display = ttk.Label(status_frame, text="0.00", style="Status.TLabel",
                                        foreground="#88C0D0")
        self.amplitude_display.grid(row=2, column=1, padx=10, pady=10)
        
        # Animation frame (for visual indicators)
        self.led_canvas = tk.Canvas(self.top_right_frame, width=250, height=100, 
                                  bg="#2E3440", highlightthickness=0)
        self.led_canvas.pack(pady=10)
        
        # Create LED indicators
        self.armed_led = self.led_canvas.create_oval(30, 30, 60, 60, fill="#434C5E", outline="#4C566A")
        self.recording_led = self.led_canvas.create_oval(130, 30, 160, 60, fill="#434C5E", outline="#4C566A")
        self.uart_led = self.led_canvas.create_oval(230, 30, 260, 60, fill="#434C5E", outline="#4C566A")
        
        # LED labels
        self.led_canvas.create_text(45, 75, text="Armed", fill="#D8DEE9", font=("Arial", 10))
        self.led_canvas.create_text(145, 75, text="Recording", fill="#D8DEE9", font=("Arial", 10))
        self.led_canvas.create_text(245, 75, text="UART", fill="#D8DEE9", font=("Arial", 10))
        
        # Recording counter
        counter_frame = ttk.Frame(self.top_right_frame)
        counter_frame.pack(pady=10)
        
        ttk.Label(counter_frame, text="Recordings made:").grid(row=0, column=0, padx=10, pady=5, sticky='w')
        
        self.recording_counter = tk.IntVar(value=0)
        counter_display = ttk.Label(counter_frame, textvariable=self.recording_counter, 
                                  font=("Arial", 16, "bold"), foreground="#EBCB8B")
        counter_display.grid(row=0, column=1, padx=10, pady=5)
    
    def create_visualizations(self):
        """Create the audio and spectrogram visualizations in the bottom frame"""
        # Create figure with two subplots
        self.fig = Figure(figsize=(12, 6), dpi=100, facecolor="#2E3440")
        
        # Adjust margins
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.3)
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(211)  # Audio waveform
        self.ax2 = self.fig.add_subplot(212)  # Mel spectrogram
        
        # Style the plots
        for ax in [self.ax1, self.ax2]:
            ax.set_facecolor("#3B4252")
            ax.tick_params(colors="#D8DEE9", which='both')
            ax.spines['bottom'].set_color("#4C566A")
            ax.spines['top'].set_color("#4C566A") 
            ax.spines['right'].set_color("#4C566A")
            ax.spines['left'].set_color("#4C566A")
            
        # Set titles and labels
        self.ax1.set_title("Audio Waveform", color="#ECEFF4", fontsize=14)
        self.ax1.set_ylabel("Amplitude", color="#E5E9F0", fontsize=12)
        
        self.ax2.set_title("Mel Spectrogram", color="#ECEFF4", fontsize=14)
        self.ax2.set_xlabel("Time Frame", color="#E5E9F0", fontsize=12)
        self.ax2.set_ylabel("Mel Frequency Bin", color="#E5E9F0", fontsize=12)
        
        # Initial plots
        self.waveform_plot, = self.ax1.plot(np.arange(1000)/self.sample_rate, 
                                           np.zeros(1000), 
                                           color="#88C0D0", linewidth=1)
        
        # Add threshold line
        self.threshold_line, = self.ax1.plot([0, 1000/self.sample_rate], 
                                           [self.threshold, self.threshold], 
                                           color="#BF616A", linestyle='--', linewidth=1)
        
        self.threshold_neg_line, = self.ax1.plot([0, 1000/self.sample_rate], 
                                               [-self.threshold, -self.threshold], 
                                               color="#BF616A", linestyle='--', linewidth=1)
        
        # Mel spectrogram
        self.spectrogram_plot = self.ax2.imshow(
            np.zeros((16, 100)),
            aspect='auto',
            origin='lower',
            interpolation='nearest',
            cmap='viridis',
            extent=[0, 100, 0, 16]
        )
        
        # Add colorbar
        cbar = self.fig.colorbar(self.spectrogram_plot, ax=self.ax2)
        cbar.ax.tick_params(colors="#D8DEE9")
        cbar.set_label("Energy (dB)", color="#E5E9F0")
        
        # Add the plots to the tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.bottom_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    def create_response_window(self):
        """Create a separate window for FPGA responses"""
        self.response_window = tk.Toplevel(self.root)
        self.response_window.title("FPGA Response")
        self.response_window.geometry("500x400")
        self.response_window.configure(bg="#2E3440")
        
        # Make it not appear as a separate application in taskbar
        self.response_window.transient(self.root)
        
        # Title label
        title_label = ttk.Label(self.response_window, text="FPGA Output", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Text area for responses
        self.response_text = scrolledtext.ScrolledText(self.response_window, 
                                                    wrap=tk.WORD, 
                                                    width=50, 
                                                    height=15, 
                                                    font=("Consolas", 12),
                                                    bg="#3B4252",
                                                    fg="#ECEFF4")
        self.response_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Save settings
        save_frame = ttk.Frame(self.response_window)
        save_frame.pack(pady=10, fill=tk.X, padx=10)
        
        ttk.Label(save_frame, text="Response Log File:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        
        self.log_filename = tk.StringVar(value="fpga_responses.txt")
        log_entry = ttk.Entry(save_frame, textvariable=self.log_filename, width=30)
        log_entry.grid(row=0, column=1, sticky='we', padx=5, pady=5)
        
        # Clear button
        clear_button = ttk.Button(save_frame, text="Clear", 
                                command=self.clear_response_text)
        clear_button.grid(row=0, column=2, padx=5, pady=5)
        
        # Hide initially - will show when needed
        self.response_window.withdraw()
    
    def clear_response_text(self):
        """Clear the FPGA response text area"""
        self.response_text.delete(1.0, tk.END)
    
    def update_threshold(self, value=None):
        """Update the amplitude threshold value"""
        self.threshold = self.threshold_var.get()
        self.threshold_value_label.config(text=f"{self.threshold:.2f}")
        
        # Update threshold lines in the plot
        self.threshold_line.set_ydata([self.threshold, self.threshold])
        self.threshold_neg_line.set_ydata([-self.threshold, -self.threshold])
    
    def update_plots(self, frame):
        """Update the visualizations with new audio data"""
        if self.is_armed or self.is_recording:
            # Display the last few seconds of audio
            display_length = min(self.sample_rate, len(self.audio_data))
            display_start = max(0, len(self.audio_data) - display_length)
            display_data = self.audio_data[display_start:display_start + display_length]
            
            # Calculate current amplitude (for threshold detection)
            current_amplitude = np.max(np.abs(display_data[-self.chunk_size:]))
            self.amplitude_display.config(text=f"{current_amplitude:.2f}")
            
            # Update waveform
            self.waveform_plot.set_data(
                np.arange(len(display_data))/self.sample_rate, 
                display_data
            )
            self.ax1.set_xlim(0, len(display_data)/self.sample_rate)
            
            # Adjust y limits to the data with some margin
            max_val = max(0.2, current_amplitude * 1.5)
            self.ax1.set_ylim(-max_val, max_val)
            
            # Update threshold lines to match the x range
            x_range = [0, len(display_data)/self.sample_rate]
            self.threshold_line.set_xdata(x_range)
            self.threshold_neg_line.set_xdata(x_range)
            
            # Check for threshold crossing if armed but not recording
            if self.is_armed and not self.is_recording and current_amplitude > self.threshold:
                self.start_recording()
            
            # Update mel spectrogram - ensure it's properly displayed
            if hasattr(self, 'mel_features') and self.mel_features is not None:
                if isinstance(self.mel_features, np.ndarray) and self.mel_features.size > 0:
                    # Update the spectrogram image
                    self.spectrogram_plot.set_array(self.mel_features)
                    # Update extent to match the actual data dimensions
                    self.spectrogram_plot.set_extent([0, self.mel_features.shape[1], 0, self.mel_features.shape[0]])
        
        return self.waveform_plot, self.spectrogram_plot, self.threshold_line, self.threshold_neg_line
    
    def update_timer_display(self):
        """Update the recording timer display"""
        start_time = time.time()
        
        while self.is_recording:
            current_time = time.time() - self.record_start_time
            minutes = int(current_time) // 60
            seconds = int(current_time) % 60
            tenths = int((current_time * 10) % 10)
            
            time_str = f"{minutes:02d}:{seconds:02d}.{tenths:1d}"
            self.timer_display.config(text=time_str)
            
            # Flash recording LED
            flash_state = int(current_time * 2) % 2 == 0
            self.led_canvas.itemconfig(self.recording_led, 
                                     fill="#BF616A" if flash_state else "#D08770")
            
            # Check if recording duration exceeded
            if current_time >= self.recording_duration:
                self.stop_recording()
                break
            
            time.sleep(0.1)
    
    def arm_recording(self):
        """Arm the system to automatically start recording when threshold is crossed"""
        self.is_armed = True
        self.system_status.config(text="Armed")
        
        # Update UI
        self.arm_button.state(['disabled'])
        self.reset_button.state(['!disabled'])
        
        # Turn on the armed LED
        self.led_canvas.itemconfig(self.armed_led, fill="#EBCB8B")
        
        # Initialize audio stream for monitoring
        self.stream = self.p.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )
        
        # Start the stream
        self.stream.start_stream()
        
        print("System armed - waiting for audio threshold...")
    
    def start_recording(self):
        """Start audio recording when threshold is crossed"""
        if not self.is_recording:
            self.is_recording = True
            self.frames = []
            self.record_start_time = time.time()
            self.system_status.config(text="Recording")
            
            # Start timer in a separate thread
            self.timer_thread = threading.Thread(target=self.update_timer_display)
            self.timer_thread.daemon = True
            self.timer_thread.start()
            
            print("Recording started automatically...")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Process audio chunks as they arrive"""
        if status:
            print(f"Audio stream status: {status}")
        
        try:
            # Save frame for WAV file if recording
            if self.is_recording:
                self.frames.append(in_data)
            
            # Convert audio to numpy array
            if self.audio_format == pyaudio.paFloat32:
                chunk_data = np.frombuffer(in_data, dtype=np.float32)
            elif self.audio_format == pyaudio.paInt16:
                chunk_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32767.0
            
            # Add to the audio buffer (with rolling)
            new_len = len(chunk_data)
            self.audio_data = np.roll(self.audio_data, -new_len)
            self.audio_data[-new_len:] = chunk_data
            
            # Extract features for the full buffer
            self.update_features()
            
            return (in_data, pyaudio.paContinue)
        
        except Exception as e:
            print(f"Error in audio callback: {e}")
            return (in_data, pyaudio.paAbort)
    
    def update_features(self):
        """Extract mel-spectrogram features from current audio data"""
        try:
            # Apply pre-emphasis
            emphasized_audio = np.append(
                self.audio_data[0], 
                self.audio_data[1:] - 0.97 * self.audio_data[:-1]
            )
            
            # Compute STFT
            _, _, spectrogram = signal.stft(
                emphasized_audio,
                fs=self.sample_rate,
                nperseg=512,
                noverlap=256,
                return_onesided=True
            )
            
            # Get power spectrogram
            power_spectrogram = np.abs(spectrogram) ** 2
            
            # Apply mel filterbank
            mel_spectrogram = np.dot(self.mel_filter, power_spectrogram)
            
            # Log mel spectrogram
            log_mel_spectrogram = np.log(mel_spectrogram + 1e-9)
            
            # Only keep the most recent 100 frames or less
            num_frames = min(log_mel_spectrogram.shape[1], 100)
            self.mel_features = log_mel_spectrogram[:, -num_frames:]
            
        except Exception as e:
            print(f"Error updating features: {e}")
    
    def _create_mel_filterbank(self):
        """Create a mel filterbank for feature extraction"""
        # Frequency range for mel filters
        low_freq = 80
        high_freq = 7600
        
        # Number of FFT points
        nfft = 512
        
        # Create mel scale points
        mel_low = 2595 * np.log10(1 + low_freq / 700)
        mel_high = 2595 * np.log10(1 + high_freq / 700)
        mel_points = np.linspace(mel_low, mel_high, self.feature_dim + 2)
        
        # Convert mel points back to frequency
        freq_points = 700 * (10 ** (mel_points / 2595) - 1)
        
        # Convert frequency points to FFT bin indices
        bin_indices = np.floor((nfft + 1) * freq_points / self.sample_rate).astype(int)
        
        # Create filterbank
        filterbank = np.zeros((self.feature_dim, nfft // 2 + 1))
        
        for i in range(1, self.feature_dim + 1):
            start, center, end = bin_indices[i-1], bin_indices[i], bin_indices[i+1]
            
            # Create triangular filter (with bounds checking)
            if start < center:
                for j in range(start, center):
                    if 0 <= j < filterbank.shape[1]:
                        filterbank[i-1, j] = (j - start) / (center - start)
            
            if center < end:            
                for j in range(center, end):
                    if 0 <= j < filterbank.shape[1]:
                        filterbank[i-1, j] = (end - j) / (end - center)
                
        return filterbank
    
    def stop_recording(self):
        """Stop audio recording and automatically upload to FPGA"""
        if self.is_recording:
            self.is_recording = False
            
            # Set recording counter and status
            self.recording_counter.set(self.recording_counter.get() + 1)
            self.system_status.config(text="Processing")
            
            # Reset recording LED
            self.led_canvas.itemconfig(self.recording_led, fill="#434C5E")
            
            # Save audio file
            filename = self.save_audio()
            
            # Automatically send to FPGA
            if filename:
                self.send_to_fpga(filename)
            
            print("Recording stopped and sent to FPGA.")
    
    def save_audio(self):
        """Save the recorded audio to a WAV file"""
        if not self.frames:
            print("No audio data to save")
            return None
        
        try:
            # Create base filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            base_filename = self.output_basename.get()
            filename = f"{base_filename}_{timestamp}.wav"
                
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.p.get_sample_size(self.audio_format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.frames))
                
            print(f"Audio saved to {filename} in {self.working_dir}")
            return filename
            
        except Exception as e:
            print(f"Error saving audio: {e}")
            return None
    
    def send_to_fpga(self, filename):
        """Send the audio data to FPGA via UART"""
        # Update status
        self.system_status.config(text="Sending to FPGA")
        
        # Start UART transmission in a separate thread
        self.uart_thread = threading.Thread(target=lambda: self.uart_transmission(filename))
        self.uart_thread.daemon = True
        self.uart_thread.start()
    
    def uart_transmission(self, filename):
        """Handle UART transmission in a separate thread"""
        try:
            # Get port and baud rate
            port = self.port_var.get()
            baud = int(self.baud_var.get())
            
            # Connect to serial port
            print(f"Opening serial port {port} at {baud} baud...")
            ser = serial.Serial(port, baud, timeout=1)
            
            # Flash the UART LED to indicate active transmission
            self.led_canvas.itemconfig(self.uart_led, fill="#A3BE8C")
            
            # Load audio data from file
            with wave.open(filename, 'rb') as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                num_frames = wf.getnframes()
                
                print(f"Sending {num_frames} frames of audio...")
                
                # Read all frames
                raw_data = wf.readframes(num_frames)
                
                # Prepare data for transmission
                # Simple packet format: [0xAA][0xBB][data length (2 bytes)][audio data][checksum (1 byte)]
                packet_size = 256  # Send data in chunks
                total_packets = (len(raw_data) + packet_size - 1) // packet_size
                
                # Send packets
                for i in range(total_packets):
                    # Flash the UART LED
                    self.led_canvas.itemconfig(self.uart_led, 
                                            fill="#A3BE8C" if i % 2 == 0 else "#8FBCBB")
                    
                    # Get packet data
                    start = i * packet_size
                    end = min(start + packet_size, len(raw_data))
                    packet_data = raw_data[start:end]
                    
                    # Packet length
                    data_length = len(packet_data)
                    length_bytes = struct.pack('<H', data_length)  # 2 bytes for length
                    
                    # Checksum (sum of all bytes mod 256)
                    checksum = sum(packet_data) & 0xFF
                    
                    # Assemble header
                    header = bytes([0xAA, 0xBB]) + length_bytes
                    
                    # Send packet
                    ser.write(header)
                    ser.write(packet_data)
                    ser.write(bytes([checksum]))
                    
                    # Update status (fraction sent)
                    progress = (i + 1) / total_packets
                    self.system_status.config(text=f"Sending: {progress:.0%}")
                    
                    # Small delay to prevent overflowing the FPGA buffer
                    time.sleep(0.01)
            
            # Read response from FPGA (wait for 1 second)
            ser.timeout = 1.0
            fpga_response = ser.read(1024)  # Read up to 1024 bytes
            
            # Close the serial port
            ser.close()
            
            # Update status
            self.system_status.config(text="Completed")
            self.led_canvas.itemconfig(self.uart_led, fill="#A3BE8C")
            
            # Process FPGA response
            self.process_fpga_response(fpga_response, filename)
            
            # Reset the system to arm state
            self.root.after(1000, self.re_arm)
            
            print("UART transmission completed.")
            
        except Exception as e:
            print(f"Error in UART transmission: {e}")
            self.system_status.config(text="UART Failed")
            self.led_canvas.itemconfig(self.uart_led, fill="#BF616A")
            
            # Reset the system to arm state after delay
            self.root.after(2000, self.re_arm)
    
    def process_fpga_response(self, response_bytes, audio_filename):
        """Process and display the response from the FPGA"""
        try:
            # Convert bytes to string (if possible)
            if isinstance(response_bytes, bytes):
                try:
                    # Try UTF-8 decoding
                    response_str = response_bytes.decode('utf-8', errors='replace')
                except:
                    # Fallback to hex representation
                    response_str = "Binary data: " + response_bytes.hex()
            else:
                response_str = str(response_bytes)
            
            # Get audio filename without path
            audio_basename = os.path.basename(audio_filename)
            
            # Format the response with timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            formatted_response = f"[{timestamp}] {audio_basename}:\n{response_str}\n"
            formatted_response += "-" * 50 + "\n"
            
            # Show response window if not visible
            if not self.response_window.winfo_viewable():
                self.response_window.deiconify()
            
            # Add to response text area
            self.response_text.insert(tk.END, formatted_response)
            self.response_text.see(tk.END)
            
            # Save to log file
            log_filename = self.log_filename.get()
            with open(log_filename, 'a', encoding='utf-8') as f:
                f.write(formatted_response)
            
            print(f"FPGA response saved to {log_filename}")
            
        except Exception as e:
            print(f"Error processing FPGA response: {e}")
    
    def re_arm(self):
        """Reset the system to armed state after processing"""
        self.system_status.config(text="Armed")
        
        # Keep the armed state, but clear the recording flags
        self.is_recording = False
        
    def reset_system(self):
        """Reset the system to initial state"""
        # Stop recording/monitoring if active
        if self.stream and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        # Reset flags
        self.is_armed = False
        self.is_recording = False
        
        # Reset LEDs
        self.led_canvas.itemconfig(self.armed_led, fill="#434C5E")
        self.led_canvas.itemconfig(self.recording_led, fill="#434C5E")
        self.led_canvas.itemconfig(self.uart_led, fill="#434C5E")
        
        # Reset display
        self.timer_display.config(text="00:00.0")
        self.system_status.config(text="Ready")
        self.amplitude_display.config(text="0.00")
        
        # Reset buttons
        self.arm_button.state(['!disabled'])
        self.reset_button.state(['disabled'])
        
        print("System reset and ready.")
    
    def get_available_ports(self):
        """Get a list of available serial ports"""
        try:
            import serial.tools.list_ports
            ports = [port.device for port in serial.tools.list_ports.comports()]
            return ports if ports else ["COM1", "COM3", "/dev/ttyUSB0", "/dev/ttyACM0"]
        except:
            return ["COM1", "COM3", "/dev/ttyUSB0", "/dev/ttyACM0"]
    
    def on_closing(self):
        """Clean up resources when the application is closed"""
        if self.stream and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
        
        if self.p:
            self.p.terminate()
        
        # Close response window
        if self.response_window:
            self.response_window.destroy()
        
        self.root.destroy()

def main():
    root = tk.Tk()
    app = SpeechRecognitionUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
