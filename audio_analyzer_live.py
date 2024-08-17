import tkinter as tk
from tkinter import filedialog
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sounddevice as sd
import threading
import datetime


class AudioPopDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Pop Sound Detector")

        # Frame for instructions
        self.instructions_frame = tk.Frame(self.root)
        self.instructions_frame.pack(pady=10)
        self.instructions_label = tk.Label(
            self.instructions_frame, text="Instructions: Load an audio file, record live audio, set the amplitude threshold, and detect pop sounds.")
        self.instructions_label.pack()

        # Frame for audio controls
        self.controls_frame = tk.Frame(self.root)
        self.controls_frame.pack(pady=10)

        # Load Audio Button
        self.load_button = tk.Button(self.controls_frame, text="Load Audio File", command=self.load_audio)
        self.load_button.grid(row=0, column=0, padx=10)

        # Record/Stop Audio Button
        self.record_button = tk.Button(self.controls_frame, text="Start Recording", command=self.toggle_recording)
        self.record_button.grid(row=0, column=1, padx=10)

        # Detect Button
        self.detect_button = tk.Button(self.controls_frame, text="Detect Pop Sound", command=self.detect_pop_sound)
        self.detect_button.grid(row=0, column=2, padx=10)

        # Threshold Scale
        self.threshold_label = tk.Label(self.controls_frame, text="Amplitude Threshold")
        self.threshold_label.grid(row=1, column=0, padx=10)
        self.threshold_scale = tk.Scale(self.controls_frame, from_=0, to=1, resolution=0.01, orient="horizontal", length=300)
        self.threshold_scale.set(0.4)
        self.threshold_scale.grid(row=1, column=1, padx=10)

        # Frame for additional info
        self.info_frame = tk.Frame(self.root)
        self.info_frame.pack(pady=10)

        self.file_info_label = tk.Label(self.info_frame, text="Audio File Information:")
        self.file_info_label.pack()

        self.audio_duration_label = tk.Label(self.info_frame, text="Duration: N/A")
        self.audio_duration_label.pack()

        self.sampling_rate_label = tk.Label(self.info_frame, text="Sampling Rate: N/A")
        self.sampling_rate_label.pack()

        self.pop_count_label = tk.Label(self.info_frame, text="Detected Pops: N/A")
        self.pop_count_label.pack()

        self.file_name_label = tk.Label(self.info_frame, text="File Name: N/A")
        self.file_name_label.pack()

        self.metadata_label = tk.Label(self.info_frame, text="Recording Date and Time: N/A")
        self.metadata_label.pack()

        # Canvas for matplotlib graph
        self.figure, self.ax = plt.subplots(figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack()

        # Variables for audio data
        self.y = np.array([])
        self.sr = 44100  # Default sample rate for recording
        self.recording = False
        self.stream = None
        self.plot_thread = None

    def load_audio(self):
        # Load audio file using a file dialog
        self.file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.m4a *.flac")])
        if self.file_path:
            # Load the audio file using librosa for waveform analysis
            self.y, self.sr = librosa.load(self.file_path, sr=None)

            # Update info labels
            duration = librosa.get_duration(y=self.y, sr=self.sr)
            self.audio_duration_label.config(text=f"Duration: {duration:.2f} seconds")
            self.sampling_rate_label.config(text=f"Sampling Rate: {self.sr} Hz")
            self.file_name_label.config(text=f"File Name: {self.file_path.split('/')[-1]}")

            # Plot the waveform
            self.plot_waveform()

    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        # Start recording live audio using sounddevice
        self.recording = True
        self.record_button.config(text="Stop Recording")
        self.y = np.array([])  # Reset the audio data

        # Start a separate thread to handle the recording
        self.stream = sd.InputStream(callback=self.audio_callback, channels=1, samplerate=self.sr)
        self.stream.start()

        # Start a thread to update the plot periodically
        if self.plot_thread is None or not self.plot_thread.is_alive():
            self.plot_thread = threading.Thread(target=self.update_plot, daemon=True)
            self.plot_thread.start()

    def stop_recording(self):
        # Stop recording and stop the stream
        self.recording = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
        
        self.record_button.config(text="Start Recording")
        self.metadata_label.config(text=f"Recording Date and Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def audio_callback(self, indata, frames, time, status):
        # This is called for every audio block captured by sounddevice
        if self.recording:
            self.y = np.append(self.y, indata.flatten())  # Append the new audio data

    def update_plot(self):
        # Update the plot every 1 second if recording is ongoing
        while self.recording:
            self.root.after(0, self.plot_waveform)
            sd.sleep(1000)  # Sleep for 1 second

    def plot_waveform(self):
        # Plot the audio waveform
        if len(self.y) > 0:
            self.ax.clear()

            # Generate the time axis based on the current length of self.y
            times = np.arange(len(self.y)) / self.sr

            # Ensure the lengths of times and y match before plotting
            if len(times) == len(self.y):
                self.ax.plot(times, self.y, label='Live Audio Waveform')
                self.ax.set_xlabel('Time (s)')
                self.ax.set_ylabel('Amplitude')
                self.ax.set_title('Audio Amplitude Over Time')
                self.ax.legend()
                self.canvas.draw()

    def detect_pop_sound(self):
        if len(self.y) > 0:
            # Get the amplitude threshold value from the scale
            threshold = self.threshold_scale.get()

            # Detect and highlight pop sounds based on amplitude
            self.highlight_pop_sounds(threshold)

    def highlight_pop_sounds(self, threshold):
        # Perform pop sound detection based on amplitude threshold
        times = np.arange(len(self.y)) / self.sr
        pop_indices = np.where(np.abs(self.y) > threshold)[0]
        pop_times = times[pop_indices]

        # Group pop times by 0.5-second intervals
        pop_half_seconds = set()
        for t in pop_times:
            half_second = int(t // 0.5) * 0.5  # Convert to 0.5-second intervals
            pop_half_seconds.add(half_second)

        # Update pop count label
        self.pop_count_label.config(text=f"Detected Pops: {len(pop_half_seconds)}")

        # Clear the plot and re-plot the waveform
        self.ax.clear()
        self.ax.plot(times, self.y, label='Audio Waveform')

        # Mark the detected pop sounds (grouped by 0.5-second intervals)
        if len(pop_half_seconds) > 0:
            print(f"Detected pop sound(s) above {threshold} amplitude at the following times (in 0.5-second intervals):")
            for half_sec in sorted(pop_half_seconds):
                print(f"{half_sec:.1f} seconds")
                self.ax.axvline(x=half_sec, color='r', linestyle='--', label=f'Pop at {half_sec:.1f}s')
        else:
            print(f"No pop sounds detected above {threshold} amplitude.")

        # Update the plot
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title('Audio Amplitude Over Time')
        self.ax.legend()
        self.canvas.draw()


# Create the main window
root = tk.Tk()
app = AudioPopDetector(root)
root.mainloop()
