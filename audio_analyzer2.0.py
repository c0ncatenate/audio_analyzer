import tkinter as tk
from tkinter import filedialog
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mutagen
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4
from mutagen.flac import FLAC
from mutagen.wave import WAVE
import datetime
from pydub import AudioSegment
import pygame
import io
import concurrent.futures

class AudioPopDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Pop Sound Detector")

        # Initialize pygame mixer
        pygame.mixer.init()

        # Frame for instructions
        self.instructions_frame = tk.Frame(self.root)
        self.instructions_frame.pack(pady=10)
        self.instructions_label = tk.Label(
            self.instructions_frame, text="Instructions: Load an audio file, set the amplitude threshold, and detect pop sounds.")
        self.instructions_label.pack()

        # Frame for audio controls
        self.controls_frame = tk.Frame(self.root)
        self.controls_frame.pack(pady=10)

        # Load Audio Button
        self.load_button = tk.Button(self.controls_frame, text="Load Audio File", command=self.load_audio)
        self.load_button.grid(row=0, column=0, padx=10)

        # Play/Stop Audio Button
        self.play_button = tk.Button(self.controls_frame, text="Play Audio", command=self.toggle_playback)
        self.play_button.grid(row=0, column=1, padx=10)

        # Detect Button
        self.detect_button = tk.Button(self.controls_frame, text="Detect Pop Sound", command=self.detect_pop_sound_parallel)
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
        self.y = None
        self.sr = None
        self.file_path = None
        self.audio_segment = None
        self.is_playing = False
        self.sound = None

    def load_audio(self):
        # Load audio file using a file dialog
        self.file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.m4a *.flac")])
        if self.file_path:
            # Load and downsample the audio file using librosa
            self.y, self.sr = librosa.load(self.file_path, sr=11025, mmap=True)  # Memory map and downsample to 11.025 kHz
            self.audio_segment = AudioSegment.from_file(self.file_path)

            # Extract metadata using mutagen
            metadata = self.get_metadata(self.file_path)

            # Update info labels
            duration = librosa.get_duration(y=self.y, sr=self.sr)
            self.audio_duration_label.config(text=f"Duration: {duration:.2f} seconds")
            self.sampling_rate_label.config(text=f"Sampling Rate: {self.sr} Hz")
            self.file_name_label.config(text=f"File Name: {self.file_path.split('/')[-1]}")  # Display only the filename

            if metadata and "date" in metadata:
                self.metadata_label.config(text=f"Recording Date and Time: {metadata['date']}")
            else:
                self.metadata_label.config(text="Recording Date and Time: Not Available")

            # Plot the waveform
            self.plot_waveform()

    def get_metadata(self, file_path):
        """
        Extract metadata from the audio file, including date and time if available.
        """
        metadata = {}
        try:
            if file_path.endswith(".mp3"):
                audio = MP3(file_path)
            elif file_path.endswith(".m4a"):
                audio = MP4(file_path)
            elif file_path.endswith(".flac"):
                audio = FLAC(file_path)
            elif file_path.endswith(".wav"):
                audio = WAVE(file_path)
            else:
                audio = mutagen.File(file_path)

            # Extract the date and time metadata
            if "©day" in audio.tags:  # For .m4a files
                metadata['date'] = audio.tags['©day'][0]
            elif "TDRC" in audio.tags:  # For .mp3 files
                tdrc = audio.tags.get("TDRC")
                if tdrc:
                    metadata['date'] = str(tdrc.text[0])

            # Attempt to parse date-time
            if 'date' in metadata:
                try:
                    metadata['date'] = self.parse_date_time(metadata['date'])
                except ValueError:
                    pass  # If parsing fails, we simply display the raw metadata

        except Exception as e:
            print(f"Error extracting metadata: {e}")

        return metadata

    def parse_date_time(self, date_str):
        """
        Parse the date and time from a string if available.
        Supports formats like YYYY-MM-DD, YYYY-MM-DD HH:MM:SS, etc.
        """
        try:
            # Try to parse date with time
            dt = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            try:
                # Fallback to date-only format
                dt = datetime.datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                # If parsing fails, return the raw string
                return date_str
        return dt.strftime('%Y-%m-%d %H:%M:%S')

    def plot_waveform(self):
        # Plot the audio waveform with downsampled data
        if self.y is not None:
            self.ax.clear()
            
            # Plot every 100th sample for efficiency
            downsample_factor = 100
            times = np.arange(len(self.y))[::downsample_factor] / self.sr
            y_downsampled = self.y[::downsample_factor]
            
            self.ax.plot(times, y_downsampled, label='Audio Waveform (Downsampled)')
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Amplitude')
            self.ax.set_title('Audio Amplitude Over Time')
            self.ax.legend()
            self.canvas.draw()

    def detect_pop_in_chunk(self, chunk, start_idx):
        # Detect pops in a single chunk of audio
        times = np.arange(start_idx, start_idx + len(chunk)) / self.sr
        threshold = self.threshold_scale.get()
        pop_indices = np.where(np.abs(chunk) > threshold)[0]
        return times[pop_indices]

    def detect_pop_sound_parallel(self):
        chunk_duration = 600  # 10-minute chunks
        chunk_size = int(self.sr * chunk_duration)
        total_chunks = len(self.y) // chunk_size

        pop_times = []

        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i in range(total_chunks + 1):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(self.y))
                chunk = self.y[start_idx:end_idx]
                futures.append(executor.submit(self.detect_pop_in_chunk, chunk, start_idx))

            for future in concurrent.futures.as_completed(futures):
                pop_times.extend(future.result())

        # Group pop times by 0.5-second intervals and update the UI
        pop_half_seconds = set(int(t // 0.5) * 0.5 for t in pop_times)
        self.pop_count_label.config(text=f"Detected Pops: {len(pop_half_seconds)}")

        # Optionally, mark the pop sounds on the plot (can be done here if needed)
        self.highlight_pop_sounds(pop_half_seconds)

    def highlight_pop_sounds(self, pop_times):
        # Clear the plot and re-plot with highlights
        if self.y is not None:
            self.ax.clear()

            # Plot every 100th sample for efficiency
            downsample_factor = 100
            times = np.arange(len(self.y))[::downsample_factor] / self.sr
            y_downsampled = self.y[::downsample_factor]

            self.ax.plot(times, y_downsampled, label='Audio Waveform (Downsampled)')
            
            # Mark pop sound times with red dots
            for pop_time in pop_times:
                self.ax.axvline(x=pop_time, color='r', linestyle='--', label='Pop Detected')

            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Amplitude')
            self.ax.set_title('Audio Amplitude Over Time with Pop Sounds Highlighted')
            self.ax.legend()
            self.canvas.draw()

    def toggle_playback(self):
        # Toggle audio playback using pygame mixer
        if self.audio_segment is None:
            return

        if self.is_playing:
            pygame.mixer.music.stop()
            self.is_playing = False
            self.play_button.config(text="Play Audio")
        else:
            if not pygame.mixer.music.get_busy():
                audio_data = io.BytesIO(self.audio_segment.export(format="wav").read())
                pygame.mixer.music.load(audio_data)
                pygame.mixer.music.play()
            self.is_playing = True
            self.play_button.config(text="Stop Audio")

# Create the main window
root = tk.Tk()

# Initialize the application
app = AudioPopDetector(root)

# Run the application
root.mainloop()
