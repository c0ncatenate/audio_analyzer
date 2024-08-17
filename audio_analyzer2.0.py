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
            # Load the audio file using librosa for waveform analysis
            self.y, self.sr = librosa.load(self.file_path, sr=None)
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
        # Plot the audio waveform
        if self.y is not None:
            self.ax.clear()
            times = np.arange(len(self.y)) / self.sr
            self.ax.plot(times, self.y, label='Audio Waveform')
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Amplitude')
            self.ax.set_title('Audio Amplitude Over Time')
            self.ax.legend()
            self.canvas.draw()

    def detect_pop_sound(self):
        if self.y is not None:
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

    def toggle_playback(self):
        if self.is_playing:
            self.stop_playback()
        else:
            self.start_playback()

    def start_playback(self):
        if self.audio_segment:
            self.is_playing = True
            self.play_button.config(text="Stop Audio")
            self.sound = pygame.mixer.Sound(self.audio_segment.raw_data)
            self.sound.play(loops=0)

    def stop_playback(self):
        if self.is_playing:
            self.is_playing = False
            self.play_button.config(text="Play Audio")
            pygame.mixer.stop()  # Stop all audio playback

# Create the main window
root = tk.Tk()
app = AudioPopDetector(root)
root.mainloop()
