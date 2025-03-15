import pyaudio 
import queue
import threading
from typing import Callable 
import config

class AudioIO:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.stop_audio = False

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio streaming"""
        self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)
    
    def start_stream(self):
        self.stream = self.p.open(
            format=config.FORMAT,
            channels=config.CHANNELS,
            rate=config.RATE,
            input=True,
            frames_per_buffer=config.CHUNK,
            stream_callback=self.audio_callback
        )

        self.stream.start_stream()

    def stop_stream(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        self.stop_audio=True

    