import threading
import queue 
import numpy as np
import audioop
import math
import logging
import traceback
import sys 
from audio.analysis import analyze_audio_patterns
import config

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, audio_queue):
        self.audio_queue = audio_queue
        self.audio_level = 0.0 
        self.audio_pattern = 0.0
        self.audio_buffers = []
        self.stop_processing = False
        self.processing_thread = None

    def start(self):
        self.processing_thread = threading.Thread(target=self.process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        logger.info("Audio processing thread started")

    def stop(self):
        self.stop_processing = True
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)

    def process_audio(self):
        while not self.stop_processing:
            try: 
                data = self.audio_queue.get(timeout=1)
                logger.debug(f"Received audio data: {len(data)} bytes, queue size: {self.audio_queue.qsize()}")

                # Volume Level
                rms = audioop.rms(data, 2)
                if rms > 0:
                    decibel = 2 * math.log10(rms)
                    normalized_db = max(0, min(1.0, decibel-10) / 40)
                else:
                    normalized_db = 0

                self.audio_level = normalized_db

                 # Store and analyze audio patterns
                self.audio_buffers.append(data)
                max_buffers = int(config.RATE * config.AUDIO_WINDOW / config.CHUNK)
                
                if len(self.audio_buffers) > max_buffers:
                    self.audio_buffers.pop(0)
                    
                if len(self.audio_buffers) > 5:
                    all_samples = np.frombuffer(b''.join(self.audio_buffers), dtype=np.int16)
                    self.audio_pattern = analyze_audio_patterns(all_samples, config.RATE)
                    
                logger.debug(f"Processed audio: level={self.audio_level:.2f}, pattern={self.audio_pattern:.2f}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in audio processing: {e}")
                logger.error(traceback.format_exc())
            finally:
                if not self.audio_queue.empty():
                    self.audio_queue.task_done()

def get_metrics(self):
    return {
        "audio_level": self.audio_level,
        "audio_pattern": self.audio_pattern
    }