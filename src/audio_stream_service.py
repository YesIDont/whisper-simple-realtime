import threading
import time
import numpy
import pyaudio

""" Simple microphone audio stream service"""
class AudioStreamService():
    def __init__(self, debug = False):
        """ Capturing audio stream """
        self.pyaudio_instance = pyaudio.PyAudio()
        self.chunk = 1024
        self.rate = 16000
        self.stream = self.pyaudio_instance.open(
            format = pyaudio.paInt16,
            channels = 1,
            rate = self.rate,
            input = True,
            frames_per_buffer = self.chunk
        )
        self.audio_frames_buffer = None

        self.debug = debug
        self.is_paused = False
        self.print_first_samples_from_chunk = False
        self.exit_flag = threading.Event()

        self.event_handlers = []

    def add_event_handler(self, callback):
        self.event_handlers.append(callback)

    def bytes_to_float_array(self, audio_data_bytes):
        raw_data = numpy.frombuffer(audio_data_bytes, dtype=numpy.int16)
        return raw_data.astype(numpy.float32) / 32768.0

    def audio_stream_loop(self):
        if len(self.event_handlers) < 1:
            if self.debug: print("[WARNING] No event handlers registered")
            return

        while not self.exit_flag.is_set():
            try:
                if self.is_paused:
                    time.sleep(0.01)
                    continue

                audio_bufer = self.stream.read(self.chunk, exception_on_overflow=False)
                frame = self.bytes_to_float_array(audio_bufer)

                if self.debug and self.print_first_samples_from_chunk:
                    print(f"[INFO] Audio samples {len(frame)}: [0]: {frame[0]}, [1]: {frame[1]}, [2]: {frame[2]}")
                    print(f"[INFO] Number of samples: {len(frame)}")

                frame = numpy.clip(frame, -1.0, 1.0)

                for event in self.event_handlers:
                    event(frame)

            except Exception as e:
                if self.debug: print(f"[ERROR] Error while capturing audio stream: {e}")
                break

    def cleanup(self):
        self.exit_flag.set()
        self.stream.close()
        self.pyaudio_instance.terminate()