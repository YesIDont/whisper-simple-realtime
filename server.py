import os
import platform
import threading
import time

import numpy
import pyaudio
import pyautogui
import pyperclip
import torch

from transcriber import WhisperModel

last_print = ""

log_timer = time.time()
log_delay = 0

""" Log method with timer funcionality that helps avoid flooding loops with the same message. """
def custom_log(text, use_log_timer = False):
    global last_print, log_timer
    if last_print != text and (not use_log_timer or log_delay == 0 or time.time() - log_timer > log_delay):
        log_timer = time.time()
        print(f"{text}")
        last_print = text

def info(text, use_log_timer = False):
    custom_log(f"[INFO]: {text}", use_log_timer)

def error(text, use_log_timer = False):
    custom_log(f"[ERROR]: {text}", use_log_timer)

class TranscriptionServer:
    def __init__(
            self,
            language = "pl",
            multilingual = True,
            vad_parameters = None,
            model_size =
                "large-v2",
                # "medium",
                # "small",
                # "base",
            debug = False,
            print_in_loop = False,
            print_transcript = True
        ):

        """ Transcription options """
        self.multilingual = multilingual
        self.language = language if self.multilingual and language != None else "en"
        self.vad_parameters = vad_parameters
        self.model_size = model_size
        self.rate = 16000
        self.same_output_occurrence = 0
        self.same_output_threshold = 5
        self.add_pause_thresh = 3
        self.min_sample_duration = 0.25 # 1.0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_index = 1
        self.speach_to_text = WhisperModel(
            self.model_size,
            device_index = self.device_index,
            device=self.device,
            # compute_type="int8" if self.device=="cpu" else "float16",
            compute_type="int8",
            local_files_only=False,
        )

        """ Transcription state """
        self.audio_frames_buffer = None

        """ This is the text from the last received segment that is considered unfinished."""
        self.last_unfinished = ""

        self.prev_out = ""
        self.transcript = []
        self.t_start = None

        """ The value that increases continuously as the application processes audio data. It represents the cumulative amount of audio (in seconds) that has been processed from the start of the audio stream. It's like a marker moving forward as time progresses, indicating how far along the audio stream the application has processed. This value does not decrease; it only grows until the application shuts down or is reset. """
        self.timestamp_offset = 0.0

        """ Grows in specific increments, particularly when parts of the audio buffer are discarded to manage memory and focus on more recent audio data. This offset adjusts to account for the removal of older frames from the buffer, ensuring that the system knows the starting point of the current buffer relative to the original audio stream. It helps in recalibrating the system's understanding of "where" in the original audio stream the current buffer begins, especially after some data has been removed. """
        self.frames_offset = 0.0

        self.clear_transcript = False

        """ Capturing audio stream """
        self.pyaudio_instance = pyaudio.PyAudio()
        self.chunk = 1024
        self.stream = self.pyaudio_instance.open(
            format = pyaudio.paInt16,
            channels = 1,
            rate = self.rate,
            input = True,
            frames_per_buffer = self.chunk
        )

        """ Application settings """
        self.exit_flag = threading.Event()
        self.debug = debug
        self.print_in_loop = print_in_loop
        self.print_transcript = print_transcript
        self.print_first_samples_from_chunk = False

    def clear(self):
        if os.name == "nt":
            os.system("cls")
            return

        os.system("clear")

        return cleared

    # use this to print the transcription right into the input box enywhere in the system
    # ! add option to clear content of the current input box
    def print_to_clipboard(self, text):
        pyperclip.copy(text)

        if platform.system() == "Darwin":
            pyautogui.hotkey("command", "v")
        else:
            pyautogui.hotkey("ctrl", "v")

    def bytes_to_float_array(self, audio_data_bytes):
        raw_data = numpy.frombuffer(audio_data_bytes, dtype=numpy.int16)
        return raw_data.astype(numpy.float32) / 32768.0

    def update_segments(self, segments, sample_duration):
        offset = None

        # process complete segments except for the last one
        if len(segments) > 1:
            for i, s in enumerate(segments[:-1]):
                text_ = s.text
                self.transcript.append(text_)
                offset = min(sample_duration, s.end)

        self.last_unfinished = segments[-1].text

        # if same incomplete segment is seen multiple times then update the offset
        # and append the segment to the list
        if self.last_unfinished.strip() == self.prev_out.strip() and self.last_unfinished != '':
            self.same_output_occurrence += 1
        else:
            self.same_output_occurrence = 0

        if self.same_output_occurrence == self.same_output_threshold:
            if not len(self.transcript) or self.transcript[-1].strip().lower() != self.last_unfinished.strip().lower():
                self.transcript.append(self.last_unfinished)
            self.last_unfinished = ''
            offset = sample_duration
            self.same_output_occurrence = 0
        else:
            self.prev_out = self.last_unfinished

        # update offset
        if offset is not None:
            self.timestamp_offset += offset

    def get_confirmed_transcript(self):
        return ''.join(self.transcript)

    def get_full_transcript(self):
        return ''.join(self.transcript) + self.last_unfinished

    def speech_to_text_thread(self):
        try:
            while not self.exit_flag.is_set():
                if self.clear_transcript:
                    self.clear_transcription()
                    self.clear_transcript = False

                text = self.get_full_transcript()
                if self.print_transcript and text != '':
                    self.clear()
                    print(text)

                # no point moving forward if we have no frames to process
                if self.audio_frames_buffer is None:
                    if self.print_in_loop: info("[LOOP] No frames to process", True)
                    time.sleep(0.01)
                    continue

                num_of_samples_to_skip = int((self.timestamp_offset - self.frames_offset) * self.rate)
                # Check if the current audio chunk exceeds duration of 25 seconds.
                if self.audio_frames_buffer[num_of_samples_to_skip:].shape[0] > 25 * self.rate:
                    # info("[LOOP] Clipping audio chunk as it exceeds 30 seconds", True)

                    # Calculate the total duration of the audio in the buffer in seconds.
                    duration = self.audio_frames_buffer.shape[0] / self.rate

                    # Update the timestamp offset to be 5 seconds less than the total duration of the audio. This effectively "clips" the audio buffer to discard older audio data, keeping the most recent.
                    self.timestamp_offset = self.frames_offset + duration - 5

                # Select a single audio sample from the buffer to process.
                # info("[LOOP] Selecting a single audio sample", True)
                samples_take = max(0, (self.timestamp_offset - self.frames_offset) * self.rate)
                # get sliced audio samples that exceed the sample_take
                input_bytes = self.audio_frames_buffer[int(samples_take):].copy()
                duration = input_bytes.shape[0] / self.rate

                if duration < self.min_sample_duration:
                    if self.print_in_loop: info("[LOOP] sample below min duration", True)
                    time.sleep(0.01)
                    continue

                try:
                    if len(input_bytes) > 0:
                        input_sample = input_bytes.copy()

                        if self.print_in_loop: info("[LOOP] started transcription", True)
                        result, _info = self.speach_to_text.transcribe(
                            input_sample,
                            language = self.language,
                            task = "transcribe",
                            vad_filter = True,
                            vad_parameters = self.vad_parameters,
                            initial_prompt = self.last_unfinished
                        )

                        if len(result) > 0:
                            if self.print_in_loop: info("[LOOP] processing result", True)
                            self.t_start = None
                            self.update_segments(result, duration)
                        else:
                            if self.t_start is None:
                                self.t_start = time.time()

                            # # add a blank if there is no speech for 3 seconds
                            if len(self.transcript) and self.transcript[-1] != '':
                                if time.time() - self.t_start > self.add_pause_thresh:
                                    self.transcript.append('')

                except Exception as e:
                    error(f"[LOOP]: Failed to transcribe audio chunk: {e}")
                    time.sleep(0.01)

        except Exception as e:
             error(f"[LOOP]: {e}")

    def add_frames(self, name_frames):
        if self.audio_frames_buffer is not None:
            # if we have some frames and the total duration of the audio exceeds 45 seconds
            if self.audio_frames_buffer.shape[0] > 45 * self.rate:
                self.frames_offset += 30.0
                # remove all frames before the last 30 seconds
                self.audio_frames_buffer = self.audio_frames_buffer[int(30 * self.rate):]

        else:
            self.audio_frames_buffer = name_frames.copy()
            return

        self.audio_frames_buffer = numpy.concatenate((self.audio_frames_buffer, name_frames), axis=0)

    def receive_audio_stream(self):
        while not self.exit_flag.is_set():
            try:
                audio_bufer = self.stream.read(self.chunk, exception_on_overflow=False)
                frame = self.bytes_to_float_array(audio_bufer)

                # if self.debug and self.print_first_samples_from_chunk:
                #     info(f"Audio samples {len(frame)}: [0]: {frame[0]}, [1]: {frame[1]}, [2]: {frame[2]}", True)
                #     info(f"Number of samples: {len(frame)}", True)

                frame = numpy.clip(frame, -1.0, 1.0)
                self.add_frames(frame)

            except Exception as e:
                error(f"Error while capturing audio stream: {e}")
                break

    def clear_transcription(self):
        self.audio_frames_buffer = None
        self.transcript = []
        self.t_start = None
        self.prev_out = ''
        self.last_unfinished = ''
        self.timestamp_offset = 0.0
        self.frames_offset = 0.0
        self.clear_transcript = False
        info("Cleared current transcription.")

    def run(self):
        try:
            audio_stream_thread = threading.Thread(target=self.receive_audio_stream, daemon=True)
            info(f"Started audio stream thread")

        except Exception as e:
            info(f"Could not initiate websocket thread: {e}")

        try:
            transcription_thread = threading.Thread(target=self.speech_to_text_thread, daemon=True)
            info(f"Started transcription thread:\n    - model: {self.model_size}\n    - device: {self.device}\n    - device index: {self.device_index}")

        except Exception as e:
            info(f"Could not initiate server thread: {e}")

        try:
            audio_stream_thread.start()
            transcription_thread.start()

            # main loop
            while not self.exit_flag.is_set():
                time.sleep(2)

        except KeyboardInterrupt:
            self.exit_flag.set()
            self.server.shutdown()
            self.stream.close()
            self.pyaudio_instance.terminate()
            info("Shutting down server...")

        audio_stream_thread.join()
        transcription_thread.join()
        info("Server shut down successfully.")
