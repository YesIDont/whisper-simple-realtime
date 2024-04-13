import os
import platform
import re as regex
import threading
import time
from typing import List

import numpy
import pyaudio
import pyautogui
import pyperclip
import torch

from transcriber import WhisperModel

last_print = ""

log_timer = time.time()
log_delay = 0.1

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

class Event:
    def __init__(self, category, callback):
        self.category = category
        self.callback = callback

class TranscriptionServer:
    def __init__(
            self,
            language = "pl",
            multilingual = True,
            transcription_task = "transcribe",
            vad_parameters = None,
            model_size =
                "large-v2",
                # "medium",
                # "small",
                # "base",
            debug = False,
            print_in_loop = False,
            print_transcript = False,
            clear_before_transcript_print = False
        ):

        """ Transcription options """
        self.multilingual = multilingual
        self.transcription_task = transcription_task
        self.language = language if self.multilingual and language != None else "en"
        self.vad_parameters = vad_parameters
        self.model_size = model_size
        self.rate = 16000
        self.same_output_occurrence = 0
        self.same_output_threshold = 3
        self.speach_silence_threshold = 1
        self.min_sample_duration = 0.25 # 1.0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_index = 1
        self.speach_to_text = WhisperModel(
            self.model_size,
            device_index = self.device_index,
            device=self.device,
            compute_type="float16" if self.device=="cuda" else "int8",
            local_files_only=False,
        )
        # phrases that will cause the segment to be dropped (not used in the final transcript)
        self.excluded_phrases = [
            "PARROT TV",
            "Thank you for your support on Patronite.",
            "Zapraszam na kolejny odcinek!",
            "Napisy stworzone przez społeczność Amara.org"
            "Amara.org",
            "Dzięki za uwagę",
            "Dziękuję za uwagę",
            "Produkcja Polskie Towarzystwo Astronomiczne Telewizja Polska",
            "Nie zapomnijcie zasubskrybować oraz zafollowować mnie na Facebooku!"
        ]

        """ Transcription state """
        self._reset_on_next_tick = False
        self.audio_frames_buffer = None

        """ This is the text from the last received segment that is considered unfinished."""
        self.last_unfinished = ""

        self.prev_out = ""
        self.transcript = []
        self.t_start = None

        """ The value that increases continuously as the application processes audio data.
        It represents the cumulative amount of audio (in seconds) that has been processed from
        the start of the audio stream. It's like a marker moving forward as time progresses,
        indicating how far along the audio stream the application has processed.
        This value does not decrease; it only grows until the application shuts down or is reset. """
        self.timestamp_offset = 0.0

        """ Grows in specific increments, particularly when parts of the audio buffer are discarded
        to manage memory and focus on more recent audio data. This offset adjusts to account for the
        removal of older frames from the buffer, ensuring that the system knows the starting point
        of the current buffer relative to the original audio stream. It helps in recalibrating
        the system's understanding of "where" in the original audio stream the current buffer begins,
        especially after some data has been removed. """
        self.frames_offset = 0.0

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
        self.is_paused: bool = False
        self.debug: bool = debug
        self.print_in_loop: bool = print_in_loop
        self.print_transcript: bool = print_transcript
        self.clear_before_transcript_print: bool = clear_before_transcript_print
        self.print_first_samples_from_chunk: bool = False

        self.audio_stream_thread = None
        self.transcription_thread = None

        self.events: List[Event] = []

        # self.on_speach_silence_detected_events = []
        # self.on_repeated_threshold_reached_events = []

    def add_event_handler(self, category, callback):
        # event fired when silence is detected after speach
        if category == 'silenceafter':
            self.events.append(Event('silenceafter', callback))
            return

        # event fired when the same output is detected given n times
        if category == 'repeatedthreshold':
            self.events.append(Event('repeatedthreshold', callback))
            return

    def get_events_by_category(self, category):
        return [event.callback for event in self.events if event.category == category]

    def clear(self):
        if os.name == "nt":
            os.system("cls")
            return

        os.system("clear")

        return cleared

    def check_for_excluded_phrases(self, text):
        clean_new_text = regex.sub(r'[^a-zA-Z]', '', text, flags=regex.UNICODE).lower()
        for phrase in self.excluded_phrases:
            clean_phrase = regex.sub(r'[^a-zA-Z]', '', phrase, flags=regex.UNICODE).lower()
            # if self.debug: info(f"Checking for excluded phrase: {clean_new_text} in {clean_phrase}")
            if phrase.strip() in text.strip() or clean_phrase in clean_new_text:
                return True

        return False

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

    def cleanup(self):
        self.exit_flag.set()
        self.stream.close()
        self.pyaudio_instance.terminate()
        if self.audio_stream_thread is not None: self.audio_stream_thread.join()
        if self.transcription_thread is not None: self.transcription_thread.join()

    def process_new_segments(self, segments, sample_duration):
        offset = None
        reached_repeat_threshold = False

        filtered_segments = []
        for segment in segments:
            # if self.debug: info(f"New segment: {segment.text}")
            if not self.check_for_excluded_phrases(segment.text):
                filtered_segments.append(segment)
            else:
                info(f"Excluded phrase detected: {segment.text}")

        if len(filtered_segments) == 0:
            return reached_repeat_threshold

        # process complete segments except for the last one
        if len(filtered_segments) > 1:
            for i, s in enumerate(filtered_segments[:-1]):
                text_ = s.text
                self.transcript.append(text_)
                offset = min(sample_duration, s.end)

        self.last_unfinished = filtered_segments[-1].text

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
            reached_repeat_threshold = True

        else:
            self.prev_out = self.last_unfinished

        # update offset
        if offset is not None:
            self.timestamp_offset += offset

        return reached_repeat_threshold

    def get_confirmed_transcript(self):
        return ''.join(self.transcript)

    def get_full_transcript(self):
        # if self.debug and len(self.transcript) > 0: info(f"Transcript: {self.transcript}")
        # if self.debug and self.last_unfinished != '': info(f"Last unfinished: {self.last_unfinished}")
        return ''.join(self.transcript) + self.last_unfinished

    def transcribe(self, audio_data):
        try:
            result, _info = self.speach_to_text.transcribe(
                audio_data,
                language = self.language,
                task = self.transcription_task,
                vad_filter = True,
                vad_parameters = self.vad_parameters,
                initial_prompt = self.last_unfinished
            )
            return result
        except Exception as e:
            if self.debug: error(f"Failed to transcribe audio chunk: {e}")

    def speech_to_text_thread(self):
        try:
            while not self.exit_flag.is_set():
                if self.is_paused:
                    time.sleep(0.01)
                    continue

                if self._reset_on_next_tick:
                    self._reset_state()
                    self._reset_on_next_tick = False
                    continue

                current = self.get_confirmed_transcript()
                if self.debug and self.print_transcript and current != '':
                    if self.clear_before_transcript_print: self.clear()
                    info(f'[TRANSCRIPT]: {current}')

                # no point moving forward if we have no frames to process
                if self.audio_frames_buffer is None:
                    if self.debug and self.print_in_loop: info("[LOOP] No frames to process", True)
                    time.sleep(0.01)
                    continue

                num_of_samples_to_skip = int((self.timestamp_offset - self.frames_offset) * self.rate)
                # Check if the current audio chunk exceeds duration of 25 seconds.
                if self.audio_frames_buffer[num_of_samples_to_skip:].shape[0] > 25 * self.rate:
                    # if self.debug and self.print_in_loop: info("[LOOP] Clipping audio chunk as it exceeds 30 seconds", True)

                    # Calculate the total duration of the audio in the buffer in seconds.
                    duration = self.audio_frames_buffer.shape[0] / self.rate

                    # Update the timestamp offset to be 5 seconds less than the total duration of the audio.
                    # This effectively "clips" the audio buffer to discard older audio data, keeping the most recent.
                    self.timestamp_offset = self.frames_offset + duration - 5

                # Select a single audio sample from the buffer to process.
                # if self.debug and self.print_in_loop: info("[LOOP] Selecting a single audio sample", True)
                samples_take = max(0, (self.timestamp_offset - self.frames_offset) * self.rate)
                # get sliced audio samples that exceed the sample_take
                input_bytes = self.audio_frames_buffer[int(samples_take):].copy()
                duration = input_bytes.shape[0] / self.rate

                if duration < self.min_sample_duration:
                    if self.debug and self.print_in_loop: info("[LOOP] sample below min duration", True)
                    time.sleep(0.01)
                    continue

                try:
                    if len(input_bytes) > 0:
                        input_sample = input_bytes.copy()

                        if self.debug and self.print_in_loop: info("[LOOP] started transcription", True)
                        result = self.transcribe(input_sample)

                        if len(result) > 0:
                            if self.debug and self.print_in_loop: info("[LOOP] processing result", True)
                            self.t_start = None
                            reached_repeat_threshold = self.process_new_segments(result, duration)

                            if reached_repeat_threshold:
                                if self.debug and self.print_in_loop: info("[EVENT]: reached repeat threshold")
                                events = self.get_events_by_category('repeatedthreshold')
                                if len(events):
                                    text = self.get_full_transcript()
                                    info(f"Transcript on reached repeat threshold: {text}")
                                    reset_state = False
                                    for event in events:
                                        should_reset = event(text)
                                        if should_reset: reset_state = True
                                    if reset_state: self._reset_state()

                        else:
                            if self.t_start is None:
                                self.t_start = time.time()

                            # add a blank if there is no speech for n seconds
                            if len(self.transcript) and self.transcript[-1] != '':
                                if time.time() - self.t_start > self.speach_silence_threshold:
                                    if self.debug and self.print_in_loop:
                                        info("[LOOP] Detected silence in audio input for {self.speach_silence_threshold} seconds")
                                    self.transcript.append('')

                                    events = self.get_events_by_category('silenceafter')
                                    if len(events):
                                        if self.debug and self.print_in_loop: info("[EVENT]: silence after speach detected")
                                        text = self.get_full_transcript()
                                        reset_state = False
                                        for event in events:
                                            should_reset = event(text)
                                            if should_reset: reset_state = True
                                        if reset_state: self._reset_state()

                except Exception as e:
                    if self.debug: error(f"[LOOP]: Failed to transcribe audio chunk: {e}")
                    time.sleep(0.01)

        except Exception as e:
             if self.debug: error(f"[LOOP]: {e}")

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
                if self.is_paused:
                    time.sleep(0.01)
                    continue

                audio_bufer = self.stream.read(self.chunk, exception_on_overflow=False)
                frame = self.bytes_to_float_array(audio_bufer)

                if self.debug and self.print_first_samples_from_chunk:
                    info(f"Audio samples {len(frame)}: [0]: {frame[0]}, [1]: {frame[1]}, [2]: {frame[2]}", True)
                    info(f"Number of samples: {len(frame)}", True)

                frame = numpy.clip(frame, -1.0, 1.0)
                self.add_frames(frame)

            except Exception as e:
                if self.debug: error(f"Error while capturing audio stream: {e}")
                break

    def _reset_state(self):
        self.transcript = []
        self.last_unfinished = ''
        self.prev_out = ''
        self.t_start = None
        self.timestamp_offset = 0.0
        self.audio_frames_buffer = None
        self.frames_offset = 0.0
        if self.debug: info("Reset to default state.")

    def reset_state(self):
        """ Reseting state from outside of the class should be done only at the specyfied point in the loop,
        othewise the cleaned data could be removed in the middle of the operation on that data. """
        self._reset_on_next_tick = True

    def pause(self):
        if self.debug: info("Pausing transcription.")
        self.is_paused = True

    def resume(self):
        if self.debug: info("Resuming transcription.")
        self.is_paused = False

    def listen(self, start_paused = False):
        self.is_paused = start_paused

        try:
            self.audio_stream_thread = threading.Thread(target=self.receive_audio_stream, daemon=True)
            if self.debug: info(f"Started audio stream thread")

        except Exception as e:
            if self.debug: error(f"Could not initiate websocket thread: {e}")

        try:
            self.transcription_thread = threading.Thread(target=self.speech_to_text_thread, daemon=True)
            if self.debug: info(f"Started transcription thread:\n    - model: {self.model_size}\n    - device: {self.device}\n    - device index: {self.device_index}")

        except Exception as e:
            if self.debug: error(f"Could not initiate server thread: {e}")

        self.audio_stream_thread.start()
        self.transcription_thread.start()

