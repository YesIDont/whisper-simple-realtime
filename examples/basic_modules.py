import threading
import time
from src.audio_stream_service import AudioStreamService
from src.transcription_service import TranscriptionService, TranscriptEvent

if __name__ == "__main__":
    transcription = TranscriptionService()
    audio_stream = AudioStreamService()

    def on_new_transcription(text):
        transcription.clear()
        print(text)

    # This event will be fired every time a new transcription is available, i.e. when current transcription is different from the previous one
    transcription.add_event_handler(TranscriptEvent.TRANSCRIPTION_CHANGE, on_new_transcription)

    # This event will be fired if the same transcription is repeated more than n times
    transcription.add_event_handler(TranscriptEvent.REPEATED_THRESHOLD, on_new_transcription)

    # The transcription service requires a audio frames input
    audio_stream.add_event_handler(transcription.add_frames)

    transcription_thread = None
    audio_stream_thread = None

    try:
        audio_stream_thread = threading.Thread(target=audio_stream.audio_stream_loop, daemon=True)

    except Exception as e:
        print(f"[ERROR] Could not initiate transcription thread: {e}")

    try:
        transcription_thread = threading.Thread(target=transcription.transcription_loop, daemon=True)

    except Exception as e:
        print(f"[ERROR] Could not initiate websocket thread: {e}")

    audio_stream_thread.start()
    print(f"[INFO] Started audio stream thread")

    transcription_thread.start()
    print(f"[INFO] Started transcription thread:")
    print(f"- model: {transcription.model_size}")
    print(f"- device: {transcription.device}")
    print(f"- device index: {transcription.device_index}")

    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        transcription.cleanup()
        quit()

    except Exception as e:
        transcription.cleanup()
        print(f"[EXCEPTION] The main loop exited with error: {e}")
        quit()