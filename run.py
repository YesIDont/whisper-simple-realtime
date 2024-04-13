import time
from server import TranscriptionServer
from server import TranscriptEvent

if __name__ == "__main__":
    transcription = TranscriptionServer()

    def on_new_transcription(text):
        transcription.clear()
        print(text)

    transcription.add_event_handler(TranscriptEvent.TRANSCRIPTION_CHANGE, on_new_transcription)

    def on_repeated_threshold_exceeded(text):
        transcription.clear()
        print(text)

    transcription.add_event_handler(TranscriptEvent.REPEATED_THRESHOLD, on_repeated_threshold_exceeded)

    transcription.listen()

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