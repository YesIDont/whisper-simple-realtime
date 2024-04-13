import time
from server import TranscriptionServer

if __name__ == "__main__":
    transcription = TranscriptionServer(
        debug = True,
        print_transcript = True
    )

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