import threading
import time
from threading import Lock

import websocket

from audio_stream_service import AudioStreamService


class MicrophoneStreamingClient(AudioStreamService):
    def __init__(
            self,
            # host="172.17.0.2",
            host="localhost",
            port=9090
        ):

        self.host = host
        self.port = port
        self.reconnect_delay = 2  # Delay before reattempting a connection
        self.reconnect_debounce_delay = 4
        self.reconnect_debounce_start = time.time()
        self.connection_lock = Lock()
        self.websocket_client = None
        # self.exited_with_error = False

        self.add_event_handler(self.send_audio_frames)

    def connect(self):
        if self.websocket_client is not None:
            self.cleanup_websocket()

        with self.connection_lock:
            self.initialize_websocket()

            # Start the WebSocket connection in a new thread outside the locked section
            # print("[INFO]: Starting new websockets thread")
            threading.Thread(target=self.websocket_client.run_forever, daemon=True).start()

    def reconnect(self):
        time_now = time.time()
        if time_now - self.reconnect_debounce_start < self.reconnect_debounce_delay:
            # print("[INFO]: Reconnect attempt debounced.")
            return

        self.reconnect_debounce_start = time_now

        time.sleep(self.reconnect_delay)  # Wait outside the lock to avoid hanging
        print(f"[INFO]: Attempting to reconnect.")
        self.connect()


    def initialize_websocket(self):
        socket_url = f"ws://{self.host}:{self.port}"
        self.websocket_client = websocket.WebSocketApp(
            socket_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )

    def cleanup_websocket(self):
        try:
            self.websocket_client.close()
        except Exception as e:
            print(f"[ERROR]: Error closing WebSocket: {e}")

        self.websocket_client = None

    def on_open(self, ws):
        print("[INFO]: Connected to server")
        ws.send("open_audio_stream")

    def on_message(self, message):
        return

    def on_close(self, ws, close_status_code, close_msg):
        # if self.exited_with_error:
        #     self.exited_with_error = False
        #     print(f"[INFO]: Closed with error: {close_status_code}: {close_msg}")
        #     return

        print(f"[INFO]: Websocket connection closed: {close_status_code}: {close_msg}")
        threading.Thread(target=self.reconnect, daemon=True).start()

    def on_error(self, ws, error):
        print(f"[ERROR]: Websocket connection error: {error}")
        # self.exited_with_error = True
        threading.Thread(target=self.reconnect, daemon=True).start()

    def send_audio_frames(self, frames):
        if self.websocket_client:
            try:
                self.websocket_client.send(frames.tobytes(), opcode=websocket.ABNF.OPCODE_BINARY)

            except Exception as e:
                print(f"[ERROR]: Error sending audio frames: {e}")

