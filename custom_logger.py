import logging

__all__ = ['logger', 'info', 'warn', 'error']

# Your custom handler
class CountingLogHandler(logging.StreamHandler):
    def __init__(self):
        super().__init__()
        self.last_record = None
        self.repeat_count = 1

    def emit(self, record):
        formatted_message = self.format(record)
        if self.last_record and self.last_record == formatted_message:
            self.repeat_count += 1
            self.stream.seek(0, 2)  # Move to the end of the stream
            self.stream.write(f"\033[F\033[K({self.repeat_count}) {formatted_message}\n")
        else:
            if self.repeat_count > 1:
                self.repeat_count = 1
            self.stream.write(f"{formatted_message}\n")
        self.flush()
        self.last_record = formatted_message

# Configure the root logger
def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set the level you need for the root logger
    # Adding a handler to the root logger
    handler = CountingLogHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

# Instantiating and configuring the logger immediately
logger = setup_logger()

def info(*args):
    logger.info(*args)

def warn(*args):
    logger.warn(*args)

def error(*args):
    logger.error(*args)
