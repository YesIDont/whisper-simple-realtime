# Whisper Simple Realtime

Based on heavily reduced ([Whisper Live](https://github.com/collabora/WhisperLive)) and ([Faster Whisper](https://github.com/SYSTRAN/faster-whisper)) WSR (Whisper Simple Realtime) gives the base for creating realtime transcription with Whisper.ai.

The implementation is very simple and lacks a lot of features on purpose, its goal is to provide crude base for more complex work.

# Installation

1. As a good practice its best to create local environment, e.g:

   `conda create -p ./.conda python=3.9`

   The maximmum working tested version of Python is 3.9

2. Activate it with:

   `conda activate ./.conda`

3. Install requirements:

   `pip install -r requirements.txt`

# Usage

After installation you can use the run.py file:

`python run.py`

This implementation will just print the continuous transcription from operating system's default audio input to the console. Anything else is for you to implement.

# Create own requirements.txt

In case you would like to add some packages remember to udpate the requirements.txt file:

`conda list --export > requirements.txt`
