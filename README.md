# VisionGpt AI Assistant

This project is a voice-activated assistant that can play YouTube videos, analyze images, and carry out text-to-speech
and speech-to-text conversions.

## Features

- Voice-activated commands
- Play YouTube videos based on user's voice command
- Analyzes images from the default camera
- Transcribes user's voice to text
- Generates AI responses to user's text
- Speaks out the AI responses

## Dependencies

- Python
- OpenAI
- pywhatkit
- cv2
- noisereduce
- numpy
- sounddevice
- loguru
- pydub
- scipy

## Setup

1. Clone the repository
2. Install the dependencies
3. Set up the OpenAI API key create config.ini file in the root directory and add the following content:

```
[
openai
]
API_KEY=
```

## Usage

Start the conversation manager by running `main.py`. The assistant will start listening for your commands. You can ask
it to play a YouTube video by saying "play" followed by the name of the video. You can also ask it to analyze an image
by saying "analyze image".

## License

This project is licensed under the terms of the MIT license.
