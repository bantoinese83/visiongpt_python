from typing import Literal
from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play
import io


class TextToSpeech:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)

    def generate_speech(self, text: str,
                        voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = "shimmer"):
        # Check if the voice is valid
        if voice not in ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]:
            raise ValueError(f"Invalid voice: {voice}. Please choose from 'alloy', 'echo', 'fable', 'onyx', 'nova', "
                             f"'shimmer'.")

        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text
            )
        except Exception as e:
            # Handle API errors
            raise Exception(f"An error occurred while generating the speech: {str(e)}")

        # Extract the binary content from the response
        audio_data = b""
        for chunk in response.iter_bytes():
            audio_data += chunk

        audio_file = io.BytesIO(audio_data)

        try:
            sound = AudioSegment.from_file(audio_file, format="mp3")
            play(sound)
        except Exception as e:
            # Handle audio playback errors
            raise Exception(f"An error occurred during audio playback: {str(e)}")
