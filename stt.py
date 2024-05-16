from openai import OpenAI
import os


class SpeechToText:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)

    def transcribe_audio(self, audio_file_path):
        # Check if the file exists
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"The file {audio_file_path} does not exist.")

        try:
            with open(audio_file_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
        except Exception as e:
            # Handle API errors
            raise Exception(f"An error occurred while transcribing the audio: {str(e)}")

        # Check if the transcription is empty
        if not transcription.text:
            raise Exception("The transcription is empty.")

        return transcription.text
