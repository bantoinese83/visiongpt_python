import datetime
import os
import queue
import threading
import time
import pywhatkit

import cv2
import noisereduce as nr
import numpy as np
import sounddevice as sd
from loguru import logger
from pydub import AudioSegment
from pydub.playback import play
from scipy.io.wavfile import write

from ia import ImageAnalyzer
from stt import SpeechToText
from tg import TextGenerator
from tts import TextToSpeech
from logging_config import setup_logger

setup_logger()

# Constants
USER_NAME = "Bryan"
BOT_NAME = "AXEL"
FS = 44100  # Sample rate
DURATION = 10  # Duration of recording
THRESHOLD = 0.02  # Audio volume threshold for starting the recording
ENERGY_THRESHOLD = 0.1  # Energy threshold for VAD
ENERGY_DECAY = 0.99  # Decay factor for energy calculation
ENERGY_THRESHOLD_MULTIPLIER = 1.5  # Multiplier for adjusting energy threshold

# API Key Constant
API_KEY = ''

# Global variables
AUDIO_QUEUE = queue.Queue()
CAMERA_OPENED = False
cap = None

class ConversationManager:
    def __init__(self, user_name, bot_name, api_key):
        self.user_name = user_name
        self.bot_name = bot_name
        self.api_key = api_key
        self.tts_service = TextToSpeech(api_key)
        self.stt_service = SpeechToText(api_key)
        self.text_generator = TextGenerator(api_key)
        self.image_analyzer = None
        self.stop_event = threading.Event()
        self.current_hour = datetime.datetime.now().hour
        self.last_ai_response = ""
        self.audio_data = None
        self.average_volume = 0.02
        self.adapt_factor = 0.9
        self.energy_threshold = ENERGY_THRESHOLD

    def get_greeting(self):
        if 5 <= self.current_hour < 7:
            return f"Early bird! Good morning, {self.user_name}. Let's seize the day!"
        elif 7 <= self.current_hour < 12:
            return f"Good morning, {self.user_name}! The day is young and full of potential."
        elif 12 <= self.current_hour < 14:
            return f"Good afternoon, {self.user_name}! Time for a lunch break, perhaps?"
        elif 14 <= self.current_hour < 17:
            return f"Good afternoon, {self.user_name}! Let's continue making progress."
        elif 17 <= self.current_hour < 20:
            return f"Good evening, {self.user_name}! How can I assist you in wrapping up your day?"
        elif 20 <= self.current_hour < 23:
            return f"Good night, {self.user_name}! Time to wrap up for the day."
        else:
            return f"Hello, {self.user_name}! Working late, I see. Remember, rest is important too!"

    def start_conversation(self):
        greeting = self.get_greeting()
        startup_sound_thread = threading.Thread(target=self.play_sound, args=("assets/startup_sound.mp3",), daemon=True)
        bot_name_thread = threading.Thread(target=self.print_and_speak, args=(self.bot_name,), daemon=True)

        startup_sound_thread.start()
        bot_name_thread.start()

        startup_sound_thread.join()
        bot_name_thread.join()

        self.print_and_speak(greeting)
        logger.info("Conversation started.")

        interrupt_thread = threading.Thread(target=self.listen_for_interrupt, daemon=True)
        audio_thread = threading.Thread(target=self.record_audio, daemon=True)
        processing_thread = threading.Thread(target=self.process_audio, daemon=True)

        interrupt_thread.start()
        audio_thread.start()
        processing_thread.start()

        interrupt_thread.join()
        audio_thread.join()
        processing_thread.join()

        logger.info("Conversation ended.")

    def listen_for_interrupt(self):
        while not self.stop_event.is_set():
            user_input = input().lower()
            if user_input in ['stop', 'exit', 'quit', 'end']:
                logger.info("User interrupted the conversation")
                self.stop_event.set()
                if CAMERA_OPENED:
                    self.close_camera()

    def process_audio(self):
        lock = threading.Lock()
        energy = None
        user_voice = None  # Initialize user_voice to None
        skip_response = False  # Add a flag to skip the AI response
        while not self.stop_event.is_set():
            with lock:
                if not AUDIO_QUEUE.empty():
                    print("AI is listening...")
                    user_voice = AUDIO_QUEUE.get()
                    user_voice = self.reduce_noise(user_voice)
                    audio_file = "user_voice.wav"
                    write(audio_file, FS, user_voice)
                    try:
                        user_text = self.stt_service.transcribe_audio(audio_file)
                        if user_text == self.last_ai_response:
                            continue
                        print(f"User said: {user_text}")
                    except Exception as transcribe_error:
                        logger.error(f"Error in transcription: {transcribe_error}")
                        continue

                    if "play" in user_text.lower():
                        search_query = user_text.replace("play", "").strip()
                        pywhatkit.playonyt(search_query)
                        self.print_and_speak(f"Playing {search_query} on YouTube.")
                        skip_response = True  # Set the flag to skip the AI response

                    if "camera" in user_text.lower():
                        self.open_camera()
                    elif "analyze image" in user_text.lower():
                        self.analyze_image()
                    elif not skip_response:  # Check the flag before generating an AI response
                        self.generate_and_speak_response(user_text)

                    skip_response = False  # Reset the flag for the next command

                if user_voice is not None:
                    energy = self.calculate_energy(user_voice)
                    if energy is not None and energy > self.energy_threshold:
                        self.energy_threshold *= ENERGY_DECAY
                    else:
                        self.energy_threshold *= ENERGY_THRESHOLD_MULTIPLIER

                AUDIO_QUEUE.queue.clear()

                if energy is not None and energy > self.energy_threshold:
                    time.sleep(1)
                else:
                    time.sleep(0.1)

    @staticmethod
    def play_sound(sound_file):
        sound = AudioSegment.from_file(sound_file)
        play(sound)

    def record_audio(self):
        while not self.stop_event.is_set():
            try:
                print("Sound detected, start recording...")
                self.play_sound("assets/rec_start_stop_3.mp3")  # Play sound when recording starts
                user_voice = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='float64')
                sd.wait()
                if user_voice.size > 0:
                    if not self.stop_event.is_set():
                        AUDIO_QUEUE.put(user_voice)
                        self.update_average_volume(user_voice)
                        if self.is_sound_detected():
                            print("Sound detected, continue recording...")
                            self.play_sound("assets/rec_start_stop_1.mp3")  # Play sound when recording continues
                        else:
                            print("Silence detected, stop recording...")
                            self.play_sound("assets/rec_start_stop_2.mp3")  # Play sound when recording stops
                            time.sleep(1)
                else:
                    print("No audio data detected.")
            except Exception as e:
                print(f"An error occurred while recording audio: {str(e)}")
                break

    def open_camera(self):
        global CAMERA_OPENED, cap
        if not CAMERA_OPENED:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                CAMERA_OPENED = True
                self.print_and_speak("Camera is now open.")
            else:
                self.print_and_speak("Failed to open the camera.")

    def close_camera(self):
        global CAMERA_OPENED, cap
        if CAMERA_OPENED and cap is not None:
            cap.release()
            CAMERA_OPENED = False
            self.print_and_speak("Camera is now closed.")

    def analyze_image(self):
        global CAMERA_OPENED, cap
        if CAMERA_OPENED and cap is not None:
            ret, frame = cap.read()
            if ret:
                image_path = "camera_capture.png"
                cv2.imwrite(image_path, frame)
                if self.image_analyzer is None:
                    self.image_analyzer = ImageAnalyzer(self.api_key)
                analysis_results = self.image_analyzer.analyze_image(image_path)
                print(f"Analysis results: {analysis_results}")
                os.remove(image_path)
            else:
                self.print_and_speak("Failed to capture an image.")
        else:
            self.print_and_speak("Camera is not open. Please say 'camera' to open the camera first.")

    def generate_and_speak_response(self, user_text):
        try:
            print("AI is thinking...")
            ai_response = self.text_generator.generate_text(user_text)
            print(f"AI Response: {ai_response}")
            self.print_and_speak(ai_response)
            self.last_ai_response = ai_response
            time.sleep(1)
        except Exception as generate_error:
            logger.error(f"Error in text generation: {generate_error}")

   


    def print_and_speak(self, message):
        print(message)
        try:
            self.tts_service.generate_speech(message)
        except Exception as error:
            logger.error(f"Error while speaking: {error}")

    def is_sound_detected(self):
        if self.audio_data is not None:
            volume = np.linalg.norm(self.audio_data) * 10
            return volume > self.average_volume  # Use average_volume as the threshold
        return False

    @staticmethod
    def reduce_noise(audio_data):
        return nr.reduce_noise(y=audio_data.flatten(), sr=FS).reshape(audio_data.shape)

    def update_average_volume(self, audio_data):
        current_volume = np.linalg.norm(audio_data) * 10
        self.average_volume = (self.adapt_factor * self.average_volume) + ((1 - self.adapt_factor) * current_volume)
        print(f"Average volume: {self.average_volume}")

    @staticmethod
    def calculate_energy(audio_data):
        # Apply pre-emphasis filter
        pre_emphasis = 0.97
        emphasized_signal = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])

        # Calculate energy
        energy = np.sum(np.abs(emphasized_signal) ** 2)
        return energy


if __name__ == "__main__":
    conversation_manager = ConversationManager(USER_NAME, BOT_NAME, API_KEY)
    conversation_manager.start_conversation()
