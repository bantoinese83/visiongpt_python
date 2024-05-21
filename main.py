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


class ConversationManager:
    FS = 44100  # Sample rate
    DURATION = 10  # Duration of recording
    THRESHOLD = 0.02  # Audio volume threshold for starting the recording
    ENERGY_THRESHOLD = 0.1  # Energy threshold for VAD
    ENERGY_DECAY = 0.99  # Decay factor for energy calculation
    ENERGY_THRESHOLD_MULTIPLIER = 1.5  # Multiplier for adjusting energy threshold

    def __init__(self, user_name, bot_name, api_key):
        self.user_name = user_name
        self.bot_name = bot_name
        self.api_key = api_key
        self.tts_service = TextToSpeech(api_key)
        self.stt_service = SpeechToText(api_key)
        self.text_generator = TextGenerator(api_key)
        self.image_analyzer = ImageAnalyzer(api_key)
        self.stop_event = threading.Event()
        self.current_hour = datetime.datetime.now().hour
        self.last_ai_response = ""
        self.audio_data = None
        self.average_volume = 0.02
        self.adapt_factor = 0.9
        self.energy_threshold = self.ENERGY_THRESHOLD
        self.audio_queue = queue.Queue()
        self.camera_opened = False
        self.cap = None
        self.cap_lock = threading.Lock()
        self.recording = False  # New state variable to track recording state

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

        startup_sound_thread = threading.Thread(target=self.play_sound, args=("assets/startup_sound.mp3",))
        startup_sound_thread.start()
        startup_sound_thread.join()

        bot_name_thread = threading.Thread(target=self.print_and_speak, args=(self.bot_name,))
        bot_name_thread.start()
        bot_name_thread.join()

        self.print_and_speak(greeting)
        logger.info("Conversation started.")

        interrupt_thread = threading.Thread(target=self.listen_for_interrupt)
        audio_thread = threading.Thread(target=self.record_audio)
        processing_thread = threading.Thread(target=self.process_audio)

        interrupt_thread.start()
        audio_thread.start()
        processing_thread.start()

        interrupt_thread.join()
        audio_thread.join()
        processing_thread.join()

        self.stop_event.wait()
        logger.info("Conversation ended.")

    def listen_for_interrupt(self):
        while not self.stop_event.is_set():
            user_input = input().lower()
            if user_input in ['stop', 'exit', 'quit', 'end']:
                logger.info("User interrupted the conversation")
                self.stop_event.set()
                if self.camera_opened:
                    self.close_camera()

    def process_audio(self):
        while not self.stop_event.is_set():
            try:
                audio_data = self.audio_queue.get()
                if audio_data.size > 0:
                    self.audio_data = audio_data
                    energy = self.calculate_energy(audio_data)
                    self.adjust_energy_threshold(energy)
                    if energy > self.energy_threshold:
                        audio_data = self.reduce_noise(audio_data)
                        write("user_audio.wav", self.FS, audio_data)
                        user_text = self.stt_service.transcribe_audio("user_audio.wav")
                        if user_text:
                            print(f"User: {user_text}")
                            if user_text.lower() == "camera":
                                self.analyze_image()
                            elif user_text.lower() == "open camera":
                                self.open_camera()
                            elif user_text.lower() == "close camera":
                                self.close_camera()
                            else:
                                self.generate_and_speak_response(user_text)
                        else:
                            print("Sorry, I couldn't understand what you said. Please try again.")
                            self.print_and_speak("Sorry, I couldn't understand what you said. Please try again.")
                    else:
                        print("Silence detected.")
            except Exception as e:
                print(f"An error occurred while processing audio: {str(e)}")
                break

    @staticmethod
    def play_sound(sound_file):
        sound = AudioSegment.from_file(sound_file)
        play(sound)

    def record_audio(self):
        while not self.stop_event.is_set():
            try:
                if not self.recording:
                    if self.is_sound_detected():
                        print("Sound detected, start recording...")
                        self.play_sound("assets/rec_start_stop_3.mp3")
                        self.recording = True

                user_voice = sd.rec(int(self.DURATION * self.FS), samplerate=self.FS, channels=1, dtype='float64')
                sd.wait()
                if user_voice.size > 0:
                    if not self.stop_event.is_set():
                        self.audio_queue.put(user_voice)
                        self.update_average_volume(user_voice)
                        if self.is_sound_detected() and not self.recording:
                            print("Sound detected, start recording...")
                            self.play_sound("assets/rec_start_stop_3.mp3")
                            self.recording = True
                        elif not self.is_sound_detected() and self.recording:
                            print("Silence detected, stop recording...")
                            self.play_sound("assets/rec_start_stop_2.mp3")
                            self.recording = False
                            time.sleep(1)
                else:
                    print("No audio data detected.")
            except Exception as e:
                print(f"An error occurred while recording audio: {str(e)}")
                break

    def open_camera(self):
        with self.cap_lock:
            if not self.camera_opened:
                self.cap = cv2.VideoCapture(0)
                if self.cap is not None and self.cap.isOpened():
                    self.camera_opened = True
                    self.print_and_speak("Camera is now open.")
                else:
                    self.print_and_speak("Failed to open the camera.")

    def close_camera(self):
        with self.cap_lock:
            if self.camera_opened and self.cap is not None:
                self.cap.release()
                self.cap = None
                self.camera_opened = False
                self.print_and_speak("Camera is now closed.")

    def analyze_image(self):
        with self.cap_lock:
            if self.camera_opened and self.cap is not None:
                ret, frame = self.cap.read()
                if ret:
                    image_path = "camera_capture.png"
                    cv2.imwrite(image_path, frame)
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
            self.print_and_speak("Let me think for a moment.")
            ai_response = self.text_generator.generate_text(user_text)
            if ai_response:
                print(f"AI Response: {ai_response}")
                self.print_and_speak(ai_response)
                self.last_ai_response = ai_response
            else:
                print("Sorry, I couldn't find a response right now. Please wait.")
                self.print_and_speak("Sorry, just a moment. I'm still thinking.")
                self.print_and_speak("Thank you for waiting. Is there anything else I can assist you with?")
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
            return volume > self.average_volume
        return False

    @staticmethod
    def reduce_noise(audio_data):
        return nr.reduce_noise(y=audio_data.flatten(), sr=ConversationManager.FS).reshape(audio_data.shape)

    def update_average_volume(self, audio_data):
        current_volume = np.linalg.norm(audio_data) * 10
        self.average_volume = (self.adapt_factor * self.average_volume) + ((1 - self.adapt_factor) * current_volume)
        print(f"Average volume: {self.average_volume}")

    @staticmethod
    def calculate_energy(audio_data):
        pre_emphasis = 0.97
        emphasized_signal = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])
        energy = np.sum(np.abs(emphasized_signal) ** 2)
        return energy

    def adjust_energy_threshold(self, energy):
        if energy is not None and energy > self.energy_threshold:
            self.energy_threshold *= self.ENERGY_DECAY
        else:
            self.energy_threshold *= self.ENERGY_THRESHOLD_MULTIPLIER

    @staticmethod
    def run_in_thread(target, args=()):
        thread = threading.Thread(target=target, args=args, daemon=True)
        thread.start()
        return thread


if __name__ == "__main__":
    USER_NAME = "Bryan"
    BOT_NAME = "AXEL"
    API_KEY = ''

    conversation_manager = ConversationManager(USER_NAME, BOT_NAME, API_KEY)
    conversation_manager.start_conversation()
