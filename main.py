#HuggingFace Microsoft Model library requirements
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
from datasets import load_dataset
from pydub import AudioSegment

#Google TTS Model Reqs
from gtts import gTTS
#from pydub import AudioSegment <-- duplicated, see above.

#library requirements for the noise reduction
import noisereduce as nr
import librosa
import soundfile as sf

#For API things
from dotenv import load_dotenv
import os
import threading 


import speech_recognition as sr
from googletrans import Translator

trans = Translator()
r = sr.Recognizer()

#-----------------------------------------------------------
def record_speech():
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Prompt user to start recording
    print("Press 's' to start recording and 'x' to stop recording.")

    # Capture audio from the microphone
    with sr.Microphone() as source:
        audio = None
        while True:
            user_input = input()
            if user_input.lower() == 's':
                print("Recording...")
                recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
                audio = recognizer.listen(source)  # Listen for speech
            elif user_input.lower() == 'x':
                if audio:
                    print("Recording stopped.")
                    break
                else:
                    print("You haven't started recording yet. Press 's' to start recording.")
            else:
                print("Invalid input. Press 's' to start recording and 'x' to stop recording.")

    if audio:
        try:
            print("Recognizing...")  # Indicate that it's recognizing speech
            text = recognizer.recognize_google(audio)  # Recognize speech using Google Speech Recognition
            print("You said:", text)  # Print the recognized text
            return text
        except sr.UnknownValueError:
            print("Sorry, I could not understand what you said.")  # Handle unrecognized speech
            return None
        except sr.RequestError as e:
            print("Sorry, an error occurred with the speech recognition service:", str(e))  # Handle errors with the speech recognition service
            return None
    else:
        return None

#-----------------------------------------------------------

def translate_to_chinese_traditional(text):
    translator = Translator()
    translated_text = translator.translate(text, src='en', dest='zh-tw')
    return translated_text.text

#-----------------------------------------------------------

def text_to_speech(text, lang='zh-tw'):
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save("output.mp3")  # Save the speech as an MP3 file
    os.system("mpg321 output.mp3")  # Play the MP3 file using mpg321 (Linux) or other suitable player

#-----------------------------------------------------------
def speech_to_english():
    # Initialize recognizer
    recognizer = sr.Recognizer()
    
    # Prompt user to start speaking
    print("Press 's' to start speaking in Chinese (traditional) and 'x' to stop.")

    # Flag to control the listening state
    listening = False
    
    # Capture audio from the microphone
    with sr.Microphone() as source:
        while True:
            user_input = input()
            if user_input.lower() == 's':
                if not listening:
                    print("Start speaking...")
                    listening = True
                    audio = recognizer.listen(source)  # Listen for speech
            elif user_input.lower() == 'x':
                if listening:
                    print("Stopped listening.")
                    break
                else:
                    print("You haven't started speaking yet. Press 's' to start speaking.")
            else:
                print("Invalid input. Press 's' to start speaking and 'x' to stop.")

    if audio:
        try:
            print("Transcribing...")  # Indicate that it's transcribing speech
            # Recognize speech using Google Speech Recognition
            recognized_text = recognizer.recognize_google(audio, language='zh-TW')
            print("You said (in Chinese):", recognized_text)  # Print the recognized text

            # Translate the recognized text to English
            translator = Translator()
            translated_text = translator.translate(recognized_text, src='zh-TW', dest='en').text
            print("Translated text (in English):", translated_text)  # Print the translated text
            
            return translated_text
        
        except sr.UnknownValueError:
            print("Sorry, I could not understand what you said.")  # Handle unrecognized speech
            return None
        except sr.RequestError as e:
            print("Sorry, an error occurred with the speech recognition service:", str(e))  # Handle errors with the speech recognition service
            return None
    else:
        return None

#-----------------------------------------------------------

def test_cuda():
    try:
        if torch.cuda.is_available():
            print("CUDA is available.")
            print("Number of CUDA devices:", torch.cuda.device_count())
            print("CUDA device name:", torch.cuda.get_device_name(0))
            print("#-----------------------------------------------------------")
        else:
            print("CUDA is not available.")
            print("#-----------------------------------------------------------")
    except Exception as e:
        print("An error occurred while testing CUDA:", e)
        print("#-----------------------------------------------------------")

#-----------------------------------------------------------

def clean_text(input_text):
    try:
        cleaned_text = input_text.replace("\"", "")
        cleaned_text = cleaned_text.replace("AITA", "am I the A Hole")
        cleaned_text = cleaned_text.replace(".", " ")
        
        return cleaned_text
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

#-----------------------------------------------------------

def TTSMS(input_text, output_path):
    print("Testing Microsoft TTS Hugging Face Model...")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    inputs = processor(text=input_text, return_tensors="pt")

    # Load xvector containing speaker's voice characteristics from a dataset
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    print("Writing Text to Speech Wav file...")
    sf.write(output_path, speech.numpy(), samplerate=16500)
    print("Complete!")

#-----------------------------------------------------------

def TTSGGL(text, language='en', slow=False, output_file='TTS_GGL.mp3'):
    try:
        tts = gTTS(text=text, lang=language, slow=slow)
        tts.save(output_file)
        print(f"Google text-to-speech audio saved as {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

#-----------------------------------------------------------

def increase_playback_speed(input_path, output_path, speed_factor=1.1):
    try:
        song = AudioSegment.from_mp3(input_path)
        
        spedup = song.speedup(playback_speed=speed_factor, chunk_size=150, crossfade=25)
        spedup.export(output_path, format="mp3")
        
        print("Playback speed increased and saved successfully.")
    except Exception as e:
        print("An error occurred:", e)

#-----------------------------------------------------------

def reduce_noise_in_audio(input_file_path, output_file_path):
    try:
        print("Starting noise reduction...")
        audio, sr = librosa.load(input_file_path, sr=None)
        reduced_audio = nr.reduce_noise(y=audio, sr=sr)
        sf.write(output_file_path, reduced_audio, sr)
        print("Noise reduction completed successfully.")
    except FileNotFoundError:
        print("Error: The input file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

#-----------------------------------------------------------

if __name__ == "__main__":

    test_cuda()
    while True:
        user_input = input("Enter a selection, 1 for EN-->CN or 2 for CN-->EN")
        if user_input =='1':
            recorded_text = record_speech()
            en_to_cn_trad = translate_to_chinese_traditional(recorded_text)
            text_to_speech(en_to_cn_trad)
        elif user_input =='2':
            translated_text = speech_to_english()
        else:
            print("Invalid Input.")





    ##THE BELOW CODE IS IF YOU'RE DOING TTS FROM A TEXT FILE

    # #set the location of the input text
    # input_text_path = "input_text.txt"
    
    # #open said text
    # with open(input_text_path, "r") as file:
    #     input_text = file.read()

    # #load .env variables 
    # load_dotenv()

    # #test tensors + run 
    # test_cuda()
    # clean_text(input_text)
    # TTSGGL(input_text,output_file='TTS_GGL.mp3')
    
