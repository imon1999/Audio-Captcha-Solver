import librosa
import numpy as np
import noisereduce as nr
import soundfile as sf
from scipy.signal import butter, lfilter
import os

# Dataset path 
dataset_path = "C:/Users/YourName/CAPTCHA_Dataset/"
audio_folder = os.path.join(dataset_path, "audio/")

def bandpass_filter(audio, sr, lowcut=300, highcut=3400):
    nyquist = 0.5 * sr
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = butter(1, [low, high], btype="band")
    return lfilter(b, a, audio)

def preprocess_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    y = bandpass_filter(y, sr)  # Apply bandpass filter
    y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.6)  
    output_path = audio_path.replace(".wav", "_processed.wav")
    sf.write(output_path, y, sr)
    return output_path


if __name__ == '__main__':
    for file in os.listdir(audio_folder):
        if file.endswith(".wav"):
            audio_path = os.path.join(audio_folder, file)
            processed_path = preprocess_audio(audio_path)
            print(f"Processed Audio: {processed_path}")
