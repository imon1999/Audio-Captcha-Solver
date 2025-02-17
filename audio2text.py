import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import re
import os
import pandas as pd

# Load Model
model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Dataset path 
dataset_path = "Dataset/"
audio_folder = os.path.join(dataset_path, "audio/")
output_csv = os.path.join(dataset_path, "audio_results.csv")

def decode_audio_whisper(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    input_features = processor(waveform.squeeze(0), sampling_rate=sample_rate, return_tensors="pt").input_features

    try:
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return re.sub(r'[^a-zA-Z0-9]', '', text.strip())

    except Exception as e:
        return f"[ERROR] Whisper failed: {e}"

# Process all audio files and save results
results = []
for file in os.listdir(audio_folder):
    if file.endswith("_processed.wav"):
        filename = file.replace("_processed.wav", "")
        audio_path = os.path.join(audio_folder, file)
        predicted_text = decode_audio_whisper(audio_path)
        results.append({"filename": filename, "audio_text": predicted_text})

# Save results
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"Audio text results saved to: {output_csv}")
