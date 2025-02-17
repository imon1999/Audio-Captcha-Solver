import pandas as pd
import jiwer
import os

# Dataset path 
dataset_path = "Dataset/"
audio_results_csv = os.path.join(dataset_path, "audio_results.csv")
image_results_csv = os.path.join(dataset_path, "image_results.csv")

# Load results
df_audio = pd.read_csv(audio_results_csv)
df_image = pd.read_csv(image_results_csv)
df = pd.merge(df_audio, df_image, on="filename")  

def evaluate_accuracy(predicted_text, ground_truth):
    if not predicted_text or not ground_truth:
        return None, None, False  # Skip empty values

    wer = jiwer.wer(ground_truth, predicted_text)
    cer = jiwer.cer(ground_truth, predicted_text)
    correct = predicted_text == ground_truth  # Exact match

    return round(wer, 3), round(cer, 3), correct

# Initialize variables
total_samples = 0
correct_predictions = 0
wer_list = []
cer_list = []

# Evaluate samples
for _, row in df.iterrows():
    wer, cer, is_correct = evaluate_accuracy(row["audio_text"], row["image_text"])
    
    if wer is not None and cer is not None:
        total_samples += 1
        wer_list.append(wer)
        cer_list.append(cer)
        if is_correct:
            correct_predictions += 1  

# Compute accuracy
accuracy = correct_predictions / total_samples if total_samples > 0 else 0
cer_avg = sum(cer_list) / len(cer_list) if cer_list else 0
wer_avg = sum(wer_list) / len(wer_list) if wer_list else 0


print(f"Total Accuracy: {accuracy * 100:.2f}%")
print(f"Average CER: {cer_avg:.3f}")
print(f"Average WER: {wer_avg:.3f}")
