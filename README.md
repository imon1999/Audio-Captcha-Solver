# CAPTCHA Solver: Audio & Image Recognition

## Project Overview
This project automatically decodes CAPTCHA text from audio and image-based CAPTCHAs using:
- Whisper ASR (OpenAI) for speech recognition.
- Tesseract OCR for image text extraction.
- WER & CER evaluation to measure accuracy.

## Installation
Install Dependencies:
   pip install -r requirements.txt

## How to Run
Step 1: Preprocess Audio CAPTCHA
   python preprocessing.py

Step 2: Convert Audio to Text (Whisper ASR)
   python audio2text.py

Step 3: Extract Text from Image CAPTCHA (Tesseract OCR)
   python image2text.py

Step 4: Evaluate Model Accuracy (WER & CER)
   python evaluate.py

## Evaluation Metrics
- Word Error Rate (WER) → Measures word-level mistakes.
- Character Error Rate (CER) → Measures character-level mistakes.
- Model Accuracy (%) → Percentage of correct matches.

Formula for Accuracy:
Accuracy = (Correct Predictions / Total Samples) * 100

## Future Improvements
- Improve ASR Accuracy → Use Whisper-medium/large.
- Enhance OCR Performance → Tune Tesseract settings.
- Parallel Processing → Speed up Whisper & OCR.
- Train Custom OCR Model → Using CNN/Transformer-based approach.

