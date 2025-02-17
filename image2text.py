from PIL import Image
import pytesseract
import re
import os
import pandas as pd

# Set path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Dataset path
dataset_path = "Dataset/"
image_folder = os.path.join(dataset_path, "Dataset/images/")
output_csv = os.path.join(dataset_path, "image_results.csv")

def preprocess_image(image):
    gray = image.convert('L')  
    binary = gray.point(lambda x: 0 if x < 140 else 255, '1')  
    return binary

def extract_text_from_image(image_path):
    processed_image = preprocess_image(Image.open(image_path))
    config = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    text = pytesseract.image_to_string(processed_image, config=config).strip()
    return re.sub(r'[^a-zA-Z0-9]', '', text)

# Process all image files and save results
results = []
for file in os.listdir(image_folder):
    if file.endswith(".png"):
        filename = file.replace(".png", "")
        image_path = os.path.join(image_folder, file)
        extracted_text = extract_text_from_image(image_path)
        results.append({"filename": filename, "image_text": extracted_text})

# Save results 
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"Image text results saved to: {output_csv}")
