import torch
from easyocr import Reader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numexpr
import math
import os

# ------------------- SETTINGS ---------------------
IMAGE_PATH = "test_img/IMG_7944.jpg"  # 🔁 Replace with your actual test image
CHECKPOINT_PATH = "model/math_ocr_epoch_50.pth"
# --------------------------------------------------

# Set device (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize EasyOCR Reader
reader = Reader(['en'], gpu=(device.type == 'cuda'))

# Extract model from reader
model = reader.recognizer
if hasattr(model, 'module'):  # Handle DataParallel
    model = model.module

# Load full checkpoint (remove weights_only=True)
if os.path.exists(CHECKPOINT_PATH):
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model checkpoint from: {CHECKPOINT_PATH}")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
else:
    print(f"Checkpoint file not found: {CHECKPOINT_PATH}")

model.to(device)
model.eval()

def evaluate_expression(prediction: str) -> float:
    """Evaluates a math expression (e.g., '2+3*4') safely."""
    try:
        sanitized = (
            prediction.replace('^', '**')
                      .replace('x', '*')
                      .replace(' ', '')
        )
        allowed_chars = set("0123456789+-*/().")
        sanitized = ''.join(c for c in sanitized if c in allowed_chars)

        if not sanitized:
            return float('nan')

        return float(numexpr.evaluate(sanitized))
    except:
        return float('nan')

def predict(image_path):
    """Predict and evaluate the math expression from an image."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 128)),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    if not os.path.exists(image_path):
        return {
            "expression": "",
            "result": float('nan'),
            "error": f"Image not found: {image_path}"
        }

    try:
        img = Image.open(image_path).convert('L')
        img_tensor = transform(img).unsqueeze(0).to(device)
        text_for_pred = torch.LongTensor(1, 26).fill_(0).to(device)

        with torch.no_grad():
            output = model(img_tensor, text_for_pred)

        # Use EasyOCR reader to decode
        pred_str = reader.readtext(image_path, detail=0)

        if not pred_str:
            return {"expression": "", "result": float('nan'), "error": "No text detected"}

        expression = pred_str[0]
        result = evaluate_expression(expression)

        return {
            "expression": expression,
            "result": result,
            "error": None if not math.isnan(result) else "Invalid math expression"
        }

    except Exception as e:
        return {
            "expression": "",
            "result": float('nan'),
            "error": f"Prediction failed: {str(e)}"
        }

def display_results(image_path, results):
    """Display image with OCR and evaluation result."""
    if not os.path.exists(image_path):
        print("Image file missing for display.")
        return

    img = Image.open(image_path)
    plt.figure(figsize=(10, 6))
    plt.imshow(img)

    if results['error']:
        title = f"Error: {results['error']}"
    else:
        title = f"Expression: {results['expression']}\nResult: {results['result']}"

    plt.title(title, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Run test
if __name__ == '__main__':
    results = predict(IMAGE_PATH)
    print("Predicted Expression:", results['expression'])

    if not math.isnan(results['result']):
        print(f"Evaluated Result: {results['result']}")
    else:
        print("Evaluation Error:", results['error'])

    display_results(IMAGE_PATH, results)
