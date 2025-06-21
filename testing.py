import torch
from easyocr import Reader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numexpr
import math

# 1. Initialize Reader and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
reader = Reader(['en'], gpu=(device.type == 'cuda'))

# 2. Load model weights
model = reader.recognizer
if hasattr(model, 'module'):  # Handle DataParallel wrapper
    model = model.module

# Load checkpoint (use weights_only for security)
checkpoint = torch.load('model/math_ocr_epoch_50.pth', 
                       map_location=device,
                       weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

def evaluate_expression(prediction: str) -> float:
    """Safely evaluates math expressions from OCR output"""
    try:
        # Sanitize input
        sanitized = (prediction
                    .replace('^', '**')  # Handle exponents
                    .replace('x', '*')    # Handle alternative multiply
                    .replace(' ', '')     # Remove spaces
                    )
        # Allow only math-safe characters
        allowed_chars = set('0123456789+-*/().**')
        sanitized = ''.join(c for c in sanitized if c in allowed_chars)
        
        if not sanitized:
            return float('nan')
            
        return float(numexpr.evaluate(sanitized))
    except:
        return float('nan')  # Return NaN for invalid expressions

def predict(image_path):
    """Predict and evaluate math expression from image"""
    # Preprocessing (must match training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 128)),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    try:
        img = Image.open(image_path).convert('L')  # Grayscale
        img_tensor = transform(img).unsqueeze(0).to(device)
        text_for_pred = torch.LongTensor(1, 26).fill_(0).to(device)
        
        # Get predictions
        with torch.no_grad():
            output = model(img_tensor, text_for_pred)
            _, preds = output.max(2)
        
        # Decode using EasyOCR's converter
        pred_str = reader.readtext(image_path, detail=0)
        if not pred_str:
            return {"expression": "", "result": float('nan'), "error": "No text detected"}
        
        expression = pred_str[0]
        result = evaluate_expression(expression)
        
        return {
            "expression": expression,
            "result": result,
            "error": None if not math.isnan(result) else "Invalid expression"
        }
        
    except Exception as e:
        return {
            "expression": "",
            "result": float('nan'),
            "error": f"Prediction error: {str(e)}"
        }

def display_results(image_path, results):
    """Display image with prediction results"""
    img = Image.open(image_path)
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    
    if results['error']:
        title = f"Error: {results['error']}"
    else:
        title = (f"Expression: {results['expression']}\n"
                f"Result: {results['result']}" if not math.isnan(results['result']) 
                else "Could not evaluate")
    
    plt.title(title, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Test prediction
if __name__ == '__main__':
    image_path = "IMG_7933.jpg"  # Replace with your image
    
    results = predict(image_path)
    print("Raw Prediction:", results['expression'])
    
    if not math.isnan(results['result']):
        print(f"Evaluation: {results['expression']} = {results['result']}")
    else:
        print("Evaluation failed:", results['error'])
    
    display_results(image_path, results)