import os
import cv2
import re
from PIL import Image
import easyocr
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def preprocess_image(image_path, handwritten=False):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("❌ Image not found or failed to load.")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if handwritten:
        # Enhanced preprocessing for handwritten text
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        thresh = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15, 8
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        processed = cv2.dilate(thresh, kernel, iterations=1)
    else:
        # Simpler preprocessing for Excalidraw/printed text
        processed = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15, 8
        )
    
    # Save processed image for EasyOCR
    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + os.path.basename(image_path))
    cv2.imwrite(processed_path, processed)
    return processed_path

def extract_text_from_image(image_path):
    # Initialize EasyOCR reader (English, no GPU for simplicity)
    reader = easyocr.Reader(['en'], gpu=False)
    # Read text from preprocessed image
    result = reader.readtext(image_path, detail=0)
    text = ' '.join(result)
    print(f"EasyOCR output: {text}")  # Debugging
    return text.strip()

def clean_ocr_text(text):
    # Replace common OCR errors
    text = text.replace('O', '0').replace('o', '0')
    text = text.replace('l', '1').replace('I', '1')
    text = text.replace('S', '5').replace('s', '5')
    text = re.sub(r'\s+', '', text)  # Remove extra spaces
    return text

def extract_and_evaluate_expression(text):
    text = clean_ocr_text(text)
    # Regex for basic math expressions
    match = re.search(r'[\d+\-*/().]+', text)
    if match:
        expr = match.group().strip()
        try:
            result = eval(expr, {"__builtins__": {}})  # Safe eval
            return expr, result
        except Exception as e:
            return expr, f"❌ Error evaluating: {e}"
    return None, "❌ No valid expression found."

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error="❌ No file uploaded")

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error="❌ No file selected")

        # Check if user indicates handwritten input
        handwritten = request.form.get('handwritten') == 'on'

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Preprocess image and get processed image path
            preprocessed_path = preprocess_image(filepath, handwritten=handwritten)
            # Extract text using EasyOCR
            text = extract_text_from_image(preprocessed_path)
            # Evaluate expression
            expression, result = extract_and_evaluate_expression(text)
            return render_template('index.html', expr=expression, result=result, image_url=filepath)
        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)