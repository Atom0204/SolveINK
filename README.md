# 🧠 Math Expression OCR Evaluator

A deep learning-based OCR system to recognize and evaluate **handwritten mathematical expressions** from images using **PyTorch**, **EasyOCR**, and **custom CNN training**.

---

## 📌 Features

- ✅ Detects and reads handwritten math expressions from images
- 🧠 Evaluates the expression using `numexpr` for safe computation
- 🎓 Custom training using EasyOCR’s recognizer
- 💾 Model checkpoints saving for later evaluation
- 📊 Visual output using `matplotlib`

---

## 🖼️ Demo

| Input Image | OCR + Evaluation |
|-------------|------------------|
| ![sample](assets/sample.jpg) | `Expression: 2+3*4`, Result: `14` |


SolveINK/
├── EasyOCR/
│   └── math_ocr_dataset/      # Contains generated CSVs + expression images
│       ├── train.csv
│       ├── val.csv
│       └── *.jpg (expression images)
│
├── model/                     # Trained model checkpoints (.pth files)
│
├── train_math_ocr.py         # Training script
├── testing.py                # Testing/evaluation script
├── custom_dataset.py         # Dataset class for OCR training
├── generator.py              # Script to generate expression images
├── exp.py                    # Script to generate expressions for OCR dataset
│
├── requirements.txt
├── .gitignore
└── README.md


---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/SolveINK.git
cd SolveINK

2. Setup Virtual Environment

3. Install Requirements
pip install -r requirements.txt

⚡ Enabling CUDA (GPU Acceleration)
This project supports GPU acceleration using CUDA-enabled PyTorch builds. By default, the scripts will automatically detect and use the GPU if available.

✅ Requirements
NVIDIA GPU

Installed CUDA Toolkit ≥ 12.1 (matching PyTorch build)

Compatible PyTorch + torchvision with CUDA support

🔧 Installing PyTorch with CUDA Support
Use the correct install command depending on your Python version:

✅ For Python 3.10 or 3.11:
pip install torch==2.5.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121
🛑 Python 3.13 is not supported by PyTorch yet. Use Python 3.10 or 3.11 instead.

💡 Notes
If CUDA is not available, the code will automatically fall back to CPU.

Training on GPU is significantly faster — use it if available!

Ensure you're not using a CPU-only PyTorch build (pip list should show +cu121 in torch version).
# Current torch version is CPU-only.
# To use GPU acceleration, uninstall this and install the CUDA version:
# pip install torch==2.5.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121



4. Run training:
python train_math_ocr.py
Model checkpoints will be saved in:- SolveINK/model/

🔍 Testing on an Image
python main.py
Edit main.py to change the test image path (IMG_7933.jpg) before running.


✅ Optional 

📁 Dataset Generation
To create synthetic math expression images:
python generator.py

To generate expressions or manipulate data:
python exp.py


⚙️ Tech Stack
PyTorch

EasyOCR

Torchvision

NumExpr

Matplotlib

PIL



🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you’d like to change.

