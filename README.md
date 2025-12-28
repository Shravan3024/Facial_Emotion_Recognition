# Facial Emotion Recognition System

A comprehensive **Facial Emotion Recognition (FER)** system designed to automatically identify and classify human emotions from facial expressions. This project integrates **computer vision** and **deep learning** techniques to analyze facial features and predict emotions with high accuracy.

The system supports both **real-time emotion detection using a webcam** and **emotion prediction from uploaded images through an interactive Streamlit-based web interface**. Deep learning models trained on facial emotion datasets learn subtle facial patterns such as eye movement, mouth shape, and facial muscle variations to accurately recognize emotions.

---

## 📌 Project Features

- 🎥 Real-time facial emotion recognition using live webcam feed
- 🖼️ Emotion prediction from uploaded facial images via web UI
- 🧠 Deep learning–based emotion classification using CNN architectures
- 😀 Recognition of 7 basic human emotions
- ⚙️ Separate modules for training, evaluation, and inference
- 🧪 Support for experimentation with multiple architectures (EfficientNet, MobileNet)
- 🌐 User-friendly Streamlit interface for easy interaction

---


## 🗂️ Project Structure

```
Facial_Emotions_Recognition/
│
├── dataset/                 # Training dataset
├── logs/                    # Training logs
├── model/                   # Saved model files (.h5 / .pth)
│
├── src/                     # Core source code
│   ├── model.py             # Model architecture
│   ├── utils.py             # Preprocessing utilities
│   ├── realtime_app.py      # Real-time webcam emotion detection
│   ├── train_model.py       # Model training script
│   ├── train_efficientnet.py
│   ├── train_transfer_mobilenet.py
│   ├── evaluate.py          # Model evaluation
│
├── ui/                      # Streamlit UI
│   ├── app.py               # Web UI for emotion prediction
│   └── fer_env/             # Virtual environment (optional)
│
├── .venv / fer_env/         # Virtual environments
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/facial-emotion-recognition.git
cd facial-emotion-recognition
```

---

### 2️⃣ Create & Activate Virtual Environment

```bash
python -m venv fer_env
fer_env\Scripts\activate   # Windows
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Make sure TensorFlow, PyTorch, OpenCV, and Streamlit are installed correctly.

---

## ▶️ Running the Applications

### 🔹 Real-Time Emotion Detection (Webcam)

```bash
cd src
python realtime_app.py
```

This will open your webcam and start detecting facial emotions in real time.

---

### 🔹 Streamlit Web UI

```bash
cd ui
streamlit run app.py
```

Open your browser at:
```
http://localhost:8501
```

Upload a face image to get emotion predictions.

---

## 🧠 Model Information

The emotion recognition model is built using **Convolutional Neural Networks (CNNs)** with transfer learning techniques. Pre-trained architectures such as **EfficientNet** and **MobileNet** are fine-tuned on facial emotion datasets to improve performance while reducing training time.

- Input: Preprocessed grayscale face images
- Face Detection: Haar Cascade Classifier (OpenCV)
- Output: Probability distribution over emotion classes
- Training Frameworks: TensorFlow / Keras (primary), PyTorch (optional)
- Model Weights: Stored separately for reuse during inference

---


## 📊 Supported Emotions

| Label | Emotion |
|------|--------|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Sad |
| 5 | Surprise |
| 6 | Neutral |

---

## 🛠️ Technologies Used

- **Python 3.10+** – Core programming language
- **TensorFlow / Keras** – Model building and training
- **PyTorch** – Alternative deep learning framework support
- **OpenCV** – Face detection and image processing
- **Streamlit** – Interactive web-based user interface
- **NumPy** – Numerical computations
- **Matplotlib** – Visualization and analysis

---






## 👩‍💻 Author

**Shravan Navale**  
Facial Emotion Recognition Project

---


## ⭐ Acknowledgements

- OpenCV Haar Cascades
- TensorFlow & PyTorch communities
- Streamlit for UI framework

---


