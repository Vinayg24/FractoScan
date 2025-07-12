<div align="center">
  
#  FractoScan: AI-Powered Bone Fracture Detection
</div>
<div align="center">
  
![Version](https://img.shields.io/badge/version-1.0.0-blue.svg?cacheSeconds=2592000)
![Built with: Python](https://img.shields.io/badge/Built%20with-Python-3776AB?logo=python&logoColor=white)
![Framework: FastAPI](https://img.shields.io/badge/Framework-FastAPI-009688?logo=fastapi&logoColor=white)
![Model: YOLOv8](https://img.shields.io/badge/Model-YOLOv8-00FFFF?logo=pytorch&logoColor=white)

</div>

<p align="center">
  <i>Advanced deep learning system for rapid and accurate bone fracture detection from X-ray images.</i>
</p>

## 📋 Overview

FractoScan is a state-of-the-art web application that uses artificial intelligence to help healthcare professionals identify bone fractures from X-ray images with high accuracy. The system leverages YOLOv8 deep learning models and provides visual explanations using Grad-CAM technology to enhance trust and interpretability.

## ✨ Key Features

- 🔍 **Advanced Detection**: Identifies various fracture types (transverse, oblique, spiral, comminuted, greenstick)
- ⚡ **Speed**: Processes X-ray images in under 2 seconds
- 🔬 **Grad-CAM Visualization**: Shows exactly where the model is focusing, increasing clinical trust
- 📊 **Detailed Analysis**: Provides confidence scores and explanations for each detection
- 🌐 **Web Interface**: User-friendly frontend for easy image uploads and result viewing
- 🔄 **API Access**: RESTful endpoints for integration with other clinical systems

## 🖼️ Screenshots

<div align="center">
  <table>
    <tr>
      <td><strong>Detection Results</strong></td>
      <td><strong>Grad-CAM Visualization</strong></td>
    </tr>
    <tr>
      <td><img src="frontend/images/sample-result.jpg" alt="Detection Results" width="100%"/></td>
      <td><img src="frontend/images/sample-gradcam.jpg" alt="Grad-CAM Visualization" width="100%"/></td>
    </tr>
  </table>
</div>

## 🏗️ Architecture

FractoScan consists of two main components:

1. **Backend (FastAPI)**: Handles image processing, runs the YOLOv8 model, and generates visualizations
2. **Frontend (HTML/CSS/JS)**: Provides user interface for uploading X-rays and viewing results

## 🛠️ Technology Stack

- **Backend**: Python, FastAPI, PyTorch, Ultralytics YOLOv8, OpenCV
- **Frontend**: HTML5, CSS3, JavaScript, FontAwesome
- **AI Model**: YOLOv8 trained on bone fracture X-ray datasets
- **Visualization**: Grad-CAM (Gradient-weighted Class Activation Mapping)

## 📂 Project Structure

```
├── graphs/
│   ├── epochs_vs_accuracy.png
│   └── confusion_matrices/
├── models/
│   ├── yolo_model.py
├── backend/                # FastAPI backend
│   ├── app.py              # Main application file
│   ├── requirements.txt    # Backend dependencies
│   ├── models/             # YOLOv8 model files
│   ├── results/            # Detection results
│   │   ├── explanations/   # Model explanation images
│   │   └── gradcam/        # Grad-CAM visualization images
│   └── uploads/            # Uploaded X-ray images
│
├── frontend/               # Static web frontend
│   ├── index.html          # Home page
│   ├── detect.html         # Detection page
│   ├── gradcam.html        # Grad-CAM explanation page
│   ├── about.html          # About page
│   ├── features.html       # Features page
│   ├── team.html           # Team page
│   ├── contact.html        # Contact page
│   ├── styles.css          # Main stylesheet
│   └── images/             # Frontend images
│
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Modern web browser

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/FractoScan.git
   cd FractoScan
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the backend server**
   ```bash
   cd backend
   uvicorn app:app --reload
   ```

4. **Open the frontend**
   - Navigate to the `frontend` folder
   - Open `index.html` in your web browser

   OR

   - Serve the frontend using a simple HTTP server:
     ```bash
     cd frontend
     python -m http.server 8080
     ```
   - Open `http://localhost:8080` in your browser

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API status check |
| `/status` | GET | Get backend system status |
| `/detect` | POST | Upload and analyze X-ray image |
| `/gradcam/{image_id}` | GET | Get Grad-CAM visualization for a specific image |

### Example API Usage

```python
import requests

# Upload an X-ray image for detection
with open('xray.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/detect', files=files)

result = response.json()
print(f"Detection ID: {result['detection_id']}")
print(f"Result image: {result['result_image']}")
print(f"Grad-CAM visualization: {result['gradcam_image']}")
```

## 💻 Development Setup

For developers who want to contribute to the project:

1. **Create a virtual environment**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

2. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run backend with debug mode**
   ```bash
   cd backend
   uvicorn app:app --reload --debug
   ```

## 👥 Our Team

FractoScan is developed by a multidisciplinary team of AI/ML developers and software engineers:

- **Laksh Vyas ** - Frontend and backend Lead
- **Vinay Goswami** -  Backend Lead
## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the object detection model
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- Medical partners for providing expertise and testing

---

<div align="center">
  <p>Made with ❤️ for improving medical diagnostics</p>
  <p>© 2025 FractoScan Team</p>
</div>