# Covid-19 Chest X-Ray Detection Web App

## Overview
This project is a web application for detecting Covid-19, Viral Pneumonia, or Normal status from chest X-ray images using a deep learning model. Users can upload an image or capture one using their device's camera. The app processes the image, predicts the class, and displays the result with a confidence score.

## Features
- Upload chest X-ray images for prediction
- Capture images directly from your device's camera
- Deep learning model (CNN) for classification
- User-friendly web interface
- Displays prediction and confidence score

## Project Structure
```
Projects/Covid19_Radiotherapy/
│
├── app.py                  # Main Flask application
├── models/
│   ├── CNN_Covid19_Xray_Version.h5   # Trained Keras model
│   └── Label_encoder.pkl             # Label encoder
├── uploads/                # Stores uploaded/captured images
├── templates/
│   ├── index.html          # Home page (upload/camera)
│   ├── camera.html         # Camera capture interface
│   └── result.html         # Result display page
└── main.ipynb              # Model training and evaluation notebook
```

## How It Works
1. **Home Page**: Upload an X-ray image or open the camera interface.
2. **Image Upload**: Image is saved and processed, then passed to the trained model for prediction.
3. **Camera Capture**: Capture an image in-browser and send it for prediction.
4. **Result**: The app displays the predicted class and confidence score.

## Tools & Libraries Used
- **Flask**: Web framework for Python
- **Keras / TensorFlow**: Deep learning model and utilities
- **OpenCV (cv2)**: Image processing
- **NumPy**: Numerical operations
- **Pandas**: Data manipulation (in notebook)
- **Matplotlib / Seaborn**: Data visualization (in notebook)
- **scikit-learn**: Label encoding, train/test split, metrics
- **HTML/CSS/JavaScript**: Frontend (camera, upload, result display)
- **Werkzeug**: Secure file handling
- **pickle**: Model/encoder serialization

## Getting Started
1. Install required packages:
   ```bash
   pip install flask keras tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn
   ```
2. Place the trained model (`CNN_Covid19_Xray_Version.h5`) and label encoder (`Label_encoder.pkl`) in the `models/` directory.
3. Run the Flask app:
   ```bash
   python app.py
   ```
4. Open your browser at `http://localhost:5000/`.

## Model Training
- The model is trained in `main.ipynb` using a labeled dataset of chest X-ray images.
- The notebook covers data loading, preprocessing, model definition, training, evaluation, and saving the model and label encoder.

## Notes
- This app is for educational and research purposes only. Not for clinical use.
- Ensure your environment has a camera if you want to use the camera capture feature.
