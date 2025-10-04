from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.preprocessing import LabelEncoder
import numpy as np
import cv2
import os
import joblib

app = Flask(__name__)

# Load model and label encoder
MODEL_PATH = r'D:\SkinDiseaseApp\my_model.h5'
model = load_model(MODEL_PATH)
label_encoder = joblib.load(r'D:\SkinDiseaseApp\label_encoder.pkl')  # Save the encoder during training

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction="No file uploaded")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction="No file selected")

    # Read image from user upload
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(np.expand_dims(img, axis=0))

    # Make prediction
    preds = model.predict(img)
    pred_index = np.argmax(preds)
    pred_label = label_encoder.classes_[pred_index]

    return render_template('index.html', prediction=f"Predicted: {pred_label}")

if __name__ == '__main__':
    # ✅ Disable reloader to fix "signal only works in main thread" error
    app.run(debug=True, use_reloader=False)
