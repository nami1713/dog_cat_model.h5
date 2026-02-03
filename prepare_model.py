import tensorflow as tf
from tensorflow.keras.models import load_model

# load data
model_path = "model/model.h5"

try:
    # Load model để kiểm tra
    model = load_model(model_path)
    model.summary()
    print("Model loaded and verified successfully!")
except Exception as e:
    print(f"Error loading model: {e}")