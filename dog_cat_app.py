from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import logging
import io
from PIL import Image

# Cấu hình logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app1 = Flask(__name__)

# Load model ONCE
# Lưu ý: Cần đảm bảo file model.h5 nằm trong thư mục model/
MODEL_PATH = "model/model.h5"
model = load_model(MODEL_PATH)

# Mapping label -> tên loài vật
# Giả định: 0 là Cat, 1 là Dog (phổ biến trong dataset Dogs vs Cats)
LABEL_MAP = {
    0: "Cat",
    1: "Dog"
}


def preprocess_image(img_bytes, target_size=(128, 128)):
    """Tiền xử lý hình ảnh trước khi đưa vào model"""
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert("RGB")  # Đảm bảo 3 kênh màu
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Chuẩn hóa pixel về [0, 1]
    return img_array


# API: Nhận file ảnh và trả về JSON
@app1.route("/predict_dogcat", methods=["POST"])
def predict_api():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        img_bytes = file.read()
        logger.info(f"Received image: {file.filename}")

        # Tiền xử lý (target_size phải khớp với lúc train model
        processed_img = preprocess_image(img_bytes, target_size=(128, 128))

        prediction = model.predict(processed_img)
        # Model .h5 thường trả về xác suất (sigmoid). > 0.5 là Dog, < 0.5 là Cat
        class_id = 1 if prediction[0][0] > 0.5 else 0
        confidence = float(prediction[0][0]) if class_id == 1 else float(1 - prediction[0][0])

        return jsonify({
            "class_id": class_id,
            "class_name": LABEL_MAP[class_id],
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        logger.exception("Error occurred during API prediction")
        return jsonify({"error": str(e)}), 500


# Web Route mặc định
@app1.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        try:
            if 'file' in request.files:
                file = request.files['file']
                if file.filename != '':
                    img_bytes = file.read()
                    processed_img = preprocess_image(img_bytes, target_size=(128, 128))

                    pred_score = model.predict(processed_img)[0][0]
                    class_id = 1 if pred_score > 0.5 else 0

                    prediction = LABEL_MAP[class_id]
                    confidence = f"{pred_score * 100:.2f}%" if class_id == 1 else f"{(1 - pred_score) * 100:.2f}%"

        except Exception as e:
            logger.error(f"Input error: {e}")
            prediction = f"Lỗi xử lý ảnh: {e}"

    return render_template("index.html", prediction=prediction, confidence=confidence)


# Web Route phụ
@app1.route("/b", methods=["GET", "POST"])
def index_b():
    prediction = None

    if request.method == "POST":
        try:
            if 'file' in request.files:
                file = request.files['file']
                img_bytes = file.read()
                processed_img = preprocess_image(img_bytes, target_size=(128,128))
                pred_score = model.predict(processed_img)[0][0]
                prediction = LABEL_MAP[1 if pred_score > 0.5 else 0]
        except Exception as e:
            prediction = f"Lỗi: {e}"

    return render_template("index_bootstrap.html", prediction=prediction)


if __name__ == "__main__":
    app1.run(
        host="127.0.0.1",
        port=8001,
        debug=False

    )
