from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import os
from ocr import ocr_it

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

# CONSTANTS
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MODEL_DIR = os.path.abspath("./saved_model/")
UPLOAD_DIR = os.path.abspath("./static/uploads/")
DETECTION_DIR = os.path.abspath("./static/detections/")
DETECTION_THRESHOLD = 0.5
REGION_THRESHOLD = 0.1

# FLASK CONFIG
app = Flask(__name__)
app.secret_key = "park-it-secret"
app.config.from_pyfile('config.py', silent=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["DETECTION_FOLDER"] = DETECTION_DIR


# TF MODEL LOAD
loading = tf.saved_model.load(MODEL_DIR)
infer = loading.signatures["serving_default"]
if infer:
    print("Model loaded successfully.")
else:
    print("Failed to load model.")

# TF OBJECT DETECTION
def box_detection(image_path, infer_model):
    img = cv2.imread(image_path)
    img_np = np.array(img)
    input_tensor = tf.convert_to_tensor(np.expand_dims(img_np, 0), dtype=tf.uint8)
    detections = infer_model(input_tensor)
    num_detections = int(detections.pop("num_detections"))
    detections = {
        key: value[0, :num_detections].numpy() for key, value in detections.items()
    }
    detections["num_detections"] = num_detections
    detections["detection_classes"] = detections["detection_classes"].astype(np.int64)
    image_np_with_detections = img_np.copy()
    image = image_np_with_detections
    return detections, image


# NUMEBR PLATE OCR
def number_plate(image_path):
    detections, image = box_detection(image_path, infer)
    final_text = ocr_it(image, detections, DETECTION_THRESHOLD, REGION_THRESHOLD)
    if len(final_text) != 0:
        return final_text[0]
    else:
        return None


# HELPER FUNCTION
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# FLASK ROUTES
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return jsonify({"message": "Hello World!"})
    elif request.method == "POST":
        if "file" not in request.files:
            return jsonify({"response": "Invalid Request."}), 400
        file = request.files["file"]
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filename = "upload." + filename.split(".")[1]
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            text = number_plate(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            return jsonify({"number_plate": text}), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
