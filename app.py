from flask import Flask, flash, request, jsonify, url_for, redirect, render_template
from werkzeug.utils import secure_filename
from flask_pymongo import PyMongo
import numpy as np
from bson.json_util import dumps
from bson.objectid import ObjectId
import json
import cv2
import os
from ocr import ocr_it
import math
from datetime import datetime

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
app.config.from_pyfile("config.py", silent=True)
app.config.from_pyfile("config.py", silent=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["DETECTION_FOLDER"] = DETECTION_DIR

# PYMONGO CONFIG
mongodb_client = PyMongo(app)
db = mongodb_client.db

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
    if final_text:
        if len(final_text) != 0:
            return final_text[0]
    return None


# HELPER FUNCTION
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def calculatePrice(start, end, price):
    diff = end - start
    diff_in_hours = diff.total_seconds() / 3600
    return math.ceil(diff_in_hours) * price


# FLASK ROUTES
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("index.html")
    elif request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filename = "upload." + filename.split(".")[1]
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            text = number_plate(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            user = db.users.find_one({"plate": text.replace(" ", "")})
            booking = (
                db.bookings.find({"user": ObjectId(user["_id"])})
                .sort([("createdAt", -1)])
                .limit(1)
            )
            for doc in booking:
                booking_object = doc
            booking_id = booking_object["_id"]
            if booking_object["status"] == "REACH ON TIME":
                db.bookings.update_one(
                    {"_id": ObjectId(booking_id)},
                    {
                        "$set": {
                            "status": "ENTRY SCAN SUCCESS",
                            "startTime": datetime.now(),
                        }
                    },
                )
            elif booking_object["status"] == "ENTRY SCAN SUCCESS":
                db.bookings.update_one(
                    {"_id": ObjectId(booking_id)},
                    {
                        "$set": {
                            "status": "EXIT SCAN SUCCESS",
                            "endTime": datetime.now(),
                        }
                    },
                )
                price = db.areas.find_one({"name": booking_object["bookedArea"]})[
                    "price"
                ]
                db.areas.update_one(
                    {
                        "name": booking_object["bookedArea"],
                        "slots.name": booking_object["bookedSlot"],
                    },
                    {"$set": {"slots.$.filled": False}},
                )
                price_amout = calculatePrice(
                    start=booking_object["startTime"], end=datetime.now(), price=price,
                )
                print(price_amout)
                db.bookings.update_one(
                    {"_id": ObjectId(booking_id)}, {"$set": {"price": price_amout}}
                )
            if text:
                flash("Detected Number Plate: " + text)
                return render_template("index.html", filename="detect.png")
            else:
                flash("Unable to detect number plate!")
                return render_template("index.html", filename=filename)

        else:
            flash("Allowed image types are - png, jpg, jpeg, gif")
            return redirect(request.url)


@app.route("/display/<filename>")
def display_image(filename):
    return redirect(url_for("static", filename="detections/" + filename), code=301)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
