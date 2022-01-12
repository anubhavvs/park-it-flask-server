from datetime import datetime
import math
from ocr import ocr_it
import os
import cv2
import json
from bson.objectid import ObjectId
import numpy as np
from flask_pymongo import PyMongo
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, url_for, redirect, render_template
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


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
    input_tensor = tf.convert_to_tensor(
        np.expand_dims(img_np, 0), dtype=tf.uint8)
    detections = infer_model(input_tensor)
    num_detections = int(detections.pop("num_detections"))
    detections = {
        key: value[0, :num_detections].numpy() for key, value in detections.items()
    }
    detections["num_detections"] = num_detections
    detections["detection_classes"] = detections["detection_classes"].astype(
        np.int64)
    image_np_with_detections = img_np.copy()
    image = image_np_with_detections
    return detections, image


# NUMEBR PLATE OCR
def number_plate(image_path):
    detections, image = box_detection(image_path, infer)
    final_text = ocr_it(image, detections,
                        DETECTION_THRESHOLD, REGION_THRESHOLD)
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
            # saves image with ROI
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            # saves number plate text
            text = number_plate(os.path.join(
                app.config["UPLOAD_FOLDER"], filename))
            # query for user with that number plate
            user = db.users.find_one({"plate": text.replace(" ", "")})
            # if no user in found
            if user is None:
                flash('No user found with number plate: '+text)
                return render_template("index.html", filename="detect.png")
            # if a user is found
            else:
                # query for the latest booking by the user
                booking = (
                    db.bookings.find({"user": ObjectId(user["_id"])})
                    .sort([("createdAt", -1)])
                    .limit(1)
                )
                booking_object = booking[0]
                # booking_id for that booking
                booking_id = booking_object["_id"]
                # booking create time
                created_at = booking_object["createdAt"]
                # checks if the booking date is today
                if datetime.utcnow().date() == created_at.date():
                    # sets the status to wait for scan
                    db.bookings.update_one(
                        {"_id": ObjectId(booking_id)},
                        {
                            "$set": {
                                "status": "WAIT FOR SCAN"
                            }
                        },
                    )
                    # checks the status for entry scan
                    if booking_object["status"] == "REACH ON TIME":
                        # updates the booking status and entry time
                        db.bookings.update_one(
                            {"_id": ObjectId(booking_id)},
                            {
                                "$set": {
                                    "status": "ENTRY SCAN SUCCESS",
                                    "startTime": datetime.utcnow(),
                                }
                            },
                        )
                    # checks the status for exit scan
                    elif booking_object["status"] == "ENTRY SCAN SUCCESS":
                        # updates the booking status and exit time
                        db.bookings.update_one(
                            {"_id": ObjectId(booking_id)},
                            {
                                "$set": {
                                    "status": "EXIT SCAN SUCCESS",
                                    "endTime": datetime.now(),
                                }
                            },
                        )
                        # query for price for the area slot
                        price = db.areas.find_one({"name": booking_object["bookedArea"]})[
                            "price"
                        ]
                        # updates the booked slot to empty
                        db.areas.update_one(
                            {
                                "name": booking_object["bookedArea"],
                                "slots.name": booking_object["bookedSlot"],
                            },
                            {"$set": {"slots.$.filled": False}},
                        )
                        # calculates the amount for the booking
                        price_amout = calculatePrice(
                            start=booking_object["startTime"], end=datetime.now(), price=price,
                        )
                        # updates the booking with the price
                        db.bookings.update_one(
                            {"_id": ObjectId(booking_id)}, {
                                "$set": {"price": price_amout}}
                        )
                    else:
                        flash('Invalid Booking Status with number plate: '+text)
                        return render_template("index.html", filename="detect.png")
                else:
                    flash('No active booking found with number plate: '+text)
                    return render_template("index.html", filename="detect.png")
            if text or booking_object:
                flash("Detected Number Plate: " + text)
                flash("Booking status: "+booking_object['status'])
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
