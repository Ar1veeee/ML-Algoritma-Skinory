from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
from google.cloud import storage
import uuid
import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Google Cloud Storage configuration
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("SERVICE_ACCOUNT_PATH")
storage_client = storage.Client()
BUCKET_NAME = "skinory-classify"
bucket = storage_client.bucket(BUCKET_NAME)

# Load trained model
MODEL_PATH = "Model/SKINORY.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Skin type labels
SKIN_TYPE_LABELS = {0: "dry", 1: "normal", 2: "oily"}

# Database connection
def get_db_connection():
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
    )
    return conn

# Preprocessing function
def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess the image for model prediction.

    Parameters:
    - image: PIL.Image, input image to preprocess.
    - target_size: tuple, the target size for the image.

    Returns:
    - numpy array of the processed image with added batch dimension.
    """
    # Konversi gambar ke RGB jika memiliki alpha channel
    if image.mode == "RGBA":
        image = image.convert("RGB")
    elif image.mode == "L":  # Jika grayscale, ubah ke RGB
        image = image.convert("RGB")

    # Resize gambar
    image = image.resize(target_size)
    image_array = np.asarray(image) / 255.0  # Normalize pixel values
    return np.expand_dims(image_array, axis=0)  # Add batch dimension


@app.route("/")
def home():
    return "Welcome to the Skin Type Prediction API!"

@app.route("/predict/<int:user_id>", methods=["POST"])
def predict(user_id):
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file in the request."}), 400

        image_file = request.files["image"]

        # Validasi format file gambar
        valid_extensions = ("jpg", "jpeg", "png")
        if not image_file.filename.lower().endswith(valid_extensions):
            return jsonify({"error": f"Invalid image file format. Only {', '.join(valid_extensions)} are accepted."}), 400

        # Tentukan nama file unik berdasarkan format asli
        file_extension = image_file.filename.split(".")[-1].lower()
        unique_filename = f"{uuid.uuid4()}.{file_extension}"

        # Upload gambar ke Google Cloud Storage
        blob = bucket.blob(unique_filename)
        blob.upload_from_file(image_file)
        image_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{unique_filename}"

        # Preprocessing gambar
        image_file.seek(0)  # Reset pointer file
        image = Image.open(image_file)
        processed_image = preprocess_image(image)

        # Prediksi menggunakan model
        predictions = model.predict(processed_image)
        class_names = ['dry', 'normal', 'oily']  # Nama kelas sesuai dataset Anda
        probabilities = predictions[0].astype(float)  # Konversi ke tipe data float Python

        # Mapping probabilitas ke nama kelas
        probabilities_dict = {
            class_name: round(float(prob) * 100, 2) for class_name, prob in zip(class_names, probabilities)
        }

        # Kelas dengan probabilitas tertinggi
        predicted_class_index = int(np.argmax(probabilities))  # Pastikan integer
        predicted_class_name = class_names[predicted_class_index]
        predicted_probability = round(float(probabilities[predicted_class_index]) * 100, 2)

        # Update jenis kulit pengguna di database
        conn = get_db_connection()
        cursor = conn.cursor()
        update_query = "UPDATE users SET skin_type = %s WHERE id = %s"
        cursor.execute(update_query, (predicted_class_name, user_id))
        conn.commit()
        cursor.close()
        conn.close()

        # Format respons
        response = {
            "Probabilities per class": probabilities_dict,
            "Predicted class": predicted_class_name,
            "Predicted probability": f"{predicted_probability}%",
            "Image URL": image_url,
        }

        return jsonify(response)

    except mysql.connector.Error as db_err:
        return jsonify({"error": f"Database error: {str(db_err)}"}), 500

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
