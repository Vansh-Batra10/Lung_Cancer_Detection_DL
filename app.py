from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from werkzeug.utils import secure_filename
import numpy as np
import os

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = "./uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load your model (replace with your actual model path)
MODEL_DIR = "./models"
model = load_model(os.path.join(MODEL_DIR, "ResNet50.h5"))

# Helper functions
def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess the image to match the model's input shape."""
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize to [0, 1]
    return img_array

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle predictions from the model."""
    if "file" not in request.files:
        return render_template('result.html', error="No file uploaded.")

    file = request.files["file"]

    if file.filename == "":
        return render_template('result.html', error="No file selected.")

    if file and allowed_file(file.filename):
        try:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Preprocess the image
            input_image = preprocess_image(filepath)

            # Perform prediction
            predictions = model.predict(input_image)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = float(np.max(predictions))

            # Remove the uploaded file
            os.remove(filepath)

            # Render the result page
            return render_template('result.html', prediction=f"Class {predicted_class}", confidence=f"{confidence:.2f}")
        except Exception as e:
            return render_template('result.html', error=str(e))
    else:
        return render_template('result.html', error="Invalid file type. Please upload a PNG, JPG, or JPEG image.")

if __name__ == '__main__':
    app.run(debug=True)
