import io
from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Load the lung cancer prediction model
# Replace 'model_path' with the actual path to your saved model
model = tf.keras.models.load_model('model.pkl')


# Function to preprocess the uploaded image
def preprocess_image(image):
    image_np = np.array(image)
    image_np = image_np.resize((128, 128))  # Resize the image to the input size of your CNN model
    image_np = image_np / 255.0  # Normalize the image
    image_np = np.expand_dims(image_np, axis=0)  # Add batch dimension
    return image_np


# Route to handle the home page
@app.route('/')
def home():
    return render_template('index.html')


# Route to handle the image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'images' not in request.files:
        return "No image uploaded.", 400

    image = request.files['images']
    if image.filename == '':
        return "No selected image.", 400

    # Preprocess the image for prediction
    processed_image = preprocess_image(Image.open(io.BytesIO(image.read())))

    # Make a prediction using the loaded model
    prediction = model.predict(processed_image)[0][0]

    # Determine the cancer prediction result
    if 0 <= prediction <= 0.5:
        result = "Cancerous"
    else:
        result = "Non-cancerous"

    # Display the result on the '/predict' page
    return render_template('prediction.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
