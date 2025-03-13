import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import io

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('cifar10_model_v1.keras')

# Class names for CIFAR-10 dataset
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Helper function to preprocess the input image
def preprocess_image(image):
    # Convert the image to RGB if it's not
    image = image.convert('RGB')
    # Resize the image to (32, 32) as the model expects 32x32 input
    image = image.resize((32, 32))
    # Convert the image to a numpy array and normalize
    image_array = np.array(image) / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the image is in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # If no file is selected, return an error
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Open the image file
        image = Image.open(file.stream)
        
        # Preprocess the image
        image_array = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions[0])
        predicted_label = class_names[predicted_class]
        
        # Return the prediction as a JSON response
        return jsonify({
            'predicted_class': int(predicted_class),  # Convert to native int
            'predicted_label': predicted_label,
            'confidence': float(np.max(predictions[0]))  # Convert to native float
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})
