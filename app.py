from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained model (replace 'model.h5' with your model file)
model = load_model('model.h5')

# Define the list of emotions
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the image to the size expected by the model (48x48 for many models)
    resized = cv2.resize(gray, (48, 48))
    # Normalize the image
    normalized = resized / 255.0
    # Expand dimensions to match the input shape of the model
    reshaped = np.expand_dims(normalized, axis=0)
    reshaped = np.expand_dims(reshaped, axis=-1)
    return reshaped

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read the image file
    file_bytes = np.fromstring(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Predict the emotion
    predictions = model.predict(preprocessed_image)
    emotion_index = np.argmax(predictions)
    emotion = emotions[emotion_index]

    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True)