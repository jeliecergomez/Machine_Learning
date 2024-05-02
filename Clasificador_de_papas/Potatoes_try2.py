# Import necessary libraries
from flask import Flask, render_template, request
from keras.models import load_model
from PIL import Image
import numpy as np


# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model
model =load_model('/UCordoba/Machine_Learning/Potato_disease/potatoes.h5')

# Define function to preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((256, 256)) # Resize image to match model input shape
    img_array = np.array(img) / 255.0 # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    return img_array

# Define function to make prediction
def predict_image(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    classes = ["Early Blight", "Late Blight", "Healthy"]
    predicted_class = classes[np.argmax(prediction)]
    return predicted_class



# Define route for home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No selected file")
        if file:
            # Save uploaded image
            image_path = "/UCordoba/Machine_Learning/Potato_disease/static/" + file.filename
            file.save(image_path)
            # Make prediction
            prediction = predict_image(image_path)
            return render_template('index.html', prediction=prediction, filenamex=file.filename, imagepath=image_path)
    return render_template('index.html')

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
