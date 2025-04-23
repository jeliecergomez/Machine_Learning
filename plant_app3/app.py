from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
model = load_model('plant_disease_model.h5', compile=False)

# Asegúrate de tener las clases en el mismo orden que en el entrenamiento del modelo
class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
                   'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                   'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
                   'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
                   'Tomato_Spider_mites_Two_spotted_spider_mite',
                   'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
                   'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

def predict_image(img_path):
    img = load_img(img_path, target_size=(224, 224))  # ajusta si tu modelo requiere otro tamaño
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(100 * np.max(prediction), 2)
    return predicted_class, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            filename = secure_filename(image_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(filepath)

            predicted_class, confidence = predict_image(filepath)
            return render_template('index.html', image_path=filepath,
                                   prediction=predicted_class,
                                   confidence=confidence)
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True)
