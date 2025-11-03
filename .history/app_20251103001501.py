import os
from flask import Flask, render_template, request, flash
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # For flash messages
app.config['UPLOAD_FOLDER'] = 'static'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Load model
try:
    model = load_model('face_emotionModel.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"ERROR loading model: {e}")
    model = None

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ensure static folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/', methods=['GET', 'POST'])
def index():
    emotion = None
    image_path = None

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded')
            return render_template('index.html')

        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return render_template('index.html')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            # Process image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                flash('Invalid image')
                return render_template('index.html')

            faces = face_cascade.detectMultiScale(img, 1.3, 5)

            if len(faces) == 0:
                emotion = 'No face detected'
            else:
                (x, y, w, h) = faces[0]
                face = cv2.resize(img[y:y+h, x:x+w], (48, 48))
                face = np.expand_dims(face, axis=(0, -1)) / 255.0
                if model:
                    pred = model.predict(face, verbose=0)
                    emotion = emotions[np.argmax(pred)]
                else:
                    emotion = 'Model not loaded'

            return render_template('index.html', emotion=emotion, image_path=image_path)
        else:
            flash('Invalid file type. Use PNG/JPG')

    return render_template('index.html', emotion=emotion, image_path=image_path)

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)