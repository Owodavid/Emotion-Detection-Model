from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sqlite3
import os

app = Flask(__name__)
model = load_model('face_emotionModel.h5')
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Database connection
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save image temporarily
            image_path = os.path.join('static', file.filename)  # Create 'static' folder if needed
            file.save(image_path)
            
            # Process image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            faces = face_cascade.detectMultiScale(img, 1.3, 5)
            
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face = cv2.resize(img[y:y+h, x:x+w], (48, 48))
                face = np.expand_dims(face, axis=(0, -1)) / 255.0
                pred = model.predict(face)
                emotion = emotions[np.argmax(pred)]
                
                # Store in DB (assuming a dummy user_id=1 for now; add auth later)
                # Inside POST, after predicting emotion
# --- REPLACE THIS BLOCK ---
conn = get_db_connection()
cursor = conn.cursor()

# Get user ID for 'testuser'
cursor.execute("SELECT id FROM users WHERE username = ?", ('testuser',))
row = cursor.fetchone()
user_id = row[0] if row else 1

# Save image + emotion
                conn.execute(
                    "INSERT INTO images (user_id, image_path, predicted_emotion) VALUES (?, ?, ?)",
                    (user_id, image_path, emotion)
                    )
                conn.commit()
                conn.close()
# --- END REPLACE ---
                
                return render_template('index.html', emotion=emotion, image_path=image_path)
    
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)  # For image uploads
    app.run(debug=True)