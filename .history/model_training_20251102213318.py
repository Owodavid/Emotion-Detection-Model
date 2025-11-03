import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Load dataset
data = pd.read_csv('data/dataset.csv')  # Assuming FER2013 format

# Preprocess
def preprocess_data(df):
    pixels = df['pixels'].apply(lambda x: np.fromstring(x, dtype=int, sep=' '))
    images = np.array([pixel.reshape(48, 48, 1) for pixel in pixels]) / 255.0  # Normalize
    labels = to_categorical(df['emotion'], num_classes=7)
    return images, labels

# Split data
train_df = data[data['Usage'] == 'Training']
test_df = data[data['Usage'] == 'PublicTest']  # Or use PrivateTest for final eval

X_train, y_train = preprocess_data(train_df)
X_test, y_test = preprocess_data(test_df)

# Build model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotions
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, epochs=25, batch_size=64, validation_data=(X_test, y_test))

# Plot accuracy
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()

# Save model
model.save('face_emotionModel.h5')
print("Model trained and saved as face_emotionModel.h5")