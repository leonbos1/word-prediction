from sklearn.model_selection import train_test_split
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import os.path
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

plot_predictions = False

data_path = 'data'
categories = ['a', 'b', 'l', 'p', 'o', 'k', 's']

data = []
labels = []

for category in categories:
    category_path = os.path.join(data_path, category)
    class_num = categories.index(category)
    for img in os.listdir(category_path):
        try:
            img_path = os.path.join(category_path, img)
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            data.append(img_array)
            labels.append(class_num)
        except Exception as e:
            pass

data = np.array(data) / 255.0
data = np.reshape(data, (data.shape[0], 16, 16, 1))
labels = np.array(labels)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(16, 16, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(categories), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])


X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2)

model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

predictions = model.predict(X_test)

predicted_labels = np.argmax(predictions, axis=1)

predicted_labels = np.argmax(predictions, axis=1)

if plot_predictions:
    plt.figure(figsize=(10, 10))

    for i in range(len(X_test)):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_test[i].reshape(16, 16), cmap='gray')
        plt.xlabel(
            f"Predicted: {categories[predicted_labels[i]]}\nTrue: {categories[y_test[i]]}")
    plt.tight_layout()

    plt.show()

word_img_path = 'data/word.png'
word_image = cv2.imread(word_img_path, cv2.IMREAD_GRAYSCALE)
print(word_image)
height, width = word_image.shape

if width % 16 != 0:
    print("The width of the word image is not divisible by 16. Please ensure the image contains whole letters.")
    exit()

blocks = [word_image[:, 16*i:16*(i+1)] for i in range(width//16)]

word = ''
for block in blocks:
    block = block / 255.0
    block = block.reshape(1, 16, 16, 1)
    prediction = model.predict(block)
    predicted_label = np.argmax(prediction, axis=1)

    word += categories[predicted_label[0]]

print(f"Predicted word: {word}")