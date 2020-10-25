import numpy as np
from tensorflow import keras
import tensorflowjs as tfjs

resolution = 28


def load_data(path):
    with open(path) as file:
        lines = file.read().split('\n')

    numbers = []
    bitmaps = []

    for i in range(len(lines)):
        if i % 2 == 0:
            numbers.append([1 if x == int(lines[i]) else 0 for x in range(10)])
        else:
            line = lines[i]
            bitmap = []
            for j in range(resolution):
                start = resolution * j
                end = resolution * (j + 1)
                row = [[int(bit)] for bit in line[start:end]]
                bitmap.append(row)
            bitmaps.append(bitmap)

    numbers = np.array(numbers)
    bitmaps = np.array(bitmaps)

    return numbers, bitmaps


numbers, bitmaps = load_data('bitmaps.txt')

print("The shape of numbers: {}".format(numbers.shape))
print("The shape of bitmaps: {}".format(bitmaps.shape))

model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    keras.layers.Conv2D(16, (3, 3), activation='relu'),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

model.summary()

model.compile(
    optimizer='adam',
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(bitmaps, numbers, epochs=5)

tfjs.converters.save_keras_model(model, 'model')
