import os
import csv
from scipy.ndimage import imread
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Cropping2D, Flatten, Dense, Lambda, Conv2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from math import ceil

model_checkpoint_callback = ModelCheckpoint(
                            filepath='checkpoint',
                            monitor = 'val_accuracy',
                            save_best_only=True)

early_stopping_callback = EarlyStopping(
                          monitor='val_loss',
                          min_delta=0.01,
                          patience=3,
                          restore_best_weights=True)
                          
lines = []
with open('/home/workspace/aditya/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

print("Lines", len(lines))
train_samples, validation_samples = train_test_split(lines, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        for offset in range(0, num_samples, batch_size):
            batch = samples[offset : offset + batch_size]
            images = []
            measurements = []
            for line in batch:
                path = line[0].split('\\')
                path = '/home/workspace/aditya/IMG/' + path[-1] 
                img = imread(path)
                images.append(img)
                measurement = float(line[3])
                measurements.append(measurement)
            x = np.array(images)
            y = np.array(measurements)
            yield (x, y)
            yield (np.flip(x, -2), y*(-1))

batch_size = 128
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50, 30), (0,0))))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=ceil(len(train_samples)/batch_size), 
                    validation_data=validation_generator, 
                    validation_steps=ceil(len(validation_samples)/batch_size), 
                    epochs=10, verbose=1, callbacks=[model_checkpoint_callback, early_stopping_callback])
          
model.save('model.h5')