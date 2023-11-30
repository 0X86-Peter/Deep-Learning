#import all the lib we need in CV deep learning
import tensorflow as tf
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from keras import *
import matplotlib.pyplot as plt
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
import os

if __name__ == '__main__':
    # Data preparation
    dataset_dir = r'F:\资料\DeepLearning\crc_skin_data\crc_skin_data'#read the picture from computer
    input_shape = (224, 224, 3)#specify picture size
    batch_size = 8
    num_classes = 2

    # Data augmentation:do rescale、shear、zoom for picture
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(
        rescale=1. / 255)
    # Generate the train dataset, validation dataset and test dataset
    train_generator = train_datagen.flow_from_directory(
        os.path.join(dataset_dir, 'train'),
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='binary'  # skin has two classes: benign and malignant
    )

    validation_generator = train_datagen.flow_from_directory(
        os.path.join(dataset_dir, 'train'),
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='binary'  # skin has two classes: benign and malignant
    )

    test_generator = test_datagen.flow_from_directory(
        os.path.join(dataset_dir, 'test'),
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='binary'  # skin has two classes: benign and malignant
    )
    # building the model: AlexNet: 8 layers - 5 Conv layers and 3 Fully-connected (FC) layers
    model = Sequential()

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    # Add the fully connected layers
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # compile the model
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    # Train the model
    history = model.fit(train_generator, epochs=10, validation_data=validation_generator, verbose=1)
    # Training accuracy and validation accuracy graph
    plt.figure(figsize=(8, 8))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')
    plt.show()
