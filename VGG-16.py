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
        horizontal_flip=True,
        validation_split=0.2)

    test_datagen = ImageDataGenerator(
        rescale=1. / 255)
    # Generate the train dataset, validation dataset and test dataset
    train_generator = train_datagen.flow_from_directory(
        os.path.join(dataset_dir, 'train'),
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='binary',# skin has two classes: benign and malignant
        subset='training')

    validation_generator = train_datagen.flow_from_directory(
        os.path.join(dataset_dir, 'train'),
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='binary',  # skin has two classes: benign and malignant
        subset='validation')

    test_generator = test_datagen.flow_from_directory(
        os.path.join(dataset_dir, 'test'),
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='binary'  # skin has two classes: benign and malignant
    )

    # building the model: VGGNet 16: 16 layers - 13 Conv layers, 5 MaxPooling Layer and 3 Fully-connected (FC) layers
    model = Sequential()
    # Add the 13 convolutional layers and 5 maxpooling layer for VGG16

    model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the layers
    model.add(Flatten())

    # Add the fully connected layers
    model.add(Dense(4096, activation='relu'))  # first FC
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))  # second FC
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # third FC

    # compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    history = model.fit(train_generator, epochs=3, validation_data=(validation_generator), verbose=1)

    # Training accuracy and validation accuracy graph
    plt.figure(figsize=(8, 8))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')
    plt.show()

    # Trainig loss and validation loss graph
    plt.figure(figsize=(8, 8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
    plt.show()

    # Evaluate the model on the testing dataset
    test_loss, test_acc = model.evaluate(test_generator)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)