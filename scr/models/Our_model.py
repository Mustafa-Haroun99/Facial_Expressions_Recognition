import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense, Activation, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import Model, Input, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_and_load_the_model():
    cnn_model = Sequential()

    # 1st convolution layer
    cnn_model.add(Conv2D(64, input_shape=(48, 48, 1), kernel_size=(3, 3), activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Conv2D(64, padding='same', kernel_size=(3, 3), activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
    cnn_model.add(Dropout(0.3))

    # 2nd convolution layer
    cnn_model.add(Conv2D(128, padding='same', kernel_size=(3, 3), activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Conv2D(128, padding='same', kernel_size=(3, 3), activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
    cnn_model.add(Dropout(0.3))

    # 3rd convolution layer
    cnn_model.add(Conv2D(256, padding='same', kernel_size=(3, 3), activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Conv2D(256, padding='same', kernel_size=(3, 3), activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
    cnn_model.add(Dropout(0.3))

    # 4th convolution layer
    cnn_model.add(Conv2D(512, padding='same', kernel_size=(3, 3), activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Conv2D(512, padding='same', kernel_size=(3, 3), activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
    cnn_model.add(Dropout(0.3))

    # Fully connected layer
    cnn_model.add(Flatten())
    cnn_model.add(Dense(512, activation='relu'))
    cnn_model.add(Dropout(0.3))
    cnn_model.add(Dense(256, activation='relu'))
    cnn_model.add(Dropout(0.3))
    cnn_model.add(Dense(64, activation='relu'))
    cnn_model.add(Dropout(0.3))

    cnn_model.add(Dense(7, activation='softmax'))
    return cnn_model
