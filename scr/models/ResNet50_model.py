import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense, Activation, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import Model, Input, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_and_load_the_model():
  base_model = tf.keras.applications.ResNet50(
    input_shape=(48, 48, 3),
    include_top=False,
    weights=None
)
  model = Sequential()
  model.add(Conv2D(3, input_shape=(48, 48, 1), kernel_size=(1, 1), activation='relu'))
  model.add(base_model)
  model.add(GlobalAveragePooling2D())
  model.add(Flatten())
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.65))
  model.add(Dense(128,kernel_initializer='he_uniform'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dense(7,activation='softmax'))
  return model