from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Flatten, BatchNormalization, Activation, Dropout


def create_and_load_the_model():
    NN_model = Sequential([
        InputLayer(input_shape=(48, 48, 3)),
        Flatten(),
        Dense(units = 256),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        Dense(256),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        Dense(units=7, activation = 'softmax')
    ])
    NN_model.build((48, 48, 3))
    return NN_model