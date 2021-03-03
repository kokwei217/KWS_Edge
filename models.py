import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten


def simple_cnn(input_shape, num_layers=12):
    # build network architecture using convolutional layers
    model = keras.models.Sequential()

    # 1st conv layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape,
                            kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'))

    # 2nd conv layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'))

    # 3rd conv layer
    model.add(layers.Conv2D(32, (2, 2), activation='relu',
                            kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(
        (2, 2), strides=(2, 2), padding='same'))

    # flatten output and feed into dense layer
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    layers.Dropout(0.3)

    # softmax output layer
    model.add(layers.Dense(num_layers, activation='softmax'))
    return model


def cnn_tutorial(input_shape, X_train, num_layers=12):
    norm_layer = preprocessing.Normalization()
    norm_layer.adapt(X_train)
    model = keras.models.Sequential([
        layers.Input(shape=input_shape),
        # preprocessing.Resizing(32, 32),
        # norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_layers),
    ])
    return model


def VGG16(input_shape, num_layers=12):
    model = keras.models.Sequential()
    # first part
    model.add(Conv2D(input_shape=(224, 224, 3), filters=64,
                     kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # 2nd part
    model.add(Conv2D(filters=128, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # 3rd part
    model.add(Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # 4th part
    model.add(Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # 5th part
    model.add(Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # final
    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(layers.Dense(num_layers, activation='softmax'))
    return model
