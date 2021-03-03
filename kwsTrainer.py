import argparse
import pathlib
import os.path
import sys
import numpy as np
import tensorflow as tf
import random
import dataset
import preprocessor
import json
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
import matplotlib.pyplot as plt

WANTED_WORDS = ["yes", "no", "up", "down",
                "left", "right", "on", "off", "stop", "go"]
DATA_DIR = "C:/SpeechCommandV2/"
SR = 16000
INPUT_TYPE = "mfcc"
LEARNING_RATE = 0.0001
EPOCHS = 40
BATCH_SIZE = 32
PATIENCE = 5
MODEL_SETTINGS = {}
RANDOM_SEED = 2177
SAVED_MODEL_PATH = "model.h5"
num_labels = 12
random.seed(RANDOM_SEED)

CWD = pathlib.Path(__file__).resolve().parent
BACKGROUND_NOISE_TRAIN_DIR = str(CWD) + '/background_noise_train/'
BACKGROUND_NOISE_SILENCE_DIR = str(CWD) + '/background_silence/'


def build_model(input_shape, X_train, loss="sparse_categorical_crossentropy", learning_rate=0.0001):
    """Build neural network using keras.

    :param input_shape (tuple): Shape of array representing a sample train. E.g.: (44, 13, 1)
    :param loss (str): Loss function to use
    :param learning_rate (float):

    :return model: TensorFlow model
    """
    # build network architecture using convolutional layers
    model = tf.keras.models.Sequential()

    # 1st conv layer
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape,
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'))

    # 2nd conv layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'))

    # 3rd conv layer
    model.add(tf.keras.layers.Conv2D(32, (2, 2), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(
        (2, 2), strides=(2, 2), padding='same'))

    # flatten output and feed into dense layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    tf.keras.layers.Dropout(0.3)

    # softmax output layer
    model.add(tf.keras.layers.Dense(12, activation='softmax'))

    optimiser = tf.optimizers.Adam(learning_rate=learning_rate)
    # todo remove above model
    # tf audio tutoorial model
    # norm_layer = preprocessing.Normalization()
    # norm_layer.adapt(X_train)
    # model = tf.keras.models.Sequential([
    #     layers.Input(shape=input_shape),
    #     # preprocessing.Resizing(32, 32),
    #     # norm_layer,
    #     layers.Conv2D(32, 3, activation='relu'),
    #     layers.Conv2D(64, 3, activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Dropout(0.25),
    #     layers.Flatten(),
    #     layers.Dense(128, activation='relu'),
    #     layers.Dropout(0.5),
    #     layers.Dense(num_labels),
    # ])
    # print model parameters on console
    model.summary()

    # compile model
    model.compile(optimizer=optimiser,
                  #   loss=loss,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=["accuracy"])

    return model


def train(model, input_shape, epochs, batch_size, patience, X_train, y_train, X_validation, y_validation):
    """Trains model

    :param epochs (int): Num training epochs
    :param batch_size (int): Samples per batch
    :param patience (int): Num epochs to wait before early stop, if there isn't an improvement on accuracy
    :param X_train (ndarray): Inputs for the train set
    :param y_train (ndarray): Targets for the train set
    :param X_validation (ndarray): Inputs for the validation set
    :param y_validation (ndarray): Targets for the validation set

    :return history: Training history
    """

    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor="accuracy", min_delta=0.001, patience=patience)

    # train model
    # input_shape = [40, 51, 1]
    X_train = X_train.reshape(
        len(X_train), input_shape[0], input_shape[1], input_shape[2])
    X_validation = X_validation.reshape(
        len(X_validation), input_shape[0], input_shape[1], input_shape[2])
    history = model.fit(X_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_validation, y_validation),
                        callbacks=[earlystop_callback])
    return history


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs

    :param history: Training history of model
    :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="accuracy")
    axs[0].plot(history.history['val_accuracy'], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")

    # create loss subplot
    axs[1].plot(history.history["loss"], label="loss")
    axs[1].plot(history.history['val_loss'], label="val_loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss evaluation")

    plt.show()


def main():
    audio_dataset = dataset.AudioDataset(
        data_dir=DATA_DIR,
        model_settings=MODEL_SETTINGS,
        wanted_words=WANTED_WORDS
    )
    audio_processor = preprocessor.AudioProcessor(
        audio_dataset,
        bg_noise_dir=BACKGROUND_NOISE_SILENCE_DIR,
        bg_noise_train_dir=BACKGROUND_NOISE_TRAIN_DIR,
        sr=SR,
        input_type=INPUT_TYPE
    )
    X_train, X_validation, X_test, y_train, y_validation, y_test = \
        audio_processor.get_processed_dataset("mfcc")
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(
        len(X_test), input_shape[0], input_shape[1], input_shape[2])
    model = build_model(input_shape, X_train, learning_rate=LEARNING_RATE)

    # train network
    history = train(model, input_shape, EPOCHS, BATCH_SIZE, PATIENCE,
                    X_train, y_train, X_validation, y_validation)

    # plot accuracy/loss for training/validation set as a function of the epochs
    plot_history(history)

    # evaluate network on test set

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("\nTest loss: {}, test accuracy: {}".format(test_loss, 100*test_acc))

    # save model
    model.save(SAVED_MODEL_PATH)


if __name__ == "__main__":
    main()
