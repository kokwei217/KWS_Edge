import librosa
import os.path
import sys
import random
import numpy as np
import math
from glob import glob
import tensorflow as tf
import json
from skimage.transform import resize

"""AudioDataset Properties
  1) word_list:
        wanted word list [ silence, background, ...10 keywords]
  2) word_to_index:
        Map of all words in dataset and their respective index in our wanted_words_list
  3) data_index:
        Map of {validation:[], testing:[], training: []}
        the array consists of mapping of label and file path
        eg: [ { label : <label>, file: <filepath>}, .... ]
  4)
  """
SR = 16000
JSON_TRAIN_PATH = "data_train.json"
JSON_VALID_PATH = "data_valid.json"
JSON_TEST_PATH = "data_test.json"

data_train_np = {
    "X_training": [],
    "y_training": []
}

data_test_np = {
    "X_testing": [],
    "y_testing": [],
}

data_validation_np = {
    "X_validation": [],
    "y_validation": [],
}


def get_one_noise(background_noises):
    """generates one single noise clip"""
    selected_noise = background_noises[random.randint(
        0, len(background_noises) - 1)]
    # only takes out 16000
    start_idx = random.randint(0, len(selected_noise) - 1 - SR)
    return selected_noise[start_idx:(start_idx + SR)]


def get_mix_noises(background_noises, num_noise=2, max_ratio=2):
    result = np.zeros(SR)
    for _ in range(num_noise):
        result += random.random() * max_ratio * get_one_noise(background_noises)
    return result / num_noise if num_noise > 0 else result


def add_noise_to_wav(wav, noise, SNR_MIN=0, SNR_MAX=40):
    SNR = random.choice([x for x in range(SNR_MIN, SNR_MAX + 5, 5)])
    wav = np.interp(wav, (wav.min(), wav.max()), (-1, 1))
    noise = np.interp(noise, (noise.min(), noise.max()), (-1, 1))
    RMS_s = math.sqrt(np.mean(wav**2))
    RMS_n = math.sqrt(np.mean(noise**2))
    scale_factor = (RMS_s/RMS_n)*(pow(10, -SNR/20))
    wav = wav + noise*scale_factor
    return np.clip(wav, -1, 1)


def timeshift(wav, ms=100):
    shift = random.randint(-ms, ms)
    a = -min(0, shift)
    b = max(0, shift)
    data = np.pad(wav, (a, b), "constant")
    return data[:len(data) - a] if a else data[b:]


def pad_and_crop_frame(wav, sr):
    if len(wav) < sr:
        wav = np.pad(wav, (0, sr - len(wav)), 'constant')
    return wav[:sr]


def preprocess_mfcc(wave, sr):
    mfcc = librosa.feature.mfcc(
        y=wave, n_mfcc=40, sr=sr, hop_length=320, n_fft=640)
    return mfcc


def preprocess_mel(data, sr, n_mels=40, normalization=False):
    spectrogram = librosa.feature.melspectrogram(
        data, sr=sr, n_mels=n_mels, hop_length=320, n_fft=640, fmin=20, fmax=4000)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    if normalization:
        spectrogram = spectrogram.spectrogram()
        spectrogram -= spectrogram
    return spectrogram


class AudioProcessor(object):
    def __init__(self, audio_dataset, bg_noise_dir, bg_noise_train_dir, sr, input_type):
        self.bg_noise_dir = bg_noise_dir
        self.bg_noise_train_dir = bg_noise_train_dir
        self.audio_dataset = audio_dataset
        self.sr = sr
        self.input_type = input_type
        self.load_background_noises()
        self.preprocess_data(input_type)

    def load_background_noises(self):
        self.background_noises = [librosa.load(x, sr=self.sr)[0] for x in glob(
            self.bg_noise_dir + "*.wav")]
        self.background_noises_train = [librosa.load(
            x, sr=self.sr)[0] for x in glob(self.bg_noise_train_dir + "*.wav")]

    def get_silent_wav(self, num_noise=2, max_ratio=2):
        return get_mix_noises(self.background_noises, num_noise=num_noise, max_ratio=max_ratio)

    def augment_train_wav(self, wav):
        noise = get_one_noise(self.background_noises_train)
        wav = timeshift(add_noise_to_wav(wav, noise))
        return wav

    def preprocess_data(self, type):
        self.processed_dataset = {
            "X_training": [],
            "X_validation": [],
            "X_testing": [],
            "y_training": [],
            "y_validation": [],
            "y_testing": [],
        }
        data_index = self.audio_dataset.data_index
        word_to_index = self.audio_dataset.word_to_index
        word_list = self.audio_dataset.words_list
        for ds in ["training", "validation", "testing"]:
            print("processing wav file... currently at " + ds)
            i = 0
            for data in data_index[ds]:
                i+=1 
                if i> 10:break
                #label = word_list[word_to_index[data["label"]]]
                label = word_to_index[data["label"]]

                wav = librosa.load(data["file"], self.sr)[0] if label != "silence" \
                    else self.get_silent_wav(
                    num_noise=random.choice([0, 1, 2]),
                    max_ratio=random.choice([x / 10. for x in range(20)])
                )
                wav = pad_and_crop_frame(wav, self.sr)
                if ds == "training" and label != "silent":
                    wav = self.augment_train_wav(wav)
                if type == "mfcc":
                    wav_np = preprocess_mfcc(wav, self.sr)
                elif type == "logmel":
                    wav_np = preprocess_mel(wav, self.sr)
                # if resize will be very huge inferencing task
                # wav_np = resize(
                #     wav_np, (64, 64), preserve_range=True)

                wav_key = "X_" + ds
                label_key = "y_" + ds
                self.processed_dataset[wav_key].append(wav_np)
                self.processed_dataset[label_key].append(label)
                # json file
                # if ds == "training":
                #     data_train_np[wav_key].append(wav_np.T.tolist())
                #     data_train_np[label_key].append(label)
                # elif ds == "validation":
                #     data_validation_np[wav_key].append(wav_np.T.tolist())
                #     data_validation_np[label_key].append(label)
                # else:
                #     data_test_np[wav_key].append(wav_np.T.tolist())
                #     data_test_np[label_key].append(label)

    def get_processed_dataset(self, type):
        processed_dataset = self.processed_dataset
        X_train = np.array(processed_dataset["X_training"])
        X_validation = np.array(processed_dataset["X_validation"])
        X_test = np.array(processed_dataset["X_testing"])
        y_train = np.array(processed_dataset["y_training"])
        y_validation = np.array(processed_dataset["y_validation"])
        y_test = np.array(processed_dataset["y_testing"])
        # with open(JSON_TRAIN_PATH, "w") as fp:
        #     json.dump(data_train_np, fp, indent=4)
        # with open(JSON_VALID_PATH, "w") as fp:
        #     json.dump(data_validation_np, fp, indent=4)
        # with open(JSON_TEST_PATH, "w") as fp:
        #     json.dump(data_test_np, fp, indent=4)
        return (X_train, X_validation, X_test, y_train, y_validation, y_test)
