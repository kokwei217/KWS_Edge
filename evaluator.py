import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import librosa
import IPython.display as ipd
from glob import glob
from preprocessor import preprocess_mfcc, preprocess_mel, pad_and_crop_frame


SAVED_MODEL_PATH = "trained_models/VGG16_model.h5"

SR = 16000


class _Keyword_Spotting_Service:
    """Singleton class for keyword spotting inference with trained models.

    :param model: Trained model
    """

    model = None
    _mapping = [
        "silence",
        "unknown",
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go"
    ]
    _instance = None

    def predict(self, file_path):
        """

        :param file_path (str): Path to audio file to predict
        :return predicted_keyword (str): Keyword predicted by the model
        """

        # extract MFCC
        wav, _ = librosa.load(file_path, sr=SR)
        wav = pad_and_crop_frame(wav, SR)
        ipd.Audio(file_path)
        MFCCs = preprocess_mfcc(wav, SR)
        # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]
        MFCCs = MFCCs.tolist()
        # get the predicted label
        num_top_predictions = 1
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mapping[predicted_index]
        # print(predictions)
        confidence = predictions[0][predicted_index]
        # top_k = predictions.argsort()[-num_top_predictions:][::-1]
        # for node_id in top_k:
        #     score = predictions[node_id]
        return predicted_keyword, confidence, predictions


def Keyword_Spotting_Service():
    """Factory function for Keyword_Spotting_Service class.

    :return _Keyword_Spotting_Service._instance (_Keyword_Spotting_Service):
    """

    # ensure an instance is created only the first time the factory function is called
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = tf.keras.models.load_model(
            SAVED_MODEL_PATH)
    return _Keyword_Spotting_Service._instance


if __name__ == "__main__":
    kss = Keyword_Spotting_Service()
    kss1 = Keyword_Spotting_Service()
    assert kss is kss1
    # model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    # print(model.summary(
    # print(tf.__version__)
    # make a prediction

    # for wav_path in glob(search_path):
    #     pass
    keyword, confidence , p= kss.predict(
        "C:/Users/kokwe/Desktop/silence.wav")
    print('%s (confidence = %.5f)' % (keyword, confidence))
    # print(p)

#big problem with silence!!