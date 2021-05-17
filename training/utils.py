import io
import numpy as np
from models import *
from tqdm import tqdm
from glob import iglob
from typing import Any, Tuple
from scipy.io.wavfile import read, write
from sklearn.base import BaseEstimator


def search_wav(data_path: str) -> list:
    """Searches for WAV files"""
    file_list = []
    for filename in iglob('{}/**/*.WAV'.format(data_path), recursive=True):
        file_list.append(str(filename))
    for filename in iglob('{}/**/*.wav'.format(data_path), recursive=True):
        file_list.append(str(filename))
    return file_list


def wav_to_np(file: str) -> Tuple[float, np.ndarray]:
    """Translate WAV file to numpy array"""
    with open(file, "rb") as wavfile:
        input_wav = wavfile.read()
        wavfile.close()
    rate, data = read(io.BytesIO(input_wav))
    return rate, data.astype(np.int16)


def np_to_wav(rate: float, data: np.ndarray, filename: str) -> None:
    """Translate numpy array to WAV file"""
    bytes_wav = bytes()
    byte_io = io.BytesIO(bytes_wav)
    write(byte_io, rate, data)
    output_wav = byte_io.read()
    with open(filename, "wb+") as wavfile:
        wavfile.write(output_wav)
        wavfile.close()


def load_samples_from_file(filename: str, window_size: int = 320) -> Tuple[list, np.ndarray]:
    """Loads WAV file into array of samples of window_size"""
    rates = []
    samples = []
    rate, data = wav_to_np(filename)
    if data.shape[0] > window_size:
        splits = data.shape[0] // window_size
        for i in range(splits):
            rates.append(rate)
            samples.append(data[i * window_size:(i+1) * window_size])
    else:
        rates.append(rate)
        samples.append(data)
    return rates, np.array(samples)


def generate_dataset(files: list) -> Tuple[list, np.ndarray]:
    """Generates dataset from list of files"""
    r, data = load_samples_from_file(files[0])
    for file in tqdm(files[1:]):
        rates, samples = load_samples_from_file(file)
        r += rates
        data = np.concatenate((data, samples), axis=0)
    return r, data.astype(np.int16)


def introduce_noise(arr: np.ndarray, loc: float = 0.0, scale: float = 1.0):
    """Introduces gaussian noise to an input array"""
    return arr + np.random.normal(loc=loc, scale=scale, size=arr.shape).astype(np.int16)


def introduce_noise_to_file(filename: str, loc: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """Introduces gaussian noise to an input file, returning its noisy array"""
    _, data = wav_to_np(filename)
    return introduce_noise(arr=data, loc=loc, scale=scale)


def clean_audio(data: np.ndarray, model: DenoisingAutoEncoder, scaler: Any, window_size: int = 320):
    """Gets noisy data and work it through the model, outputting a clean audio"""
    data = data.reshape((data.size // window_size, window_size))
    data = scaler.transform(data)
    encoded = model.encoder(data).numpy()
    decoded = model.decoder(encoded).numpy()
    clean_data: np.ndarray = scaler.inverse_transform(decoded)
    return clean_data.flatten().astype(np.int16)
