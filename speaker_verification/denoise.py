import os
import sys

import noisereduce as nr
import numpy as np
from scipy.io import wavfile


def denoise(input_file: str) -> (np.ndarray, int):
    """
    Denoise audio file using noisereduce.
    """

    # TODO: Read other audio types beside .wav
    # Read audio data
    rate, data = wavfile.read(input_file)
    data = data.astype(np.float64)

    # Section of data that is noisy.
    # Since we don't know, just select the whole data
    noise_clip = data

    # Perform noise reduction
    noise_reduced_data = nr.reduce_noise(audio_clip=data, noise_clip=noise_clip, verbose=False)

    return rate, noise_reduced_data


def store_wav_audio(wav_data: np.ndarray, rate: int, output_file: str):
    """
    Store WAV data into a .wav file.
    """

    wavfile.write(output_file, rate, wav_data.astype(np.int16))


if __name__ == '__main__':
    input_file = sys.argv[1]
    rate, clean_audio = denoise(input_file)

    output_folder = sys.argv[2]
    file_name = f"{os.path.basename(input_file).split('.')[0]}_cleaned.wav"
    output_file = os.path.join(output_folder, file_name)
    store_wav_audio(clean_audio, rate, output_file)
