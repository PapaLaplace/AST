import numpy as np
from scipy.io import wavfile
from scipy import interpolate
from librosa.core.spectrum import stft, istft
from librosa import fft_frequencies
from tqdm import tqdm
import json
import os


from config.settings import PROCESSING


def log_scale_freq(freq, log_base, num_samples):
    if num_samples is None:
        num_samples = len(freq)

    start, stop = np.log(freq[1]) / np.log(log_base), np.log(freq[-1]) / np.log(log_base)
    freq_logspace = np.logspace(start=start, stop=stop, num=num_samples - 1, base=log_base)
    new_freq = np.append((0,), np.log(freq_logspace) / np.log(log_base))

    return new_freq, freq_logspace


def log_scale_at_t(freq, log_base, num_samples):
    new_freq, freq_logspace = log_scale_freq(freq, log_base, num_samples)

    def _log_scale_at_t(energy_at_t):
        f = interpolate.interp1d(freq, energy_at_t)
        new_energy_at_t = f(freq_logspace)

        return np.append(new_energy_at_t, energy_at_t[-1])
    return new_freq, _log_scale_at_t


def log_scale(freq, spectr, log_base=10, num_samples=None, return_freq=False):
    new_freq, log_scaler = log_scale_at_t(freq, log_base, num_samples)
    new_spectr = np.apply_along_axis(log_scaler, 0, spectr)

    if return_freq:
        return new_freq, new_spectr
    else:
        return new_spectr


def inverse_log_scale_at_t(freq, log_base, num_samples):
    start, stop = np.power(log_base, freq[1]), np.power(log_base, freq[-1])
    linear_freq = np.power(log_base, freq)
    freq_linspace = np.linspace(start=start, stop=stop, num=num_samples - 1)
    new_freq = np.append((0,), freq_linspace)

    def _inverse_log_scale_at_t(energy_at_t):
        f = interpolate.interp1d(linear_freq, energy_at_t)
        new_energy_at_t = f(freq_linspace)

        return np.append(new_energy_at_t, energy_at_t[-1])
    return new_freq, _inverse_log_scale_at_t


def inverse_log_scale(freq, spectr, log_base=10, num_samples=None, return_freq=False):
    if num_samples is None:
        num_samples = len(freq)

    new_freq, inverse_log_scaler = inverse_log_scale_at_t(freq, log_base, num_samples)
    new_spectr = np.apply_along_axis(inverse_log_scaler, 0, spectr)

    if return_freq:
        return new_freq, new_spectr
    else:
        return new_spectr


def wav_to_spectrogram(filename, debug=False):
    rate, sig = wavfile.read(filename)

    if debug:
        print(f"signal shape: {sig.shape}")
        print(f"sample rate: {rate}\n")

    def get_spectrogram(n_fft=PROCESSING.n_fft, win_length=PROCESSING.win_length, return_freq=False, **kwargs):
        spectr = stft(sig, n_fft=n_fft, win_length=win_length, **kwargs)

        if debug:
            print(f"spectrogram shape: {spectr.shape}\n")

        if return_freq:
            freq = fft_frequencies(sr=rate, n_fft=n_fft)
            return freq, spectr
        else:
            return spectr
    return get_spectrogram


def spectrogram_to_wav(spectr, rate=PROCESSING.sampling_rate, debug=False):
    if debug:
        print(f"spectrogram shape: {spectr.shape}\n")

    def get_wav(n_fft=PROCESSING.n_fft, win_length=PROCESSING.win_length, **kwargs):
        sig = istft(spectr, n_fft=n_fft, win_length=win_length, **kwargs)

        if debug:
            print(f"signal shape: {sig.shape}")
            print(f"sample rate: {rate}\n")

        return sig, rate
    return get_wav


def spectrogram_to_tensor(spectr, scalar=None, debug=False):
    m1 = np.abs(spectr)
    m2 = np.angle(spectr)

    scalar = np.max(m1) if scalar is None else scalar

    m1 = m1 / scalar
    m2 = (m2 + np.pi) / (2 * np.pi)

    if debug:
        print(f"m1 shape: {m1.shape}")
        print(f"m2 shape: {m2.shape}\n")

    return np.stack((m1, m2), axis=2), scalar


def tensor_to_spectrogram(tensor, scalar, debug=False):
    m1 = tensor[:, :, 0]
    m2 = tensor[:, :, 1]

    if debug:
        print(f"m1 shape: {m1.shape}")
        print(f"m2 shape: {m2.shape}\n")

    m1 = m1 * scalar
    m2 = np.pi * (2 * m2 - 1)

    return m1 * (np.cos(m2) + 1j * np.sin(m2))


def split_audio(original_fname, target_fname,
                input_folder=PROCESSING.original_audio_folder,
                output_folder=PROCESSING.split_audio_folder,
                test_length_sec=PROCESSING.test_sample_length_seconds,
                train_percentage_to_use=1):
    x_rate, x = wavfile.read(input_folder / original_fname)
    y_rate, y = wavfile.read(input_folder / target_fname)

    if x_rate != y_rate:
        raise Exception("Sampling rates aren't equal for the original and the target audio")
    else:
        rate = x_rate

    test_slice = int(test_length_sec * rate)

    x_train, x_test = x[:-test_slice], x[-test_slice:]
    y_train, y_test = y[:-test_slice], y[-test_slice:]

    x_train_l_slice = x_train[:int(len(x_train) * train_percentage_to_use / 2)]
    x_train_r_slice = x_train[-int(len(x_train) * train_percentage_to_use / 2):]
    x_train = np.concatenate((x_train_l_slice, x_train_r_slice), axis=0)

    y_train_l_slice = y_train[:int(len(y_train) * train_percentage_to_use / 2)]
    y_train_r_slice = y_train[-int(len(y_train) * train_percentage_to_use / 2):]
    y_train = np.concatenate((y_train_l_slice, y_train_r_slice), axis=0)

    wavfile.write(output_folder / "x_train.wav", rate, x_train)
    wavfile.write(output_folder / "x_test.wav", rate, x_test)
    wavfile.write(output_folder / "y_train.wav", rate, y_train)
    wavfile.write(output_folder / "y_test.wav", rate, y_test)


def slice_tensor(tensor, batch_width):
    indices = range(batch_width, tensor.shape[1] - (tensor.shape[1] % batch_width), batch_width)
    slices = np.split(tensor, indices, axis=1)[:-1]
    slices = np.stack(slices, axis=0)

    return slices


def get_split_tensors(input_folder=PROCESSING.split_audio_folder,
                      output_folder=PROCESSING.tensors_folder,
                      slice_width=PROCESSING.slice_width,
                      win_length=PROCESSING.win_length,
                      n_fft=PROCESSING.n_fft):
    max_abs = {}

    for fname in tqdm(['x_train.wav', 'y_train.wav']):
        freq, sg = wav_to_spectrogram(input_folder / fname)(win_length=win_length, n_fft=n_fft, return_freq=True)
        scaled_sg = log_scale(freq, sg)
        tensor, scalar = spectrogram_to_tensor(scaled_sg)

        max_abs[fname.split('_')[0]] = scalar

        slices = slice_tensor(tensor, slice_width)
        np.save(os.path.join(output_folder, f"{fname.split('.')[0]}.npy"), slices)

    for fname in tqdm(['x_test.wav', 'y_test.wav']):
        freq, sg = wav_to_spectrogram(input_folder / fname)(win_length=win_length, n_fft=n_fft, return_freq=True)
        scaled_sg = log_scale(freq, sg)
        tensor, _ = spectrogram_to_tensor(scaled_sg, max_abs[fname.split('_')[0]])

        slices = slice_tensor(tensor, slice_width)
        np.save(os.path.join(output_folder, f"{fname.split('.')[0]}.npy"), slices)

    with open(output_folder / "scalars.json", 'w') as f:
        json.dump(max_abs, f)


def load_split_tensors(input_folder):
    x_train = np.load(input_folder / "x_train.npy")
    y_train = np.load(input_folder / "y_train.npy")

    x_test = np.load(input_folder / "x_test.npy")
    y_test = np.load(input_folder / "y_test.npy")

    with open(input_folder / "scalars.json", 'r') as f:
        scalars = json.load(f)

    return x_train, y_train, x_test, y_test, scalars


def batch_to_tensor(batch):
    split_batch = np.split(batch, batch.shape[0], axis=0)
    squeezed_batch = [np.squeeze(p, axis=0) for p in split_batch]

    tensor = np.concatenate(squeezed_batch, axis=1)

    return tensor
