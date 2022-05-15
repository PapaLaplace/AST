import numpy as np
import matplotlib.pyplot as plt


def plot_spectrogram(freq, spectr, mode='Abs'):
    modes = {'Abs': np.abs, 'Re': np.real, 'Im': np.imag, 'Arg': np.angle}

    plt.rcParams["figure.figsize"] = (15, 10)
    plt.pcolormesh(np.linspace(0, 1, num=spectr.shape[1]), freq, modes[mode](spectr), shading='gouraud')
    plt.title(f'STFT {mode}')
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.show()
