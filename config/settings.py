from prodict import Prodict
from pathlib import Path


REPO_PATH = Path(__file__).parent.parent.absolute()


class PROCESSING(Prodict):
    original_audio_folder: Path = REPO_PATH / Path("data/original_audio/")
    split_audio_folder: Path = REPO_PATH / Path("data/train_data/audio/")
    tensors_folder: Path = REPO_PATH / Path("data/train_data/tensors/")

    sampling_rate: int = 44100
    win_length: int = 1022
    n_fft: int = 1022

    log_scale: bool = True
    log_base: int = 10
    num_samples: int = None

    slice_width = 256
    test_sample_length_seconds: int = 10


class MODEL(Prodict):
    input_shape: tuple = (PROCESSING.n_fft // 2 + 1, PROCESSING.slice_width, 2)
    code_size: int = 256

    optimizer: str = 'Adam'
    loss: str = 'mean_squared_error'
