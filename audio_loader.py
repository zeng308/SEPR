import os
import os.path
import soundfile as sf
import librosa
import numpy as np
import torch
import torch.utils.data as data

AUDIO_EXTENSIONS = [
    '.wav', '.WAV',
]

spect_to_path_dict = dict()


def make_dataset(dir):
    noisy = []
    for root, dirs, files in os.walk(dir + '/noisy'):
        for filename in files[:1500]:
            noisy.append(root + '/' + filename)
    clean = []
    for root, dirs, files in os.walk(dir + '/clean'):
        for filename in files[:1500]:
            clean.append(root + '/' + filename)

    return list(zip(noisy, clean))


def spect_loader(path, window_size, window_stride, window, normalize, max_len=101, test=False):
    y, sr = sf.read(path)
    # y, sr = librosa.load(path, sr=None)
    # n_fft = 4096
    n_fft = int(sr * window_size)
    win_length = n_fft
    hop_length = int(sr * window_stride)

    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    spect, phase = librosa.magphase(D)

    # S = log(S+1)
    spect = np.log1p(spect)

    # make all spects with the same dims
    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
    elif spect.shape[1] > max_len:
        spect = spect[:, :max_len]
    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    spect = torch.FloatTensor(spect)

    # z-score normalization
    if normalize:
        mean = spect.mean()
        std = spect.std()
        if std != 0:
            spect.add_(-mean)
            spect.div_(std)

    # a map to map the original file to the spectrogram
    if test:
        spect_to_path_dict[spect] = path

    return spect


def spect_loader2(path, window_size, window_stride, window, normalize, max_len=101, test=False):
    y, sr = sf.read(path)

    # matching all the arrays to be 2 seconds
    if len(y) < sr:
        y = np.pad(y, (0, sr * 2 - len(y)), 'constant')
    elif len(y) > sr:
        y = y[:sr]

    y = y[::2]

    spect = torch.FloatTensor(y)

    # z-score normalization
    if normalize:
        mean = spect.mean()
        std = spect.std()
        if std != 0:
            spect.add_(-mean)
            spect.div_(std)

    # a map to map the original file to the spectrogram
    if test:
        spect_to_path_dict[spect] = path

    return spect


class AudioLoader(data.Dataset):
    """A google command data set loader where the wavs are arranged in this way: ::
        root/one/xxx.wav
        root/one/xxy.wav
        root/one/xxz.wav
        root/head/123.wav
        root/head/nsdf3.wav
        root/head/asd932_.wav
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        window_size: window size for the stft, default value is .02
        window_stride: window stride for the stft, default value is .01
        window_type: typye of window to extract the stft, default value is 'hamming'
        normalize: boolean, whether or not to normalize the spect to have zero mean and one std
        max_len: the maximum length of frames to use
     Attributes:
        classes (list): List of the class names.
        spects (list): List of (spects path, class_index) tuples
        STFT parameter: window_size, window_stride, window_type, normalize
    """

    def __init__(self, root, test_mode=False, transform=None, target_transform=None, window_size=.02,
                 window_stride=.01, window_type='hamming', normalize=True, max_len=101):
        spects = make_dataset(root)
        if len(spects) == 0:
            raise (RuntimeError("Found 0 sound files in subfolders of: " + root +
                                "Supported audio file extensions are: " + ",".join(AUDIO_EXTENSIONS)))

        self.root = root
        self.spects = spects
        self.transform = transform
        self.target_transform = target_transform
        self.loader = spect_loader2
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len
        self.test_mode = test_mode

    def __getitem__(self, index):
        path_noisy, path_clean = self.spects[index]

        spect_noisy = self.loader(path_noisy, self.window_size, self.window_stride, self.window_type,
                                  self.normalize, self.max_len, self.test_mode)
        if self.transform is not None:
            spect_noisy = self.transform(spect_noisy)
        spect_clean = self.loader(path_clean, self.window_size, self.window_stride, self.window_type,
                                  self.normalize, self.max_len, self.test_mode)
        if self.transform is not None:
            spect_clean = self.transform(spect_clean)

        return spect_noisy, spect_clean

    def __len__(self):
        return len(self.spects)

    # def back_to_audio():
