from functools import partial

import librosa
import numpy as np
import scipy


class Sequential:
    def __init__(self, *args):
        self.transforms = args

    def __call__(self, inp: np.ndarray):
        res = inp
        for transform in self.transforms:
            res = transform(res)
        return res


class Windowing:
    def __init__(self, window_size=1024, hop_length=None):
        self.window_size = window_size
        self.hop_length = hop_length if hop_length else self.window_size // 2
    
    def __call__(self, waveform):
        wav_length = waveform.shape[0]
        new_waveform = np.pad(waveform, (self.window_size//2, self.window_size//2))
        
        num_windows = (wav_length - self.window_size % 2) // self.hop_length + 1
        
        windows = np.zeros((num_windows, self.window_size), dtype=waveform.dtype)
        
        for idx in range(num_windows):
            start = idx * self.hop_length
            end = start + self.window_size
            windows[idx] = new_waveform[start: end]
        
        return windows
    

class Hann:
    def __init__(self, window_size=1024):
        self.taps = scipy.signal.windows.hann(window_size, sym=False)
    
    def __call__(self, windows):
        return windows * self.taps



class DFT:
    def __init__(self, n_freqs=None):
        self.n_freqs = n_freqs

    def __call__(self, windows):
        spec = np.abs(np.fft.rfft(windows))
        if self.n_freqs is not None:
            spec = spec[:, :self.n_freqs]
            
        return spec


class Square:
    def __call__(self, array):
        return np.square(array)


class Mel:
    def __init__(self, n_fft, n_mels=80, sample_rate=22050):
        self.mel_matrix = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=1, fmax=8192, htk=False)
        self.mel_matrix_pinv = np.linalg.pinv(self.mel_matrix)


    def __call__(self, spec):
        return spec @ self.mel_matrix.T

    def restore(self, mel):
        return mel @ self.mel_matrix_pinv.T


class GriffinLim:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None):
        self.griffin_lim = partial(
            librosa.griffinlim,
            n_iter=32,
            hop_length=hop_length,
            win_length=window_size,
            n_fft=window_size,
            window='hann'
        )

    def __call__(self, spec):
        return self.griffin_lim(spec.T)


class Wav2Spectrogram:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None):
        self.windowing = Windowing(window_size=window_size, hop_length=hop_length)
        self.hann = Hann(window_size=window_size)
        self.fft = DFT(n_freqs=n_freqs)
        # self.square = Square()
        self.griffin_lim = GriffinLim(window_size=window_size, hop_length=hop_length, n_freqs=n_freqs)

    def __call__(self, waveform):
        return self.fft(self.hann(self.windowing(waveform)))

    def restore(self, spec):
        return self.griffin_lim(spec)


class Wav2Mel:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None, n_mels=80, sample_rate=22050):
        self.wav_to_spec = Wav2Spectrogram(
            window_size=window_size,
            hop_length=hop_length,
            n_freqs=n_freqs)
        self.spec_to_mel = Mel(
            n_fft=window_size,
            n_mels=n_mels,
            sample_rate=sample_rate)

    def __call__(self, waveform):
        return self.spec_to_mel(self.wav_to_spec(waveform))

    def restore(self, mel):
        return self.wav_to_spec.restore(self.spec_to_mel.restore(mel))


class TimeReverse:
    def __call__(self, mel):
        return mel[::-1, :].copy()

class Loudness:
    def __init__(self, loudness_factor):
        self.loudness_factor = loudness_factor

    def __call__(self, mel):
        return mel * self.loudness_factor




class PitchUp:
    def __init__(self, num_mels_up):
        self.num_mels_up = num_mels_up


    def __call__(self, mel):
        res = np.roll(mel, shift=self.num_mels_up, axis=-1)
        res[:, :self.num_mels_up] = 0
        return res



class PitchDown:
    def __init__(self, num_mels_down):
        self.num_mels_down = num_mels_down


    def __call__(self, mel):
        res = np.roll(mel, shift=-self.num_mels_down, axis=-1)
        res[:, -self.num_mels_down:] = 0
        return res


# class SpeedUpDown:
#     def __init__(self, speed_up_factor=1.0):
#         self.speed_up_factor = speed_up_factor

#     def __call__(self, mel):
#         num_frames, num_features = mel.shape
#         new_num_frames = int(num_frames / self.speed_up_factor)
#         new_mel = np.zeros((new_num_frames, num_features), dtype=mel.dtype)
#         idx = np.arange(num_frames)
#         new_idx = np.clip(np.round(idx / self.speed_up_factor).astype(np.int64), 0, new_num_frames - 1)
#         new_mel[new_idx] = mel
        
#         return new_mel
class SpeedUpDown:
    def __init__(self, speed_up_factor=1.0):
        self.speed_up_factor = speed_up_factor

    def __call__(self, mel):
        indices = np.linspace(0, mel.shape[0] - 1, int(self.speed_up_factor * mel.shape[0]))
        indices = np.round(indices).astype("int")
        return mel[indices, :]



class FrequenciesSwap:
    def __call__(self, mel):
        return mel[:, ::-1].copy()



class WeakFrequenciesRemoval:
    def __init__(self, quantile=0.05):
        self.quantile = quantile


    def __call__(self, mel):
        thres = np.quantile(mel, self.quantile)
        new_mel = np.where(mel < thres, 0, mel)
        return new_mel



class Cringe1:
    def __init__(self):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^



class Cringe2:
    def __init__(self):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^

