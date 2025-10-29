# Copyright 2025 Sony AI

import numpy as np
import librosa
import random


class SpectralFeature(object):
    def __init__(self, wav=None, fft_size=None, stft_hop_size=None, center=None, config=None):
        self._wav_shape = wav.shape
        self._fft_size = fft_size
        self._stft_hop_size = stft_hop_size
        self._center = center
        self._config = config

        self._wav_ch = wav.shape[1]
        wav_c_contiguous_example = np.require(wav[:, 0], dtype=np.float32, requirements=['C'])
        spec_example = librosa.core.stft(wav_c_contiguous_example,
                                         n_fft=fft_size,
                                         hop_length=stft_hop_size,
                                         center=center)  # used for num_frame
        self._num_bin = int(fft_size / 2) + 1
        self._num_frame = spec_example.shape[1]
        self._complex_spec = np.ones((self._wav_ch, self._num_bin, self._num_frame), dtype='complex64')

        self._complex_spec[0] = spec_example
        for i in range(1, self._wav_ch):
            wav_c_contiguous = np.require(wav[:, i], dtype=np.float32, requirements=['C'])
            self._complex_spec[i] = librosa.core.stft(wav_c_contiguous,
                                                      n_fft=fft_size,
                                                      hop_length=stft_hop_size,
                                                      center=center)

    def get_istft_wav(self):
        self._istft_wav = np.zeros(self._wav_shape)
        for i in range(0, self._wav_ch):
            self._istft_wav[:, i] = librosa.core.istft(self._complex_spec[i],
                                                       hop_length=self._stft_hop_size,
                                                       center=self._center,
                                                       dtype=np.float32,
                                                       length=self._wav_shape[0])
        return self._istft_wav

    def amplitude(self):
        amp = np.zeros((self._wav_ch, self._num_bin, self._num_frame), dtype='float32')
        for i in range(self._wav_ch):
            amp[i] = np.abs(self._complex_spec[i])
        return amp

    def phasediff(self):
        phasediff = np.zeros((self._wav_ch - 1, self._num_bin, self._num_frame), dtype='float32')
        wav_ch_base = self._config["base_channel"]
        spec_angle_base = np.angle(self._complex_spec[wav_ch_base])
        ch_wo_base = np.delete(np.arange(self._wav_ch), wav_ch_base)
        for enu_i, i in enumerate(ch_wo_base):
            spec_angle = np.angle(self._complex_spec[i]) - spec_angle_base
            spec_angle[spec_angle < 0] += 2 * np.pi
            phasediff[enu_i] = spec_angle
        return phasediff

    def eqda(self, fs=24000, H=None):  # hard coding
        if H is None:
            n_fft = self._fft_size

            f0 = 300 * np.random.rand() + 100
            g = 12 * np.random.rand() - 6
            q = 4 * np.random.rand() + 0.7071
            mpf = MPF(f0, g, q, fs=fs)
            H = mpf.calc_freq_char(n_fft)

            f0 = 1600 * np.random.rand() + 440
            g = 12 * np.random.rand() - 6
            q = 4 * np.random.rand() + 0.7071
            mpf = MPF(f0, g, q, fs=fs)
            H2 = mpf.calc_freq_char(n_fft)

            f0 = 6000 * np.random.rand() + 2200
            g = 12 * np.random.rand() - 6
            q = 4 * np.random.rand() + 0.7071
            mpf = MPF(f0, g, q, fs=fs)
            H3 = mpf.calc_freq_char(n_fft)

            H = H * H2 * H3

        self._complex_spec = self._complex_spec * H[:, np.newaxis]

        return H

    def rotate_foa(self):  # hard coding
        azi_reflection_list = [1, -1]
        azi_rotation_list = [0, np.pi]
        # azi_rotation_list = [-np.pi / 2, 0, np.pi / 2, np.pi]
        ele_reflection_list = [1, -1]
        azi_reflection = random.choice(azi_reflection_list)
        azi_rotation = random.choice(azi_rotation_list)
        ele_reflection = random.choice(ele_reflection_list)

        no_rotated_complex_spec = self._complex_spec
        # X is wav[3, :, :], Y is wav[1, :, :]
        if [azi_reflection, azi_rotation] == [1, 0]:
            self._complex_spec[3, :, :] = no_rotated_complex_spec[3, :, :]
            self._complex_spec[1, :, :] = no_rotated_complex_spec[1, :, :]
        elif [azi_reflection, azi_rotation] == [1, np.pi]:
            self._complex_spec[3, :, :] = -no_rotated_complex_spec[3, :, :]
            self._complex_spec[1, :, :] = -no_rotated_complex_spec[1, :, :]
        elif [azi_reflection, azi_rotation] == [-1, 0]:
            self._complex_spec[3, :, :] = no_rotated_complex_spec[3, :, :]
            self._complex_spec[1, :, :] = -no_rotated_complex_spec[1, :, :]
        elif [azi_reflection, azi_rotation] == [-1, np.pi]:
            self._complex_spec[3, :, :] = -no_rotated_complex_spec[3, :, :]
            self._complex_spec[1, :, :] = no_rotated_complex_spec[1, :, :]
        elif [azi_reflection, azi_rotation] == [1, -np.pi / 2]:
            self._complex_spec[3, :, :] = no_rotated_complex_spec[1, :, :]
            self._complex_spec[1, :, :] = -no_rotated_complex_spec[3, :, :]
        elif [azi_reflection, azi_rotation] == [1, np.pi / 2]:
            self._complex_spec[3, :, :] = -no_rotated_complex_spec[1, :, :]
            self._complex_spec[1, :, :] = no_rotated_complex_spec[3, :, :]
        elif [azi_reflection, azi_rotation] == [-1, -np.pi / 2]:
            self._complex_spec[3, :, :] = -no_rotated_complex_spec[1, :, :]
            self._complex_spec[1, :, :] = -no_rotated_complex_spec[3, :, :]
        elif [azi_reflection, azi_rotation] == [-1, np.pi / 2]:
            self._complex_spec[3, :, :] = no_rotated_complex_spec[1, :, :]
            self._complex_spec[1, :, :] = no_rotated_complex_spec[3, :, :]
        # Z is wav[2, :, :]
        if ele_reflection == 1:
            self._complex_spec[2, :, :] = no_rotated_complex_spec[2, :, :]
        elif ele_reflection == -1:
            self._complex_spec[2, :, :] = -no_rotated_complex_spec[2, :, :]

        return [azi_reflection, azi_rotation, ele_reflection]


class PEQ(object):
    def __init__(self, f0, g=0, Q=0.7071, fs=16000):
        self.coef = np.zeros(5, dtype=np.float32)
        self.fs = fs

    def calc_freq_char(self, n_fft):
        w = 2 * np.pi * np.arange(n_fft // 2 + 1) / n_fft
        emjw = np.exp(-1j * w)
        emj2w = np.exp(-2j * w)
        self.H = (self.coef[0] + self.coef[1] * emjw + self.coef[2] * emj2w) / (1. - self.coef[3] * emjw - self.coef[4] * emj2w)
        return self.H


class MPF(PEQ):
    def __init__(self, f0, g, Q=0.7071, fs=16000):
        super().__init__(f0=f0, g=g, Q=Q, fs=fs)
        self.calc_param(f0, g, Q, fs)

    def calc_param(self, f0, g, Q=0.7071, fs=16000):
        w0 = 2 * np.pi * f0 / fs
        w1 = w0 * (-1 + np.sqrt(1 + 4 * Q * Q)) / (2 * Q)
        dw = 2 * np.arctan((np.cos(w1) - np.cos(w0)) / np.sin(w1))
        K = 10**(g / 20) - 1.
        t = np.tan(0.5 * dw)
        c = np.cos(w0)
        if K >= 0:
            b = (1 - t) / (1 + t)
        else:
            b = ((1 + K) - t) / ((1 + K) + t)
        self.coef[0] = 1 + 0.5 * K * (1 - b)
        self.coef[1] = -(1 + b) * c
        self.coef[2] = b - 0.5 * K * (1 - b)
        self.coef[3] = -self.coef[1]
        self.coef[4] = -b
