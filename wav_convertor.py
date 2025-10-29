# Copyright 2025 Sony AI

import numpy as np
import soundfile as sf
import json

from feature.feature import SpectralFeature


class WavConvertor(object):
    def __init__(self, args):
        self._args = args
        with open(self._args.feature_config, 'r') as f:
            self._feature_config = json.load(f)
        self._fs = self._args.sampling_frequency
        self._frame_len_in_train = round(self._args.train_wav_length * self._fs / self._args.stft_hop_size) + 1
        self._hop_frame_len = round(self._args.test_wav_hop_length * self._fs / self._args.stft_hop_size)

    def wav_path2wav(self, wav_path):
        wav, _ = sf.read(wav_path, dtype='float32', always_2d=True)
        wav_ch = wav.shape[1]
        if len(wav) % self._args.stft_hop_size != 0:
            wav = wav[0:-(len(wav) % self._args.stft_hop_size)]
        wav_pad = np.concatenate((np.zeros((self._args.fft_size - self._args.stft_hop_size, wav_ch), dtype='float32'), wav), axis=0)
        duration = len(wav) / self._fs
        return wav_pad, duration

    def wav2spec(self, wav_pad):
        spec_feature = SpectralFeature(wav=wav_pad,
                                       fft_size=self._args.fft_size,
                                       stft_hop_size=self._args.stft_hop_size,
                                       center=False,
                                       config=self._feature_config)
        if self._args.feature == 'amp_phasediff':
            spec = np.concatenate((spec_feature.amplitude(),
                                   spec_feature.phasediff()))

        pad_init = spec[:, :, -int(np.floor((self._frame_len_in_train - self._hop_frame_len) / 2)):]  # loop
        # pad_init = spec[:, :, :int(np.floor((self._frame_len_in_train - self._hop_frame_len) / 2))][:, :, ::-1]  # reflect
        pad_end = spec[:, :, :self._frame_len_in_train]  # loop
        # pad_end = spec[:, :, -self._frame_len_in_train:][:, :, ::-1]  # reflect
        spec_pad = np.concatenate((pad_init, spec, pad_end), axis=2)

        return spec_pad
