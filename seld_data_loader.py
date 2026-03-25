# Copyright 2025 Sony AI

import numpy as np
import pandas as pd
import soundfile as sf
import random
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tqdm
import librosa

from util.func_seld_data_loader import select_time, get_label, get_label_mix
from feature.feature import SpectralFeature


def create_data_loader(args, clap_embedding):
    data_set = SELDDataSet(args, clap_embedding)
    return DataLoader(data_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker, pin_memory=True)


class SELDDataSet(Dataset):
    def __init__(self, args, clap_embedding):
        self._args = args
        self._fs = self._args.sampling_frequency

        self._train_wav_dict = {}
        self._time_array_dict = {}
        self._wav_path_list = pd.read_table(self._args.train_wav_txt, header=None).values.tolist()
        if self._args.disk_config is not None:
            with open(self._args.disk_config, 'r') as f:
                disk_config = json.load(f)
            dir_old = disk_config["dir_old"]
            dir_new = disk_config["dir_new"]
            if self._args.disk_local_storage is not None:
                dir_new = self._args.disk_local_storage
            self._train_wav_path_list = [x[0].replace(dir_old, dir_new) for x in self._wav_path_list]
        else:
            self._train_wav_path_list = [x[0] for x in self._wav_path_list]
        if self._args.quick_check:
            self._train_wav_path_list = self._train_wav_path_list[::2000]
        for train_wav_path in tqdm.tqdm(self._train_wav_path_list, desc='[Train initial setup]'):
            if self._args.train_wav_from == "disk_wav":
                self._train_wav_dict[train_wav_path] = None
            real_csv = train_wav_path.replace('mic', 'metadata').replace('foa', 'metadata').replace('.wav', '.csv')
            self._time_array_dict[train_wav_path] = pd.read_csv(real_csv, header=None).values

        with open(self._args.feature_config, 'r') as f:
            self._feature_config = json.load(f)

        self._clap_embedding = clap_embedding

    def __len__(self):
        return self._args.batch_size * self._args.max_iter  # e.g., 64 * 40000

    def __getitem__(self, idx):  # idx is dummy
        if np.random.rand() < 0.2:  # hard coding
            input_a, label_xyz, label_emb, label_LADtext, name_data = self._data_seld_mix()
        else:
            input_a, label_xyz, label_emb, label_LADtext, name_data = self._data_seld()
        return input_a, label_xyz, label_emb, label_LADtext, name_data

    def _data_seld(self):
        path, time_array, wav, fs, start = self._choice_wav(self._train_wav_dict)
        input_wav = wav[start: start + round(self._args.train_wav_length * fs)]
        input_spec, rotation_pattern = self._wav2spec(input_wav, is_rotation=True)

        label_xyz, label_emb, label_LADtext = get_label(self._args.train_wav_length,
                                                        time_array,
                                                        start / fs,
                                                        rotation_pattern,
                                                        self._args.target_embed_size,
                                                        self._clap_embedding,
                                                        self._args.num_track)
        label_xyz_float = label_xyz.astype(np.float32)
        label_emb_float = label_emb.astype(np.float32)
        label_LADtext_int = label_LADtext.astype(np.int64)

        start_sec = start / fs

        return input_spec, label_xyz_float, label_emb_float, label_LADtext_int, '{}_{}'.format(path, start_sec)

    def _data_seld_mix(self):
        path_1, time_array_1, wav_1, fs, start_1 = self._choice_wav(self._train_wav_dict)
        path_2, time_array_2, wav_2, fs, start_2 = self._choice_wav(self._train_wav_dict)
        input_wav_1 = wav_1[start_1: start_1 + round(self._args.train_wav_length * fs)]
        input_wav_2 = wav_2[start_2: start_2 + round(self._args.train_wav_length * fs)]
        if np.random.rand() < 0.5:  # hard coding
            # input_wav_1 -> comp_spec_1 -> rotated_comp_spec_1 -> rotated_wav_1
            tmp_spec_feature_1 = SpectralFeature(wav=input_wav_1,
                                                 fft_size=self._args.fft_size,
                                                 stft_hop_size=self._args.stft_hop_size,
                                                 center=True,
                                                 config=self._feature_config)
            rotation_pattern_1 = tmp_spec_feature_1.rotate_foa()
            input_wav_1 = tmp_spec_feature_1.get_istft_wav()
        else:
            rotation_pattern_1 = None
        if np.random.rand() < 0.5:  # hard coding
            tmp_spec_feature_2 = SpectralFeature(wav=input_wav_2,
                                                 fft_size=self._args.fft_size,
                                                 stft_hop_size=self._args.stft_hop_size,
                                                 center=True,
                                                 config=self._feature_config)
            rotation_pattern_2 = tmp_spec_feature_2.rotate_foa()
            input_wav_2 = tmp_spec_feature_2.get_istft_wav()
        else:
            rotation_pattern_2 = None
        input_wav = self._mix_wav(input_wav_1, input_wav_2)
        input_spec, _ = self._wav2spec(input_wav, is_rotation=False)

        label_xyz, label_emb, label_LADtext = get_label_mix(self._args.train_wav_length,
                                                            time_array_1,
                                                            start_1 / fs,
                                                            rotation_pattern_1,
                                                            time_array_2,
                                                            start_2 / fs,
                                                            rotation_pattern_2,
                                                            self._args.target_embed_size,
                                                            self._clap_embedding,
                                                            self._args.num_track)
        label_xyz_float = label_xyz.astype(np.float32)
        label_emb_float = label_emb.astype(np.float32)
        label_LADtext_int = label_LADtext.astype(np.int64)

        return input_spec, label_xyz_float, label_emb_float, label_LADtext_int, '{}_{}'.format(path_1, start_1 / fs)

    def _choice_wav(self, train_wav_dict):
        path, wav_fs = random.choice(list(train_wav_dict.items()))
        time_array = self._time_array_dict[path]
        if self._args.train_wav_from == "disk_wav":
            info = sf.info(path)
            if info.samplerate == self._fs:
                wav, fs = sf.read(path, dtype='float32', always_2d=True)
            else:
                wav, fs = librosa.load(path, sr=self._fs, mono=False)  # if sampling frequency is different, librosa will resample
                wav = wav.T  # (ch, time) to (time, ch)
        start = select_time(self._args.train_wav_length, wav, fs)
        return path, time_array, wav, fs, start

    def _wav2spec(self, input_wav, is_rotation):
        spec_feature = SpectralFeature(wav=input_wav,
                                       fft_size=self._args.fft_size,
                                       stft_hop_size=self._args.stft_hop_size,
                                       center=True,
                                       config=self._feature_config)

        if is_rotation is True and np.random.rand() < 0.5:  # hard coding
            rotation_pattern = spec_feature.rotate_foa()
        else:
            rotation_pattern = None

        if np.random.rand() < 0.2:  # hard coding
            spec_feature.eqda()

        if self._args.feature == 'amp_phasediff':
            input_spec = np.concatenate((spec_feature.amplitude(),
                                         spec_feature.phasediff()))

        return input_spec, rotation_pattern

    def _mix_wav(self, input_wav_1, input_wav_2):
        random4gain = random.uniform(0.5, 2)  # hard coding
        input_wav = input_wav_1 + random4gain * input_wav_2

        mean_upper_bound = 0.2  # hard coding
        if np.mean(np.abs(input_wav)) > mean_upper_bound:
            input_wav = input_wav / np.mean(np.abs(input_wav)) * mean_upper_bound

        clipping_bound = 0.99
        input_wav[input_wav > clipping_bound] = clipping_bound

        return input_wav
