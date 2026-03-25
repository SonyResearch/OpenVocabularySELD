# Copyright 2025 Sony AI

import torch
import torch.nn
import numpy as np
import json

from net.net_seld import create_net_seld

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # temporary for pandas
import pandas as pd


class SELDClassifier(object):
    def __init__(self, args):
        self._args = args

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._net = create_net_seld(self._args)
        self._net.to(self._device)
        self._net.eval()
        checkpoint = torch.load(self._args.test_model, map_location=lambda storage, loc: storage)
        self._net.load_state_dict(checkpoint['model_state_dict'])

        fs = self._args.sampling_frequency
        self._frame_per_sec = round(fs / self._args.stft_hop_size)
        self._frame_length = round(self._args.train_wav_length * fs / self._args.stft_hop_size) + 1
        self._hop_frame = round(self._args.test_wav_hop_length * self._frame_per_sec)

    def set_input(self, spec_pad):
        self._spec_pad = spec_pad

    def receive_input(self, time_array):
        features = np.zeros(tuple([self._args.batch_size]) + (self._spec_pad[:, :, :self._frame_length]).shape)

        for index, time in enumerate(time_array):
            frame_idx = int(time * self._frame_per_sec)
            features[index] = self._spec_pad[:, :, frame_idx: frame_idx + self._frame_length]

        self._input_a = torch.tensor(features, dtype=torch.float).to(self._device)

    def calc_output(self):
        output_net = self._net(self._input_a)
        self._output_xyz = output_net[0]
        self._output_emb = output_net[1]

    def get_output(self):
        cut_frame = int(np.floor((self._frame_length - self._hop_frame) / 2))
        output_xyz = self._output_xyz.cpu().detach().numpy()
        output_emb = self._output_emb.cpu().detach().numpy()
        self._output_xyz = 0  # for memory release
        self._output_emb = 0
        # only use output from cut [frame] to cut + hop [frame]
        return output_xyz[:, :, :, cut_frame: cut_frame + self._hop_frame], \
               output_emb[:, :, :, cut_frame: cut_frame + self._hop_frame]

    def get_loss(self):
        # not implemented since we use SELD scores for validation
        return 0


class SELDDetector(object):
    def __init__(self, args, clap_embedding, list_clap_embedding_infer=None):
        self._args = args
        self._clap_embedding = clap_embedding
        self._list_clap_embedding_infer = list_clap_embedding_infer
        with open(args.threshold_config, 'r') as f:
            threshold_config = json.load(f)
        self._thresh_bin = threshold_config['threshold_bin']
        self._size_emb = self._args.target_embed_size

        fs = self._args.sampling_frequency
        self._frame_per_sec = round(fs / self._args.stft_hop_size)
        self._hop_frame = round(self._args.test_wav_hop_length * self._frame_per_sec)

    def set_duration(self, duration):
        self._duration = duration
        test_wav_hop_length = self._args.test_wav_hop_length
        if (self._duration % test_wav_hop_length == 0) or (np.abs((self._duration % test_wav_hop_length) - test_wav_hop_length) < 1e-10):
            self._time_array = np.arange(0, self._duration + test_wav_hop_length, test_wav_hop_length)
        else:
            self._time_array = np.arange(0, self._duration, test_wav_hop_length)

        self._df = pd.DataFrame()
        self._minibatch_result_xyz = np.zeros((
            len(self._time_array) + self._args.batch_size,
            3,
            3,
            self._hop_frame))
        self._raw_output_array_xyz = np.zeros((
            3,
            3,
            len(self._time_array) * self._hop_frame))
        self._minibatch_result_emb = np.zeros((
            len(self._time_array) + self._args.batch_size,
            3,
            self._size_emb,
            self._hop_frame))
        self._raw_output_array_emb = np.zeros((
            3,
            self._size_emb,
            len(self._time_array) * self._hop_frame))

    def get_time_array(self):
        return self._time_array

    def set_minibatch_result(self, index, result):
        result_xyz, result_emb = result
        self._minibatch_result_xyz[
            index * self._args.batch_size: (index + 1) * self._args.batch_size
        ] = result_xyz
        self._minibatch_result_emb[
            index * self._args.batch_size: (index + 1) * self._args.batch_size
        ] = result_emb

    def minibatch_result2raw_output_array(self):
        array_len = (self._minibatch_result_xyz.shape[0]) * self._hop_frame
        result_array_xyz = np.zeros((3, 3, array_len))
        for index, each_result in enumerate(self._minibatch_result_xyz):
            result_array_xyz[
                :, :, index * self._hop_frame: (index + 1) * self._hop_frame
            ] = each_result
        self._raw_output_array_xyz = result_array_xyz[:, :, : len(self._time_array) * self._hop_frame]
        result_array_emb = np.zeros((3, self._size_emb, array_len))
        for index, each_result in enumerate(self._minibatch_result_emb):
            result_array_emb[
                :, :, index * self._hop_frame: (index + 1) * self._hop_frame
            ] = each_result
        self._raw_output_array_emb = result_array_emb[:, :, : len(self._time_array) * self._hop_frame]

    def detect(self, index, time):
        x0 = self._raw_output_array_xyz[0, 0, index * self._hop_frame: (index + 1) * self._hop_frame]
        y0 = self._raw_output_array_xyz[0, 1, index * self._hop_frame: (index + 1) * self._hop_frame]
        z0 = self._raw_output_array_xyz[0, 2, index * self._hop_frame: (index + 1) * self._hop_frame]
        x1 = self._raw_output_array_xyz[1, 0, index * self._hop_frame: (index + 1) * self._hop_frame]
        y1 = self._raw_output_array_xyz[1, 1, index * self._hop_frame: (index + 1) * self._hop_frame]
        z1 = self._raw_output_array_xyz[1, 2, index * self._hop_frame: (index + 1) * self._hop_frame]
        x2 = self._raw_output_array_xyz[2, 0, index * self._hop_frame: (index + 1) * self._hop_frame]
        y2 = self._raw_output_array_xyz[2, 1, index * self._hop_frame: (index + 1) * self._hop_frame]
        z2 = self._raw_output_array_xyz[2, 2, index * self._hop_frame: (index + 1) * self._hop_frame]
        azi0, ele0, bin0 = self._xyz2azi_ele_bin(x0, y0, z0)
        azi1, ele1, bin1 = self._xyz2azi_ele_bin(x1, y1, z1)
        azi2, ele2, bin2 = self._xyz2azi_ele_bin(x2, y2, z2)

        emb0 = self._raw_output_array_emb[0, :, index * self._hop_frame: (index + 1) * self._hop_frame]
        emb1 = self._raw_output_array_emb[1, :, index * self._hop_frame: (index + 1) * self._hop_frame]
        emb2 = self._raw_output_array_emb[2, :, index * self._hop_frame: (index + 1) * self._hop_frame]

        self._frame_per_sec4csv = 10  # hard coding for csv setting
        hop_frame4csv = int(self._hop_frame / (self._frame_per_sec / self._frame_per_sec4csv))  # e.g., 12 [frame in csv]
        for csv_idx, frame in enumerate(range(round(time * self._frame_per_sec4csv), round(time * self._frame_per_sec4csv) + hop_frame4csv)):
            csv2net = int(self._frame_per_sec / self._frame_per_sec4csv)  # e.g., 100 [frame for net] / 10 [frame for csv]
            net_idx_start = csv_idx * csv2net
            net_idx_end = (csv_idx + 1) * csv2net
            azi_mean0, ele_mean0, bin_mean0, emb_mean0 = self._azi_ele_bin_emb2mean(azi0, ele0, bin0, emb0, net_idx_start, net_idx_end)
            azi_mean1, ele_mean1, bin_mean1, emb_mean1 = self._azi_ele_bin_emb2mean(azi1, ele1, bin1, emb1, net_idx_start, net_idx_end)
            azi_mean2, ele_mean2, bin_mean2, emb_mean2 = self._azi_ele_bin_emb2mean(azi2, ele2, bin2, emb2, net_idx_start, net_idx_end)

            if bin_mean0 > self._thresh_bin:
                self._append_df(frame, emb_mean0, azi_mean0, ele_mean0, bin_mean0)
            if bin_mean1 > self._thresh_bin:
                self._append_df(frame, emb_mean1, azi_mean1, ele_mean1, bin_mean1)
            if bin_mean2 > self._thresh_bin:
                self._append_df(frame, emb_mean2, azi_mean2, ele_mean2, bin_mean2)

    def _xyz2azi_ele_bin(self, x, y, z):
        azi = np.arctan2(y, x)
        ele = np.arctan2(z, np.sqrt(x**2 + y**2))
        bin = np.sqrt(x**2 + y**2 + z**2)
        bin[bin > 1] = 1
        return azi, ele, bin

    def _azi_ele_bin_emb2mean(self, azi, ele, bin, emb, idx_start, idx_end):
        bin_mean = np.mean(bin[idx_start: idx_end])
        azi_mean = np.sum(bin[idx_start: idx_end] * azi[idx_start: idx_end]) / np.sum(bin[idx_start: idx_end])
        ele_mean = np.sum(bin[idx_start: idx_end] * ele[idx_start: idx_end]) / np.sum(bin[idx_start: idx_end])
        emb_mean = np.sum(bin[idx_start: idx_end] * emb[:, idx_start: idx_end], axis=1) / np.sum(bin[idx_start: idx_end])
        return azi_mean, ele_mean, bin_mean, emb_mean

    def _append_df(self, frame, emb_mean, azi_mean, ele_mean, bin_mean):
        event_class = self._clap_embedding.emb2class(emb_mean, self._list_clap_embedding_infer)
        if event_class != -1:
            self._df = self._df.append(pd.DataFrame([(frame, event_class, azi_mean / np.pi * 180, ele_mean / np.pi * 180)]))

    def save_df(self, pred_path):
        if not self._df.empty:
            self._df = self._df.sort_values(0)
            self._df = self._df[self._df[0] < int(self._duration * self._frame_per_sec4csv)]  # cut frames after duration
        self._df.to_csv(pred_path, sep=',', index=False, header=False)


class SELDDetectorInference(SELDDetector):
    def _append_df(self, frame, emb_mean, azi_mean, ele_mean, bin_mean):
        top3_categories, top3_similarities = self._clap_embedding.emb2top3_cat_with_sim(emb_mean)
        if top3_categories[0] != "silent":
            # rounding for better readability in csv during this inference
            self._df = self._df.append(pd.DataFrame([(
                frame,
                round(bin_mean, 3),
                round(azi_mean / np.pi * 180, 1),
                round(ele_mean / np.pi * 180, 1),
                top3_categories[0], round(top3_similarities[0], 3),
                top3_categories[1], round(top3_similarities[1], 3),
                top3_categories[2], round(top3_similarities[2], 3),
            )]))
