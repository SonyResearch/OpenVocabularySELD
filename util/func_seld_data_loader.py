# Copyright 2025 Sony AI

import numpy as np
import random
import math


def select_time(train_wav_length, wav, fs):
    center = random.randrange(round(0        + train_wav_length / 2 * fs),
                              round(len(wav) - train_wav_length / 2 * fs))
    start = center - round(train_wav_length / 2 * fs)

    return start


def add_label_axis_emb_each_frame(label_axis, label_emb, label_LADtext,
                                  clap_embedding, time_array4frame_event, start_frame):
    azi_rad = time_array4frame_event[3] / 180 * np.pi
    ele_rad = time_array4frame_event[4] / 180 * np.pi
    x_axis = 1 * np.cos(ele_rad) * np.cos(azi_rad)
    y_axis = 1 * np.cos(ele_rad) * np.sin(azi_rad)
    z_axis = 1 * np.sin(ele_rad)
    label_axis[0, start_frame: start_frame + 10] = x_axis
    label_axis[1, start_frame: start_frame + 10] = y_axis
    label_axis[2, start_frame: start_frame + 10] = z_axis

    fsd_id = time_array4frame_event[5]
    # CLAP embedding from fsd_id wav
    label_emb[:, start_frame: start_frame + 10] = clap_embedding.fsd_id_wav2emb(fsd_id)[:, np.newaxis]
    # class from fsd_id; we might use "class = time_array4frame_event[1] + 1"
    label_LADtext[start_frame: start_frame + 10] = clap_embedding.fsd_id2class(fsd_id)

    return label_axis, label_emb, label_LADtext


class label4SELDUNK(object):
    def __init__(self, num_axis, num_emb, num_frame_wide, clap_embedding):
        super().__init__()
        self._clap_embedding = clap_embedding

        self._label_wide_axis_0 = np.zeros([num_axis, num_frame_wide])      # a0----
        self._label_wide_axis_1 = np.zeros([num_axis, num_frame_wide])      # --b0--
        self._label_wide_axis_2 = np.zeros([num_axis, num_frame_wide])      # --b1--
        self._label_wide_axis_3 = np.zeros([num_axis, num_frame_wide])      # ----c0
        self._label_wide_axis_4 = np.zeros([num_axis, num_frame_wide])      # ----c1
        self._label_wide_axis_5 = np.zeros([num_axis, num_frame_wide])      # ----c2
        self._label_wide_emb_0 = np.zeros([num_emb, num_frame_wide])        # a0----
        self._label_wide_emb_1 = np.zeros([num_emb, num_frame_wide])        # --b0--
        self._label_wide_emb_2 = np.zeros([num_emb, num_frame_wide])        # --b1--
        self._label_wide_emb_3 = np.zeros([num_emb, num_frame_wide])        # ----c0
        self._label_wide_emb_4 = np.zeros([num_emb, num_frame_wide])        # ----c1
        self._label_wide_emb_5 = np.zeros([num_emb, num_frame_wide])        # ----c2
        self._label_wide_LADtext_0 = np.zeros([num_frame_wide])             # a0----
        self._label_wide_LADtext_1 = np.zeros([num_frame_wide])             # --b0--
        self._label_wide_LADtext_2 = np.zeros([num_frame_wide])             # --b1--
        self._label_wide_LADtext_3 = np.zeros([num_frame_wide])             # ----c0
        self._label_wide_LADtext_4 = np.zeros([num_frame_wide])             # ----c1
        self._label_wide_LADtext_5 = np.zeros([num_frame_wide])             # ----c2

    def add_label_each_frame(self, list_time_array4frame_event, start_frame):
        if len(list_time_array4frame_event) == 1:
            self._label_wide_axis_0, self._label_wide_emb_0, self._label_wide_LADtext_0 = \
                add_label_axis_emb_each_frame(self._label_wide_axis_0, self._label_wide_emb_0, self._label_wide_LADtext_0,
                                              self._clap_embedding, list_time_array4frame_event[0], start_frame)
        elif len(list_time_array4frame_event) == 2:
            self._label_wide_axis_1, self._label_wide_emb_1, self._label_wide_LADtext_1 = \
                add_label_axis_emb_each_frame(self._label_wide_axis_1, self._label_wide_emb_1, self._label_wide_LADtext_1,
                                              self._clap_embedding, list_time_array4frame_event[0], start_frame)
            self._label_wide_axis_2, self._label_wide_emb_2, self._label_wide_LADtext_2 = \
                add_label_axis_emb_each_frame(self._label_wide_axis_2, self._label_wide_emb_2, self._label_wide_LADtext_2,
                                              self._clap_embedding, list_time_array4frame_event[1], start_frame)
        else:  # more than ov2
            self._label_wide_axis_3, self._label_wide_emb_3, self._label_wide_LADtext_3 = \
                add_label_axis_emb_each_frame(self._label_wide_axis_3, self._label_wide_emb_3, self._label_wide_LADtext_3,
                                              self._clap_embedding, list_time_array4frame_event[0], start_frame)
            self._label_wide_axis_4, self._label_wide_emb_4, self._label_wide_LADtext_4 = \
                add_label_axis_emb_each_frame(self._label_wide_axis_4, self._label_wide_emb_4, self._label_wide_LADtext_4,
                                              self._clap_embedding, list_time_array4frame_event[1], start_frame)
            self._label_wide_axis_5, self._label_wide_emb_5, self._label_wide_LADtext_5 = \
                add_label_axis_emb_each_frame(self._label_wide_axis_5, self._label_wide_emb_5, self._label_wide_LADtext_5,
                                              self._clap_embedding, list_time_array4frame_event[2], start_frame)

    def concat(self, index_diff, num_frame):
        label_axis = np.stack((
            self._label_wide_axis_0[:, index_diff: index_diff + num_frame],
            self._label_wide_axis_1[:, index_diff: index_diff + num_frame],
            self._label_wide_axis_2[:, index_diff: index_diff + num_frame],
            self._label_wide_axis_3[:, index_diff: index_diff + num_frame],
            self._label_wide_axis_4[:, index_diff: index_diff + num_frame],
            self._label_wide_axis_5[:, index_diff: index_diff + num_frame]
        ))
        label_emb = np.stack((
            self._label_wide_emb_0[:, index_diff: index_diff + num_frame],
            self._label_wide_emb_1[:, index_diff: index_diff + num_frame],
            self._label_wide_emb_2[:, index_diff: index_diff + num_frame],
            self._label_wide_emb_3[:, index_diff: index_diff + num_frame],
            self._label_wide_emb_4[:, index_diff: index_diff + num_frame],
            self._label_wide_emb_5[:, index_diff: index_diff + num_frame]
        ))
        label_LADtext = np.stack((
            self._label_wide_LADtext_0[index_diff: index_diff + num_frame],
            self._label_wide_LADtext_1[index_diff: index_diff + num_frame],
            self._label_wide_LADtext_2[index_diff: index_diff + num_frame],
            self._label_wide_LADtext_3[index_diff: index_diff + num_frame],
            self._label_wide_LADtext_4[index_diff: index_diff + num_frame],
            self._label_wide_LADtext_5[index_diff: index_diff + num_frame]
        ))
        return label_axis, label_emb, label_LADtext


def rotate_time_array(time_array4frame, rotation_pattern):
    if rotation_pattern:
        azi_reflection, azi_rotation, ele_reflection = rotation_pattern
    else:
        azi_reflection, azi_rotation, ele_reflection = [1, 0, 1]  # if None, no rotation

    time_array4frame[:, 3] = azi_reflection * time_array4frame[:, 3] + azi_rotation / np.pi * 180
    time_array4frame[:, 4] = ele_reflection * time_array4frame[:, 4]
    return time_array4frame


def get_label(train_wav_length, time_array, start_sec, rotation_pattern, target_embed_size, clap_embedding, num_track):
    num_axis = 3  # X, Y, Z
    num_emb = target_embed_size
    num_frame = round(train_wav_length * 100) + 1

    end_sec = start_sec + train_wav_length

    index_diff = int(math.modf(start_sec * 10)[0] * 10)  # get second decimal place
    num_frame_wide = (int(np.ceil(end_sec * 10)) - int(np.floor(start_sec * 10)) + 1) * 10
    # "+ 1" is buffer for numerical error, such as index_diff=3 and num_frame_wide=130

    if num_track == 3:
        label_class = label4SELDUNK(num_axis, num_emb, int(num_frame_wide), clap_embedding)

    for index, frame in enumerate(range(int(np.floor(start_sec * 10)), int(np.ceil(end_sec * 10)))):
        time_array4frame = rotate_time_array(time_array[time_array[:, 0] == frame], rotation_pattern)  # (0, 5) shape is ok
        sorted_time_array4frame = time_array4frame[np.argsort(time_array4frame[:, 1])]

        list_time_array4frame_event = []
        for i in range(len(sorted_time_array4frame)):
            list_time_array4frame_event.append(sorted_time_array4frame[i])
            if i == len(sorted_time_array4frame) - 1:  # if the last
                label_class.add_label_each_frame(list_time_array4frame_event, index * 10)
                list_time_array4frame_event = []

    label_axis, label_emb, label_LADtext = label_class.concat(int(index_diff), num_frame)

    return label_axis, label_emb, label_LADtext


def get_label_mix(train_wav_length, time_array_1, start_sec_1, rotation_pattern_1,
                  time_array_2, start_sec_2, rotation_pattern_2, target_embed_size, clap_embedding, num_track):
    num_axis = 3  # X, Y, Z
    num_emb = target_embed_size
    num_frame = round(train_wav_length * 100) + 1

    end_sec_1 = start_sec_1 + train_wav_length
    end_sec_2 = start_sec_2 + train_wav_length

    index_diff_array = np.zeros(2)
    num_frame_wide_array = np.zeros(2)
    index_diff_array[0] = int(math.modf(start_sec_1 * 10)[0] * 10)  # get second decimal place
    num_frame_wide_array[0] = (int(np.ceil(end_sec_1 * 10)) - int(np.floor(start_sec_1 * 10)) + 1) * 10
    index_diff_array[1] = int(math.modf(start_sec_2 * 10)[0] * 10)
    num_frame_wide_array[1] = (int(np.ceil(end_sec_2 * 10)) - int(np.floor(start_sec_2 * 10)) + 1) * 10
    index_diff = index_diff_array[np.argmax(num_frame_wide_array)]
    num_frame_wide = np.max(num_frame_wide_array)
    # "+ 1" is buffer for numerical error, such as index_diff=3 and num_frame_wide=130

    if num_track == 3:
        label_class = label4SELDUNK(num_axis, num_emb, int(num_frame_wide), clap_embedding)

    frames_1 = np.arange(int(np.floor(start_sec_1 * 10)), int(np.ceil(end_sec_1 * 10)))
    frames_2 = np.arange(int(np.floor(start_sec_2 * 10)), int(np.ceil(end_sec_2 * 10)))
    for index in range(min(len(frames_1), len(frames_2))):
        time_array4frame_1 = rotate_time_array(time_array_1[time_array_1[:, 0] == frames_1[index]], rotation_pattern_1)  # (0, 5) shape is ok
        time_array4frame_2 = rotate_time_array(time_array_2[time_array_2[:, 0] == frames_2[index]], rotation_pattern_2)
        time_array4frame = np.concatenate((time_array4frame_1, time_array4frame_2))
        sorted_time_array4frame = time_array4frame[np.argsort(time_array4frame[:, 1])]

        list_time_array4frame_event = []
        for i in range(len(sorted_time_array4frame)):
            list_time_array4frame_event.append(sorted_time_array4frame[i])
            if i == len(sorted_time_array4frame) - 1:  # if the last
                label_class.add_label_each_frame(list_time_array4frame_event, index * 10)
                list_time_array4frame_event = []

    label_axis, label_emb, label_LADtext = label_class.concat(int(index_diff), num_frame)

    return label_axis, label_emb, label_LADtext
