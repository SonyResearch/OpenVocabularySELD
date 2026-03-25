# Copyright 2025 Sony AI

import argparse
import os


def file_path(string):
    if os.path.isfile(string):
        return string
    elif string == 'None':
        return None
    else:
        raise FileNotFoundError(string)


def dir_path(string):
    if os.path.isdir(string):
        return string
    elif string == 'None':
        return None
    else:
        raise NotADirectoryError(string)


def model_monitor_path(string):
    if string == './data_fsd50k_tau-srir/model_monitor':  # default is OK even if not a dir
        return string
    else:
        return dir_path(string)


def get_args():
    parser = argparse.ArgumentParser()

    # setup
    parser.add_argument('--train', '-train', action='store_true', help='Train.')
    parser.add_argument('--val', '-val', action='store_true', help='Val.')
    parser.add_argument('--eval', '-eval', action='store_true', help='Eval.')
    parser.add_argument('--inference', '-inference', action='store_true', help='Inference.')
    parser.add_argument('--quick-check', action='store_true', help='Quick check for new implementation.')
    parser.add_argument('--monitor-path', '-m', type=model_monitor_path, default='./data_fsd50k_tau-srir/model_monitor', help='Path monitoring logs saved.')
    parser.add_argument('--random-seed', '-rs', type=int, default=0, help='Seed number for random and np.random.')
    parser.add_argument('--train-wav-from', default='disk_wav', choices=['disk_wav'], help='Train wav from memory or disk.')
    parser.add_argument('--test-wav-from', default='disk', choices=['disk'], help='Test wav from memory or disk.')
    parser.add_argument('--disk-config', type=file_path, default=None, help='Config file can be used to change disk of data.')
    parser.add_argument('--disk-config-test', type=file_path, default=None, help='Config file can be used to change disk of test data.')
    parser.add_argument('--disk-local-storage', type=dir_path, default=None, help='Directory of local storage in computing servers to overwrite the disk config.')
    parser.add_argument('--num-worker', type=int, default=0, help='Number of workers for torch.utils.data.DataLoader.')
    parser.add_argument('--use-raw-output-array', default='none', choices=['none'], help='None, save, or load of raw_output_array in inference.')
    parser.add_argument('--job-id', type=int, default=0, help='Job ID for model monitor.')
    parser.add_argument('--keep-all-checkpoint', action='store_true', help='If true, keep all checkpoints. If false, keep the latest one.')
    # task
    parser.add_argument('--train-wav-txt', type=file_path, default=None, help='Train wave file list text.')
    parser.add_argument('--test-wav-txt', type=file_path, default=None, help='Test wave file list text, e.g., for validation or evaluation.')
    parser.add_argument('--test-model', type=file_path, default=None, help='Test model.')
    parser.add_argument('--test-dataset', default='FSD50K_TAU-SRIR_part', choices=['STARSS23', 'TNSSE21', 'FSD50K_TAU-SRIR_part', 'INFERENCE'], help='Test dataset to switch samples.')
    parser.add_argument('--list-test-wav-txt', type=file_path, default=None, help='List of test wave file list text for FSD50K_TAU-SRIR_part.')
    # net
    parser.add_argument('--net', '-n', default='embaccdoa', choices=['embaccdoa'], help='Neural network architecture.')
    parser.add_argument('--target-embed-size', type=int, default=512, help='Target embedding size.')
    parser.add_argument('--detector', default='embaccdoa', choices=['embaccdoa'], help='SELD detector.')
    parser.add_argument('--num-track', type=int, default=3, help='Total number of tracks.')
    parser.add_argument('--LADaudio-target-embed-BGN', default='zero', choices=['zero'], help='Type of LAD-audio target embedding of BGN.')
    parser.add_argument('--LADtext-target-embed-BGN', default='silent', choices=['silent'], help='Type of LAD-text target embedding of BGN.')
    parser.add_argument('--normalize', default='none', choices=['none'], help='Type of normalization of embedding in network.')
    parser.add_argument('--net-config', type=file_path, default='./net/net_medium.json', help='Config file is required for net.')
    # optimizer
    parser.add_argument('--batch-size', '-b', type=int, default=64)
    parser.add_argument('--learning-rate', '-l', type=float, default=0.0001)
    parser.add_argument('--weight-decay', '-w', type=float, default=0.000001, help='Weight decay factor of SGD update.')
    parser.add_argument('--max-iter', '-i', type=int, default=40000, help='Max iteration of training.')
    parser.add_argument('--model-save-interval', '-s', type=int, default=1000, help='The interval of saving model parameters.')
    parser.add_argument('--loss', default='embaccdoa_pit', choices=['embaccdoa_pit'], help='Loss function.')
    parser.add_argument('--similarity-type', default='cosine', choices=['cosine'], help='Similarity type for loss function and inference.')
    parser.add_argument('--coef-xyz-loss', type=float, default=0.4, help='Coefficient of XYZ loss.')
    parser.add_argument('--coef-emb-loss', type=float, default=0.6, help='Coefficient of embedding loss.')
    parser.add_argument('--coef-LADtext-loss', type=float, default=0.3, help='Coefficient of LAD-text loss.')
    parser.add_argument('--LADaudio-loss-type', default='cosine', choices=['cosine'], help='Type of LAD-audio loss.')
    parser.add_argument('--LADtext-loss-type', default='crossentropy', choices=['crossentropy'], help='Type of LAD-text loss.')
    parser.add_argument('--LADtext-temperature', type=float, default=1.0, help='Temperature of softmax in cross entropy for LAD-text.')
    # feature
    parser.add_argument('--sampling-frequency', '-fs', type=int, default=24000, help='Sampling frequency.')
    parser.add_argument('--feature', default='amp_phasediff', choices=['amp_phasediff'], help='Input audio feature type.')
    parser.add_argument('--fft-size', type=int, default=512, help='FFT size.')
    parser.add_argument('--stft-hop-size', type=int, default=240, help='STFT hop size.')
    parser.add_argument('--train-wav-length', type=float, default=2.55, help='Train wav length [seconds].')
    parser.add_argument('--test-wav-hop-length', type=float, default=2.40, help='Test wav hop length [seconds].')
    parser.add_argument('--feature-config', type=file_path, default='./feature/feature.json', help='Config file is required for feature.')
    # threshold
    parser.add_argument('--threshold-config', type=file_path, default='./util/threshold_pit.json', help='Config file is required for threshold.')
    parser.add_argument('--thresh-emb-interference', type=float, default=0.0, help='Threshold of an embedding vector to remove an interference event.')
    # CLAP
    parser.add_argument('--teacher-model', default='630k-audioset-best', choices=['630k-audioset-best'], help='Teacher model of CLAP.')
    parser.add_argument('--prompt', default='thisisasoundof', choices=['thisisasoundof'], help='Prompt for CLAP text encoder.')

    args = parser.parse_args()

    return args
