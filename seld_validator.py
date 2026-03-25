# Copyright 2025 Sony AI

import os
import tqdm
import pandas as pd
import json
import numpy as np
import codecs

from wav_convertor import WavConvertor
from seld_predictor import SELDClassifier, SELDDetector, SELDDetectorInference
from seld_eval_dcase2024 import all_seld_eval


class SELDValidator(object):
    def __init__(self, args, clap_embedding, monitor_path):
        self._args = args
        self._clap_embedding = clap_embedding
        self._wav_convertor = WavConvertor(self._args)

        if self._args.test_dataset in ['STARSS23', 'TNSSE21']:
            if self._args.val:
                tag = '{}_tmp4val'.format(os.path.splitext(os.path.basename(self._args.test_wav_txt))[0])
            elif self._args.eval:
                tag = '{}_{}'.format(os.path.splitext(os.path.basename(self._args.test_wav_txt))[0],
                                     os.path.splitext(os.path.basename(self._args.test_model))[0][-7:])  # iteration
            self._pred_dir, self._result_path = self._prepare_output_path(monitor_path, tag)
            self._test_wav_path_list = self._make_test_wav_path_list(self._args.test_wav_txt)

        elif self._args.test_dataset == 'FSD50K_TAU-SRIR_part':
            list_test_wav_txt = [x[0] for x in pd.read_table(self._args.list_test_wav_txt, header=None).values.tolist()]
            if self._args.quick_check:
                list_test_wav_txt = list_test_wav_txt[::10]
            if self._args.val:
                overall_tag = '{}_tmp4val'.format(os.path.splitext(os.path.basename(self._args.list_test_wav_txt))[0])
            elif self._args.eval:
                overall_tag = '{}_{}'.format(os.path.splitext(os.path.basename(self._args.list_test_wav_txt))[0],
                                             os.path.splitext(os.path.basename(self._args.test_model))[0][-7:])  # iteration
            overall_dir = os.path.join(monitor_path, 'pred_result_{}'.format(overall_tag))
            os.makedirs(overall_dir, exist_ok=True)
            self._overall_result_path = os.path.join(monitor_path, 'result_{}.txt'.format(overall_tag))

            self._dict_output_path = {}
            self._dict_test_wav_path_list = {}
            self._dict_part_category_txt = {}
            for test_wav_txt in list_test_wav_txt:
                key = os.path.splitext(os.path.basename(test_wav_txt))[0][-16:]  # e.g., part_category_00
                each_tag = '{}'.format(key)
                self._dict_output_path[key] = self._prepare_output_path(overall_dir, each_tag)
                self._dict_test_wav_path_list[key] = self._make_test_wav_path_list(test_wav_txt)
                self._dict_part_category_txt[key] = test_wav_txt.replace("dataset", "category").replace("fsd50k_tau-srir_foa_val_", "")

    def _prepare_output_path(self, dir_path, tag):
        pred_dir = os.path.join(dir_path, 'pred_{}'.format(tag))
        os.makedirs(pred_dir, exist_ok=True)
        result_path = os.path.join(dir_path, 'result_{}.txt'.format(tag))
        return pred_dir, result_path

    def _make_test_wav_path_list(self, test_wav_txt):
        wav_path_list = pd.read_table(test_wav_txt, header=None).values.tolist()
        if self._args.disk_config_test is not None:
            with open(self._args.disk_config_test, 'r') as f:
                disk_config_test = json.load(f)
            dir_old = disk_config_test['dir_old']
            dir_new = disk_config_test['dir_new']
            if self._args.disk_local_storage is not None:
                dir_new = self._args.disk_local_storage
            test_wav_path_list = [x[0].replace(dir_old, dir_new) for x in wav_path_list]
        else:
            test_wav_path_list = [x[0] for x in wav_path_list]
        if self._args.quick_check:
            return test_wav_path_list[::50]
        else:
            return test_wav_path_list

    def validation(self, model_path):
        if self._args.test_dataset in ['STARSS23', 'TNSSE21']:
            list_clap_embedding_infer = self._clap_embedding.get_list_clap_embedding_infer(self._args.test_dataset)
            all_test_metric, test_loss = self._valid(model_path, self._test_wav_path_list, self._pred_dir, self._result_path,
                                                     list_clap_embedding_infer, self._args.test_dataset)
            return all_test_metric, test_loss
        elif self._args.test_dataset == 'FSD50K_TAU-SRIR_part':
            avg_all_test_metric, avg_test_loss = self._repeat_valid(model_path, self._dict_test_wav_path_list, self._dict_output_path,
                                                                    self._dict_part_category_txt, self._overall_result_path)
            return avg_all_test_metric, avg_test_loss

    def _valid(self, model_path, test_wav_path_list, pred_dir, result_path,
               list_clap_embedding_infer, test_dataset,
               part_category_txt=None, print_stdout=True, desc='[Test]'):
        self._args.test_model = model_path  # temporary replace args for validation
        if self._args.detector == 'embaccdoa':
            self._seld_classifier = SELDClassifier(self._args)
            if self._args.num_track == 3:
                self._seld_detector = SELDDetector(self._args,
                                                   self._clap_embedding,
                                                   list_clap_embedding_infer)
        test_loss = 0
        for test_wav_path in tqdm.tqdm(test_wav_path_list, desc=desc):
            pred_path = os.path.join(pred_dir, '{}.csv'.format(os.path.splitext(os.path.basename(test_wav_path))[0]))
            if self._args.use_raw_output_array == 'none':
                wav_loss = self._pred_wav(test_wav_path, pred_path)
            test_loss += wav_loss
        test_loss = test_loss / len(test_wav_path_list)

        ref_metadata_dir = os.path.dirname(os.path.dirname(test_wav_path_list[0].replace("foa", "metadata").replace("mic", "metadata")))
        all_test_metric = all_seld_eval(ref_files_folder=ref_metadata_dir, pred_output_format_files=pred_dir,
                                        test_dataset=test_dataset, part_category_txt=part_category_txt,
                                        print_stdout=print_stdout, result_path=result_path)

        return all_test_metric, test_loss

    def _pred_wav(self, wav_path, pred_path):
        # input setup
        if self._args.test_wav_from == 'disk':
            wav_pad, duration = self._wav_convertor.wav_path2wav(wav_path)
            spec_pad = self._wav_convertor.wav2spec(wav_pad)
            duration = duration

        # classifier and detector setup
        self._seld_classifier.set_input(spec_pad)
        self._seld_detector.set_duration(duration)
        time_array = self._seld_detector.get_time_array()

        # minibatch-like processing for classifier
        wav_loss = 0
        for index, time in enumerate(time_array[::self._args.batch_size]):
            self._seld_classifier.receive_input(
                time_array[index * self._args.batch_size: (index + 1) * self._args.batch_size])
            self._seld_classifier.calc_output()
            wav_loss += self._seld_classifier.get_loss()
            self._seld_detector.set_minibatch_result(
                index=index,
                result=self._seld_classifier.get_output()
            )
        self._seld_detector.minibatch_result2raw_output_array()
        wav_loss = wav_loss / len(time_array[::self._args.batch_size])

        # online-like processing for detector
        for index, time in enumerate(time_array):
            self._seld_detector.detect(index=index, time=time)
        self._seld_detector.save_df(pred_path)

        return wav_loss

    def _repeat_valid(self, model_path, dict_test_wav_path_list, dict_output_path,
                      dict_part_category_txt, overall_result_path):
        sum_test_metric = np.zeros(4)  # er20, f20, le, lr; seld_err is computed after averaging
        sum_other_scores = np.zeros(7)  # er20_d, er20_i, er20_s, pre, rec, lf, lp; ignore classwise_other_results
        sum_test_loss = 0

        for key in dict_test_wav_path_list:
            test_wav_path_list = dict_test_wav_path_list[key]
            pred_dir, result_path = dict_output_path[key]
            part_category_txt = dict_part_category_txt[key]
            list_clap_embedding_infer = self._clap_embedding.get_list_clap_embedding_infer(self._args.test_dataset, part_category_txt)
            all_test_metric, test_loss = self._valid(model_path, test_wav_path_list, pred_dir, result_path,
                                                     list_clap_embedding_infer, self._args.test_dataset,
                                                     part_category_txt=part_category_txt, print_stdout=False,
                                                     desc='[Test {}]'.format(key))

            er20, f20, le, lr, seld_err, other_scores = all_test_metric
            er20_d, er20_i, er20_s, pre, rec, lf, lp, classwise_other_results = other_scores
            sum_test_metric += np.array([er20, f20, le, lr])
            sum_other_scores += np.array([er20_d, er20_i, er20_s, pre, rec, lf, lp])
            sum_test_loss += test_loss

        total_test = len(dict_test_wav_path_list)
        avg_er20, avg_f20, avg_le, avg_lr = sum_test_metric / total_test
        avg_seld_err = np.mean([avg_er20, 1 - avg_f20, avg_le / 180, 1 - avg_lr])  # compute seld_err with average results
        avg_other_scores = sum_other_scores / total_test
        avg_test_loss = sum_test_loss / total_test

        self._print_avg_result(avg_er20, avg_f20, avg_le, avg_lr, avg_seld_err, avg_other_scores, overall_result_path)

        # keep similar shape as _valid for ValidationMonitor
        return [avg_er20, avg_f20, avg_le, avg_lr, avg_seld_err, avg_other_scores], avg_test_loss

    def _print_avg_result(self, avg_er20, avg_f20, avg_le, avg_lr, avg_seld_err, avg_other_scores, overall_result_path):
        print('SELD scores')
        print('All\tER\tF\tLE\tLR\tSELD\tER_D\tER_I\tER_S\tP\tR\tLF\tLP')
        print('All\t{:0.3f}\t{:0.3f}\t{:0.2f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}'.format(
            avg_er20, avg_f20, avg_le, avg_lr, avg_seld_err,
            avg_other_scores[0], avg_other_scores[1], avg_other_scores[2],
            avg_other_scores[3], avg_other_scores[4], avg_other_scores[5], avg_other_scores[6]))

        print('SELD scores',
              file=codecs.open(overall_result_path, 'w', 'utf-8'))
        print('All\tER\tF\tLE\tLR\tSELD\tER_D\tER_I\tER_S\tP\tR\tLF\tLP',
              file=codecs.open(overall_result_path, 'a', 'utf-8'))
        print('All\t{:0.3f}\t{:0.3f}\t{:0.2f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}'.format(
            avg_er20, avg_f20, avg_le, avg_lr, avg_seld_err,
            avg_other_scores[0], avg_other_scores[1], avg_other_scores[2],
            avg_other_scores[3], avg_other_scores[4], avg_other_scores[5], avg_other_scores[6]),
            file=codecs.open(overall_result_path, 'a', 'utf-8'))


class SELDInference(object):
    def __init__(self, args, clap_embedding_inference, monitor_path):
        self._args = args
        self._clap_embedding_inference = clap_embedding_inference
        self._wav_convertor = WavConvertor(self._args)

        wav_txt_name = os.path.splitext(os.path.basename(self._args.test_wav_txt))[0]
        model_iter = os.path.splitext(os.path.basename(self._args.test_model))[0][-7:]  # iteration
        tag = f'{wav_txt_name}_{model_iter}'
        self._pred_dir = os.path.join(monitor_path, f'inference_{tag}')
        os.makedirs(self._pred_dir, exist_ok=True)

        wav_path_list = pd.read_table(self._args.test_wav_txt, header=None).values.tolist()
        assert self._args.disk_config_test is None, (
            "Using disk_config_test is not supported in inference. This arg is for validation to flexibly change disk of test data."
        )
        self._test_wav_path_list = [x[0] for x in wav_path_list]

    def inference(self):
        assert self._args.detector == 'embaccdoa', f"Unsupported detector: {self._args.detector}. Only 'embaccdoa' is supported."
        self._seld_classifier = SELDClassifier(self._args)

        for test_wav_path in tqdm.tqdm(self._test_wav_path_list, desc='[Inference]'):
            pred_name = os.path.splitext(os.path.basename(test_wav_path))[0]
            pred_path = os.path.join(self._pred_dir, f'{pred_name}.csv')

            test_category_txt_path = test_wav_path.replace("wav", "txt")
            self._clap_embedding_inference.set_list_clap_embedding_infer_from_category_txt(test_category_txt_path)
            assert self._args.num_track == 3, f"Unsupported num_track: {self._args.num_track}. Only num_track=3 is supported."
            self._seld_detector = SELDDetectorInference(self._args, self._clap_embedding_inference)

            assert self._args.use_raw_output_array == 'none', (
                f"Unsupported use_raw_output_array: {self._args.use_raw_output_array}. Only 'none' is supported."
            )
            _ = self._pred_wav(test_wav_path, pred_path)

        return 0

    def _pred_wav(self, wav_path, pred_path):
        # input setup
        assert self._args.test_wav_from == 'disk', f"Unsupported test_wav_from: {self._args.test_wav_from}. Only 'disk' is supported."
        wav_pad, duration = self._wav_convertor.wav_path2wav(wav_path)
        spec_pad = self._wav_convertor.wav2spec(wav_pad)

        # classifier and detector setup
        self._seld_classifier.set_input(spec_pad)
        self._seld_detector.set_duration(duration)
        time_array = self._seld_detector.get_time_array()

        # minibatch-like processing for classifier
        for index, time in enumerate(time_array[::self._args.batch_size]):
            self._seld_classifier.receive_input(
                time_array[index * self._args.batch_size: (index + 1) * self._args.batch_size])
            self._seld_classifier.calc_output()
            self._seld_detector.set_minibatch_result(
                index=index,
                result=self._seld_classifier.get_output()
            )
        self._seld_detector.minibatch_result2raw_output_array()

        # online-like processing for detector
        for index, time in enumerate(time_array):
            self._seld_detector.detect(index=index, time=time)
        self._seld_detector.save_df(pred_path)

        return 0
