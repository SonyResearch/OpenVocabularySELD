# Copyright 2025 Sony AI

import os
import datetime
import json
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import tqdm
from collections import OrderedDict
import time

from util.args import get_args
from util.validation_monitor import ValidationMonitor
from clap_embedding import CLAPEmbedding
from clap_embedding_inference import CLAPEmbeddingInference
from seld_trainer import SELDTrainer
from seld_validator import SELDValidator, SELDInference


def main():
    args = get_args()
    if args.inference:
        inference(args)
    elif args.eval:
        evaluation(args)
    elif args.train:
        train(args)


def train(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    start_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    each_monitor_path = '{}/{}_{}'.format(args.monitor_path, start_time, args.job_id)
    os.makedirs(each_monitor_path, exist_ok=True)

    clap_embedding = CLAPEmbedding(args.target_embed_size, args.similarity_type,
                                   args.LADaudio_target_embed_BGN, args.LADtext_target_embed_BGN,
                                   args.teacher_model, args.prompt, args.thresh_emb_interference)
    seld_trainer = SELDTrainer(args, clap_embedding)
    seld_validator = SELDValidator(args, clap_embedding, each_monitor_path)

    writer = SummaryWriter(log_dir=each_monitor_path)
    monitor_val = ValidationMonitor(writer)
    with open(os.path.join(each_monitor_path, 'args.json'), 'x') as fout:
        json.dump(vars(args), fout, indent=4)

    with tqdm.tqdm(enumerate(seld_trainer.data_loader), total=len(seld_trainer.data_loader), desc='[Train]') as pbar:
        for batch_ndx, sample in pbar:
            i = batch_ndx + 1
            seld_trainer.receive_input(sample)
            seld_trainer.back_propagation()
            writer.add_scalar('Loss/train', seld_trainer.get_loss(), i)
            writer.add_scalar('Optimizer/lr', seld_trainer.get_lr(), i)
            pbar.set_postfix(OrderedDict(loss=seld_trainer.get_loss()))

            if i % args.model_save_interval == 0:
                seld_trainer.swa_update(i)
                seld_trainer.save(each_monitor_path, i, start_time, args.job_id)
                if args.val:
                    val_results = seld_validator.validation(seld_trainer.get_each_model_path(i))
                    monitor_val.add(i, val_results)
                    seld_trainer.lr_step(i)  # check iteration

    time.sleep(0.1)  # wait for TensorBoard writing of the last iteration


def evaluation(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    clap_embedding = CLAPEmbedding(args.target_embed_size, args.similarity_type,
                                   args.LADaudio_target_embed_BGN, args.LADtext_target_embed_BGN,
                                   args.teacher_model, args.prompt, args.thresh_emb_interference)
    seld_validator = SELDValidator(args, clap_embedding, os.path.dirname(args.test_model))
    _ = seld_validator.validation(args.test_model)


def inference(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    clap_embedding_inference = CLAPEmbeddingInference(args.target_embed_size, args.similarity_type,
                                                      args.teacher_model, args.prompt)
    seld_inference = SELDInference(args, clap_embedding_inference, os.path.dirname(args.test_model))
    _ = seld_inference.inference()


if __name__ == '__main__':
    main()
