# Copyright 2025 Sony AI

import torch
import torch.nn
import torch.nn.functional
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from timm.scheduler import CosineLRScheduler
import os

from seld_data_loader import create_data_loader
from net.net_seld import create_net_seld


class SELDTrainer(object):
    def __init__(self, args, clap_embedding):
        self._args = args

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.data_loader = create_data_loader(self._args, clap_embedding)  # also used in seld.py

        self._net = create_net_seld(self._args)
        self._net.to(self._device)
        self._net.train()

        if self._args.loss == 'embaccdoa_pit':
            if self._args.similarity_type == "cosine":
                if self._args.num_track == 3:
                    self._criterion = EmbACCDOA_PIT_Cosine_LADtext(self._args.coef_xyz_loss,
                                                                   self._args.coef_emb_loss,
                                                                   self._args.coef_LADtext_loss,
                                                                   self._args.LADaudio_loss_type,
                                                                   self._args.LADaudio_target_embed_BGN,
                                                                   self._args.LADtext_loss_type,
                                                                   self._args.LADtext_target_embed_BGN,
                                                                   clap_embedding.get_all_class_LADtext_emb().to(self._device),
                                                                   self._args.LADtext_temperature)
        self._optimizer = optim.Adam(
            self._net.parameters(),
            lr=self._args.learning_rate,
            weight_decay=self._args.weight_decay
        )
        self._lr_scheduler = CosineLRScheduler(self._optimizer,
                                               t_initial=25,
                                               lr_min=self._args.learning_rate * 0.5,
                                               warmup_t=5,
                                               warmup_lr_init=self._args.learning_rate * 0.2,
                                               warmup_prefix=True)  # hard coding

        self._swa_model = AveragedModel(self._net)
        self._swa_start = 30  # hard coding
        self._swa_scheduler = SWALR(self._optimizer, swa_lr=self._args.learning_rate * 0.5)  # hard coding

        self._each_checkpoint_path = None
        self._each_checkpoint_path_prev = None
        self._each_swa_checkpoint_path = None
        self._each_swa_checkpoint_path_prev = None

    def receive_input(self, sample):
        _input_a, _label_xyz, _label_emb, _label_LADtext, _ = sample

        self._input_a = _input_a.to(self._device, non_blocking=True)
        self._label_xyz = _label_xyz.to(self._device, non_blocking=True)
        self._label_emb = _label_emb.to(self._device, non_blocking=True)
        self._label_LADtext = _label_LADtext.to(self._device, non_blocking=True)

    def back_propagation(self):
        self._net.train()
        self._optimizer.zero_grad()

        output_net = self._net(self._input_a)
        self._output_xyz = output_net[0]
        self._output_emb = output_net[1]
        self._loss = self._criterion(self._output_xyz, self._output_emb, self._label_xyz, self._label_emb, self._label_LADtext)
        self._loss.backward()

        self._optimizer.step()

    def save(self, each_monitor_path=None, iteration=None, start_time=None, job_id=None):
        self._each_checkpoint_path = '{}/params_{}_{}_{:07}.pth'.format(
            each_monitor_path,
            start_time,
            job_id,
            iteration)
        torch_net_state_dict = self._net.state_dict()
        checkpoint = {'model_state_dict': torch_net_state_dict,
                      'optimizer_state_dict': self._optimizer.state_dict(),
                      'scheduler_state_dict': self._lr_scheduler.state_dict(),
                      'rng_state': torch.get_rng_state(),
                      'cuda_rng_state': torch.cuda.get_rng_state()}
        torch.save(checkpoint, self._each_checkpoint_path)
        print('save checkpoint to {}.'.format(self._each_checkpoint_path))
        if ((not self._args.keep_all_checkpoint)
            and (self._each_checkpoint_path_prev is not None)
            and (os.path.exists(self._each_checkpoint_path_prev))):
            os.remove(self._each_checkpoint_path_prev)
            print('remove checkpoint on {}.'.format(self._each_checkpoint_path_prev))
        self._each_checkpoint_path_prev = self._each_checkpoint_path

        pseudo_epoch = int(iteration / self._args.model_save_interval)
        if pseudo_epoch > self._swa_start:
            iter_times = 10  # hard coding
            batches = torch.zeros((iter_times,
                                   self._input_a.shape[0],
                                   self._input_a.shape[1],
                                   self._input_a.shape[2],
                                   self._input_a.shape[3])).to(self._device)
            for i, each_batch in enumerate(self.data_loader):
                if i == iter_times:
                    break
                batches[i] = each_batch[0]
            my_update_bn(batches, self._swa_model)

            self._each_swa_checkpoint_path = self._each_checkpoint_path.replace("params_", "params_swa_")
            torch_net_state_dict = self._swa_model.module.state_dict()
            checkpoint = {'model_state_dict': torch_net_state_dict,
                          'optimizer_state_dict': self._optimizer.state_dict(),
                          'scheduler_state_dict': self._lr_scheduler.state_dict(),
                          'rng_state': torch.get_rng_state(),
                          'cuda_rng_state': torch.cuda.get_rng_state()}
            torch.save(checkpoint, self._each_swa_checkpoint_path)
            print('save checkpoint to {}.'.format(self._each_swa_checkpoint_path))
            if ((not self._args.keep_all_checkpoint)
                and (self._each_swa_checkpoint_path_prev is not None)
                and (os.path.exists(self._each_swa_checkpoint_path_prev))):
                os.remove(self._each_swa_checkpoint_path_prev)
                print('remove checkpoint on {}.'.format(self._each_swa_checkpoint_path_prev))
            self._each_swa_checkpoint_path_prev = self._each_swa_checkpoint_path

    def lr_step(self, iteration):
        pseudo_epoch = int(iteration / self._args.model_save_interval)
        if pseudo_epoch > self._swa_start:
            self._swa_scheduler.step()
        else:
            self._lr_scheduler.step(pseudo_epoch)

    def swa_update(self, iteration):
        pseudo_epoch = int(iteration / self._args.model_save_interval)
        if pseudo_epoch > self._swa_start:
            self._swa_model.update_parameters(self._net)

    def get_loss(self):
        return self._loss.cpu().detach().numpy()

    def get_lr(self):
        return self._optimizer.state_dict()['param_groups'][0]['lr']

    def get_each_model_path(self, iteration):
        pseudo_epoch = int(iteration / self._args.model_save_interval)
        if pseudo_epoch > self._swa_start:
            return self._each_swa_checkpoint_path
        else:
            return self._each_checkpoint_path


def my_update_bn(loader, model, device=None):  # mainly from torch.optim.swa_utils.update_bn
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = None
            module.num_batches_tracked *= 0

    for input in loader:
        if isinstance(input, (list, tuple)):
            input = input[0]
        if device is not None:
            input = input.to(device)

        model(input)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)


class EmbACCDOA_PIT_Cosine_LADtext(object):
    def __init__(self, coef_xyz_loss=0.4, coef_emb_loss=0.6, coef_LADtext_loss=0.3,
                 LADaudio_loss='cosine', LADaudio_target_embed_BGN='zero',
                 LADtext_loss='crossentropy', LADtext_target_embed_BGN='silent',
                 all_class_LADtext_emb=None, temperature=1.0):
        super().__init__()
        self.eps = 1e-6
        self.coef_xyz_loss = coef_xyz_loss
        self.coef_emb_loss = coef_emb_loss
        self.coef_LADtext_loss = coef_LADtext_loss
        self.each_xyz_loss = torch.nn.MSELoss(reduction='none')
        self.LADaudio_loss = LADaudio_loss
        if self.LADaudio_loss == 'cosine':
            self.each_emb_loss = torch.nn.CosineSimilarity(dim=2, eps=self.eps)
        self.LADtext_loss = LADtext_loss
        if self.LADtext_loss == 'crossentropy':
            self.each_LADtext_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.LADaudio_target_embed_BGN = LADaudio_target_embed_BGN
        self.LADtext_target_embed_BGN = LADtext_target_embed_BGN
        self.all_class_LADtext_emb = all_class_LADtext_emb
        self.temperature = temperature
        self.num_class = self.all_class_LADtext_emb.shape[1]

    def _each_xyz_calc(self, output, target):
        return self.each_xyz_loss(output, target).mean(dim=(1, 2))

    def _each_emb_calc(self, output_emb, target_emb):
        if self.LADaudio_loss == 'cosine':
            return 1 - self.each_emb_loss(output_emb, target_emb).mean(dim=1)

    def _each_LADtext_calc(self, output, target):
        if self.LADtext_loss == 'crossentropy':
            return self.each_LADtext_loss(output / self.temperature, target).mean(dim=1)

    def __call__(self, output_xyz, output_emb, target_xyz, target_emb, target_LADtext):
        """
        Permutation Invariant Training (PIT) for 15 (=3+6+6) possible combinations
        Args:
            output_xyz: [batch_size, num_track=3, num_xyz=3,   num_frames]
            output_emb: [batch_size, num_track=3, num_emb=512, num_frames]
            target_xyz: [batch_size, num_track=6, num_xyz=3,   num_frames]
            target_emb: [batch_size, num_track=6, num_emb=512, num_frames]
            target_LADtext: [batch_size, num_track=6, num_frames]
        Return:
            loss: scalar
        """
        target_xyz_A0 = target_xyz[:, 0, :, :]  # A0, no ov, [batch_size, num_xyz=3, num_frames]
        target_xyz_B0 = target_xyz[:, 1, :, :]  # B0, ov with 2 sources
        target_xyz_B1 = target_xyz[:, 2, :, :]  # B1
        target_xyz_C0 = target_xyz[:, 3, :, :]  # C0, ov with 3 sources
        target_xyz_C1 = target_xyz[:, 4, :, :]  # C1
        target_xyz_C2 = target_xyz[:, 5, :, :]  # C2
        target_xyz___ = torch.zeros_like(target_xyz_A0)
        target_xyz_A0____ = torch.stack((target_xyz_A0, target_xyz___, target_xyz___), 1)  # 3 permutation of A (no ov), [batch_size, num_track=3, num_xyz=3, num_frames]
        target_xyz___A0__ = torch.stack((target_xyz___, target_xyz_A0, target_xyz___), 1)
        target_xyz_____A0 = torch.stack((target_xyz___, target_xyz___, target_xyz_A0), 1)
        target_xyz_B0B1__ = torch.stack((target_xyz_B0, target_xyz_B1, target_xyz___), 1)  # 6 permutations of B (ov with 2 sources)
        target_xyz_B1B0__ = torch.stack((target_xyz_B1, target_xyz_B0, target_xyz___), 1)
        target_xyz_B0__B1 = torch.stack((target_xyz_B0, target_xyz___, target_xyz_B1), 1)
        target_xyz_B1__B0 = torch.stack((target_xyz_B1, target_xyz___, target_xyz_B0), 1)
        target_xyz___B0B1 = torch.stack((target_xyz___, target_xyz_B0, target_xyz_B1), 1)
        target_xyz___B1B0 = torch.stack((target_xyz___, target_xyz_B1, target_xyz_B0), 1)
        target_xyz_C0C1C2 = torch.stack((target_xyz_C0, target_xyz_C1, target_xyz_C2), 1)  # 6 permutations of C (ov with 3 sources)
        target_xyz_C0C2C1 = torch.stack((target_xyz_C0, target_xyz_C2, target_xyz_C1), 1)
        target_xyz_C1C0C2 = torch.stack((target_xyz_C1, target_xyz_C0, target_xyz_C2), 1)
        target_xyz_C1C2C0 = torch.stack((target_xyz_C1, target_xyz_C2, target_xyz_C0), 1)
        target_xyz_C2C0C1 = torch.stack((target_xyz_C2, target_xyz_C0, target_xyz_C1), 1)
        target_xyz_C2C1C0 = torch.stack((target_xyz_C2, target_xyz_C1, target_xyz_C0), 1)
        pad_xyz4A = target_xyz_B0B1__ + target_xyz_C0C1C2
        pad_xyz4B = target_xyz_A0____ + target_xyz_C0C1C2
        pad_xyz4C = target_xyz_A0____ + target_xyz_B0B1__
        # padded in order to avoid to set zero as target
        loss_xyz00 = self._each_xyz_calc(output_xyz, target_xyz_A0____ + pad_xyz4A)
        loss_xyz01 = self._each_xyz_calc(output_xyz, target_xyz___A0__ + pad_xyz4A)
        loss_xyz02 = self._each_xyz_calc(output_xyz, target_xyz_____A0 + pad_xyz4A)
        loss_xyz03 = self._each_xyz_calc(output_xyz, target_xyz_B0B1__ + pad_xyz4B)
        loss_xyz04 = self._each_xyz_calc(output_xyz, target_xyz_B1B0__ + pad_xyz4B)
        loss_xyz05 = self._each_xyz_calc(output_xyz, target_xyz_B0__B1 + pad_xyz4B)
        loss_xyz06 = self._each_xyz_calc(output_xyz, target_xyz_B1__B0 + pad_xyz4B)
        loss_xyz07 = self._each_xyz_calc(output_xyz, target_xyz___B0B1 + pad_xyz4B)
        loss_xyz08 = self._each_xyz_calc(output_xyz, target_xyz___B1B0 + pad_xyz4B)
        loss_xyz09 = self._each_xyz_calc(output_xyz, target_xyz_C0C1C2 + pad_xyz4C)
        loss_xyz10 = self._each_xyz_calc(output_xyz, target_xyz_C0C2C1 + pad_xyz4C)
        loss_xyz11 = self._each_xyz_calc(output_xyz, target_xyz_C1C0C2 + pad_xyz4C)
        loss_xyz12 = self._each_xyz_calc(output_xyz, target_xyz_C1C2C0 + pad_xyz4C)
        loss_xyz13 = self._each_xyz_calc(output_xyz, target_xyz_C2C0C1 + pad_xyz4C)
        loss_xyz14 = self._each_xyz_calc(output_xyz, target_xyz_C2C1C0 + pad_xyz4C)

        target_emb_A0 = target_emb[:, 0, :, :]  # A0, no ov, [batch_size, num_emb=512, num_frames]
        target_emb_B0 = target_emb[:, 1, :, :]  # B0, ov with 2 sources
        target_emb_B1 = target_emb[:, 2, :, :]  # B1
        target_emb_C0 = target_emb[:, 3, :, :]  # C0, ov with 3 sources
        target_emb_C1 = target_emb[:, 4, :, :]  # C1
        target_emb_C2 = target_emb[:, 5, :, :]  # C2
        target_emb___ = torch.zeros_like(target_emb_A0)
        target_emb_A0____ = torch.stack((target_emb_A0, target_emb___, target_emb___), 1)  # 3 permutation of A (no ov), [batch_size, num_track=3, num_emb=512, num_frames]
        target_emb___A0__ = torch.stack((target_emb___, target_emb_A0, target_emb___), 1)
        target_emb_____A0 = torch.stack((target_emb___, target_emb___, target_emb_A0), 1)
        target_emb_B0B1__ = torch.stack((target_emb_B0, target_emb_B1, target_emb___), 1)  # 6 permutations of B (ov with 2 sources)
        target_emb_B1B0__ = torch.stack((target_emb_B1, target_emb_B0, target_emb___), 1)
        target_emb_B0__B1 = torch.stack((target_emb_B0, target_emb___, target_emb_B1), 1)
        target_emb_B1__B0 = torch.stack((target_emb_B1, target_emb___, target_emb_B0), 1)
        target_emb___B0B1 = torch.stack((target_emb___, target_emb_B0, target_emb_B1), 1)
        target_emb___B1B0 = torch.stack((target_emb___, target_emb_B1, target_emb_B0), 1)
        target_emb_C0C1C2 = torch.stack((target_emb_C0, target_emb_C1, target_emb_C2), 1)  # 6 permutations of C (ov with 3 sources)
        target_emb_C0C2C1 = torch.stack((target_emb_C0, target_emb_C2, target_emb_C1), 1)
        target_emb_C1C0C2 = torch.stack((target_emb_C1, target_emb_C0, target_emb_C2), 1)
        target_emb_C1C2C0 = torch.stack((target_emb_C1, target_emb_C2, target_emb_C0), 1)
        target_emb_C2C0C1 = torch.stack((target_emb_C2, target_emb_C0, target_emb_C1), 1)
        target_emb_C2C1C0 = torch.stack((target_emb_C2, target_emb_C1, target_emb_C0), 1)
        pad_emb4A = target_emb_B0B1__ + target_emb_C0C1C2
        pad_emb4B = target_emb_A0____ + target_emb_C0C1C2
        pad_emb4C = target_emb_A0____ + target_emb_B0B1__
        # padded in order to avoid to set zero as target
        loss_emb00 = self._each_emb_calc(output_emb, target_emb_A0____ + pad_emb4A)
        loss_emb01 = self._each_emb_calc(output_emb, target_emb___A0__ + pad_emb4A)
        loss_emb02 = self._each_emb_calc(output_emb, target_emb_____A0 + pad_emb4A)
        loss_emb03 = self._each_emb_calc(output_emb, target_emb_B0B1__ + pad_emb4B)
        loss_emb04 = self._each_emb_calc(output_emb, target_emb_B1B0__ + pad_emb4B)
        loss_emb05 = self._each_emb_calc(output_emb, target_emb_B0__B1 + pad_emb4B)
        loss_emb06 = self._each_emb_calc(output_emb, target_emb_B1__B0 + pad_emb4B)
        loss_emb07 = self._each_emb_calc(output_emb, target_emb___B0B1 + pad_emb4B)
        loss_emb08 = self._each_emb_calc(output_emb, target_emb___B1B0 + pad_emb4B)
        loss_emb09 = self._each_emb_calc(output_emb, target_emb_C0C1C2 + pad_emb4C)
        loss_emb10 = self._each_emb_calc(output_emb, target_emb_C0C2C1 + pad_emb4C)
        loss_emb11 = self._each_emb_calc(output_emb, target_emb_C1C0C2 + pad_emb4C)
        loss_emb12 = self._each_emb_calc(output_emb, target_emb_C1C2C0 + pad_emb4C)
        loss_emb13 = self._each_emb_calc(output_emb, target_emb_C2C0C1 + pad_emb4C)
        loss_emb14 = self._each_emb_calc(output_emb, target_emb_C2C1C0 + pad_emb4C)

        target_LADtext_A0 = target_LADtext[:, 0, :]  # A0, no ov, [batch_size, num_frames]
        target_LADtext_B0 = target_LADtext[:, 1, :]  # B0, ov with 2 sources
        target_LADtext_B1 = target_LADtext[:, 2, :]  # B1
        target_LADtext_C0 = target_LADtext[:, 3, :]  # C0, ov with 3 sources
        target_LADtext_C1 = target_LADtext[:, 4, :]  # C1
        target_LADtext_C2 = target_LADtext[:, 5, :]  # C2
        target_LADtext___ = torch.zeros_like(target_LADtext_A0)
        target_LADtext_A0____ = torch.stack((target_LADtext_A0, target_LADtext___, target_LADtext___), 1)  # 3 permutation of A (no ov), [batch_size, num_track=3, num_frames]
        target_LADtext___A0__ = torch.stack((target_LADtext___, target_LADtext_A0, target_LADtext___), 1)
        target_LADtext_____A0 = torch.stack((target_LADtext___, target_LADtext___, target_LADtext_A0), 1)
        target_LADtext_B0B1__ = torch.stack((target_LADtext_B0, target_LADtext_B1, target_LADtext___), 1)  # 6 permutations of B (ov with 2 sources)
        target_LADtext_B1B0__ = torch.stack((target_LADtext_B1, target_LADtext_B0, target_LADtext___), 1)
        target_LADtext_B0__B1 = torch.stack((target_LADtext_B0, target_LADtext___, target_LADtext_B1), 1)
        target_LADtext_B1__B0 = torch.stack((target_LADtext_B1, target_LADtext___, target_LADtext_B0), 1)
        target_LADtext___B0B1 = torch.stack((target_LADtext___, target_LADtext_B0, target_LADtext_B1), 1)
        target_LADtext___B1B0 = torch.stack((target_LADtext___, target_LADtext_B1, target_LADtext_B0), 1)
        target_LADtext_C0C1C2 = torch.stack((target_LADtext_C0, target_LADtext_C1, target_LADtext_C2), 1)  # 6 permutations of C (ov with 3 sources)
        target_LADtext_C0C2C1 = torch.stack((target_LADtext_C0, target_LADtext_C2, target_LADtext_C1), 1)
        target_LADtext_C1C0C2 = torch.stack((target_LADtext_C1, target_LADtext_C0, target_LADtext_C2), 1)
        target_LADtext_C1C2C0 = torch.stack((target_LADtext_C1, target_LADtext_C2, target_LADtext_C0), 1)
        target_LADtext_C2C0C1 = torch.stack((target_LADtext_C2, target_LADtext_C0, target_LADtext_C1), 1)
        target_LADtext_C2C1C0 = torch.stack((target_LADtext_C2, target_LADtext_C1, target_LADtext_C0), 1)
        pad_LADtext4A = target_LADtext_B0B1__ + target_LADtext_C0C1C2
        pad_LADtext4B = target_LADtext_A0____ + target_LADtext_C0C1C2
        pad_LADtext4C = target_LADtext_A0____ + target_LADtext_B0B1__
        # calculate softmax of cosine similarity for LADtext
        output_cos_sim = torch.matmul(output_emb.transpose(2, 3), self.all_class_LADtext_emb) /\
            (torch.linalg.norm(output_emb.transpose(2, 3), axis=3)[:, :, :, None] * torch.linalg.norm(self.all_class_LADtext_emb, axis=0)[None, :] + self.eps)
        output_LADtext = output_cos_sim.transpose(1, 3).transpose(2, 3)  # [batch_size, num_class, num_track, num_frames]
        # padded in order to avoid to set zero as target
        loss_LADtext00 = self._each_LADtext_calc(output_LADtext, target_LADtext_A0____ + pad_LADtext4A)
        loss_LADtext01 = self._each_LADtext_calc(output_LADtext, target_LADtext___A0__ + pad_LADtext4A)
        loss_LADtext02 = self._each_LADtext_calc(output_LADtext, target_LADtext_____A0 + pad_LADtext4A)
        loss_LADtext03 = self._each_LADtext_calc(output_LADtext, target_LADtext_B0B1__ + pad_LADtext4B)
        loss_LADtext04 = self._each_LADtext_calc(output_LADtext, target_LADtext_B1B0__ + pad_LADtext4B)
        loss_LADtext05 = self._each_LADtext_calc(output_LADtext, target_LADtext_B0__B1 + pad_LADtext4B)
        loss_LADtext06 = self._each_LADtext_calc(output_LADtext, target_LADtext_B1__B0 + pad_LADtext4B)
        loss_LADtext07 = self._each_LADtext_calc(output_LADtext, target_LADtext___B0B1 + pad_LADtext4B)
        loss_LADtext08 = self._each_LADtext_calc(output_LADtext, target_LADtext___B1B0 + pad_LADtext4B)
        loss_LADtext09 = self._each_LADtext_calc(output_LADtext, target_LADtext_C0C1C2 + pad_LADtext4C)
        loss_LADtext10 = self._each_LADtext_calc(output_LADtext, target_LADtext_C0C2C1 + pad_LADtext4C)
        loss_LADtext11 = self._each_LADtext_calc(output_LADtext, target_LADtext_C1C0C2 + pad_LADtext4C)
        loss_LADtext12 = self._each_LADtext_calc(output_LADtext, target_LADtext_C1C2C0 + pad_LADtext4C)
        loss_LADtext13 = self._each_LADtext_calc(output_LADtext, target_LADtext_C2C0C1 + pad_LADtext4C)
        loss_LADtext14 = self._each_LADtext_calc(output_LADtext, target_LADtext_C2C1C0 + pad_LADtext4C)

        loss00 = self.coef_xyz_loss * loss_xyz00 + self.coef_emb_loss * loss_emb00 + self.coef_LADtext_loss * loss_LADtext00
        loss01 = self.coef_xyz_loss * loss_xyz01 + self.coef_emb_loss * loss_emb01 + self.coef_LADtext_loss * loss_LADtext01
        loss02 = self.coef_xyz_loss * loss_xyz02 + self.coef_emb_loss * loss_emb02 + self.coef_LADtext_loss * loss_LADtext02
        loss03 = self.coef_xyz_loss * loss_xyz03 + self.coef_emb_loss * loss_emb03 + self.coef_LADtext_loss * loss_LADtext03
        loss04 = self.coef_xyz_loss * loss_xyz04 + self.coef_emb_loss * loss_emb04 + self.coef_LADtext_loss * loss_LADtext04
        loss05 = self.coef_xyz_loss * loss_xyz05 + self.coef_emb_loss * loss_emb05 + self.coef_LADtext_loss * loss_LADtext05
        loss06 = self.coef_xyz_loss * loss_xyz06 + self.coef_emb_loss * loss_emb06 + self.coef_LADtext_loss * loss_LADtext06
        loss07 = self.coef_xyz_loss * loss_xyz07 + self.coef_emb_loss * loss_emb07 + self.coef_LADtext_loss * loss_LADtext07
        loss08 = self.coef_xyz_loss * loss_xyz08 + self.coef_emb_loss * loss_emb08 + self.coef_LADtext_loss * loss_LADtext08
        loss09 = self.coef_xyz_loss * loss_xyz09 + self.coef_emb_loss * loss_emb09 + self.coef_LADtext_loss * loss_LADtext09
        loss10 = self.coef_xyz_loss * loss_xyz10 + self.coef_emb_loss * loss_emb10 + self.coef_LADtext_loss * loss_LADtext10
        loss11 = self.coef_xyz_loss * loss_xyz11 + self.coef_emb_loss * loss_emb11 + self.coef_LADtext_loss * loss_LADtext11
        loss12 = self.coef_xyz_loss * loss_xyz12 + self.coef_emb_loss * loss_emb12 + self.coef_LADtext_loss * loss_LADtext12
        loss13 = self.coef_xyz_loss * loss_xyz13 + self.coef_emb_loss * loss_emb13 + self.coef_LADtext_loss * loss_LADtext13
        loss14 = self.coef_xyz_loss * loss_xyz14 + self.coef_emb_loss * loss_emb14 + self.coef_LADtext_loss * loss_LADtext14

        loss_min = torch.min(
            torch.stack((loss00,
                         loss01,
                         loss02,
                         loss03,
                         loss04,
                         loss05,
                         loss06,
                         loss07,
                         loss08,
                         loss09,
                         loss10,
                         loss11,
                         loss12,
                         loss13,
                         loss14), dim=0),
            dim=0).indices

        loss = (loss00 * (loss_min == 0) +
                loss01 * (loss_min == 1) +
                loss02 * (loss_min == 2) +
                loss03 * (loss_min == 3) +
                loss04 * (loss_min == 4) +
                loss05 * (loss_min == 5) +
                loss06 * (loss_min == 6) +
                loss07 * (loss_min == 7) +
                loss08 * (loss_min == 8) +
                loss09 * (loss_min == 9) +
                loss10 * (loss_min == 10) +
                loss11 * (loss_min == 11) +
                loss12 * (loss_min == 12) +
                loss13 * (loss_min == 13) +
                loss14 * (loss_min == 14)).mean()

        return loss
