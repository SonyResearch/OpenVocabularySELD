# Copyright 2025 Sony AI

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np

from net.net_util import interpolate


def create_net_seld(args):
    with open(args.feature_config, 'r') as f:
        feature_config = json.load(f)
    with open(args.net_config, 'r') as f:
        net_config = json.load(f)
    if args.net == 'embaccdoa':
        if args.num_track == 3:
            Net = EmbedACCDOA(in_channels=feature_config[args.feature]["ch"],
                              target_embed_size=args.target_embed_size,
                              normalize=args.normalize,
                              net_config=net_config)

    return Net


class EmbedACCDOA(nn.Module):
    def __init__(self, in_channels, target_embed_size, normalize, net_config, interp_ratio=4):
        super().__init__()

        self.pe_enable = False  # True | False
        if normalize == "none":
            self.normalize_enable = False

        self.out_channels1 = net_config["conv"]["out_channels1"]
        self.out_channels2 = net_config["conv"]["out_channels2"]
        self.out_channels3 = net_config["conv"]["out_channels3"]

        self.d_model = net_config["transformer"]["d_model"]
        self.d_ff = net_config["transformer"]["d_ff"]
        self.n_heads = net_config["transformer"]["n_heads"]
        self.n_layers = net_config["transformer"]["n_layers"]

        self.in_channels = in_channels
        self.target_embed_size = target_embed_size
        self.interp_ratio = interp_ratio

        # some network hyper-params are hard coding
        self.downsample_ratio = 2 ** 2
        self.sed_conv_block1 = nn.Sequential(
            DoubleConv(in_channels=4, out_channels=self.out_channels1),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        self.sed_conv_block2 = nn.Sequential(
            DoubleConv(in_channels=self.out_channels1, out_channels=self.out_channels2),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        self.sed_conv_block3 = nn.Sequential(
            DoubleConv(in_channels=self.out_channels2, out_channels=self.out_channels3),
            nn.AvgPool2d(kernel_size=(1, 2)),
        )
        self.sed_conv_block4 = nn.Sequential(
            DoubleConv(in_channels=self.out_channels3, out_channels=self.d_model),
            nn.AvgPool2d(kernel_size=(1, 2)),
        )

        self.doa_conv_block1 = nn.Sequential(
            DoubleConv(in_channels=self.in_channels, out_channels=self.out_channels1),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        self.doa_conv_block2 = nn.Sequential(
            DoubleConv(in_channels=self.out_channels1, out_channels=self.out_channels2),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        self.doa_conv_block3 = nn.Sequential(
            DoubleConv(in_channels=self.out_channels2, out_channels=self.out_channels3),
            nn.AvgPool2d(kernel_size=(1, 2)),
        )
        self.doa_conv_block4 = nn.Sequential(
            DoubleConv(in_channels=self.out_channels3, out_channels=self.d_model),
            nn.AvgPool2d(kernel_size=(1, 2)),
        )

        self.stitch1 = nn.Parameter(torch.FloatTensor(self.out_channels1, 2, 2).uniform_(0.1, 0.9))
        self.stitch2 = nn.Parameter(torch.FloatTensor(self.out_channels2, 2, 2).uniform_(0.1, 0.9))
        self.stitch3 = nn.Parameter(torch.FloatTensor(self.out_channels3, 2, 2).uniform_(0.1, 0.9))

        if self.pe_enable:
            self.pe = PositionalEncoding(pos_len=100, d_model=self.d_model, pe_type='t', dropout=0.0)
        self.sed_trans_track1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads, dim_feedforward=self.d_ff, dropout=0.2),
            num_layers=self.n_layers, enable_nested_tensor=False)
        self.sed_trans_track2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads, dim_feedforward=self.d_ff, dropout=0.2),
            num_layers=self.n_layers, enable_nested_tensor=False)
        self.sed_trans_track3 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads, dim_feedforward=self.d_ff, dropout=0.2),
            num_layers=self.n_layers, enable_nested_tensor=False)
        self.doa_trans_track1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads, dim_feedforward=self.d_ff, dropout=0.2),
            num_layers=self.n_layers, enable_nested_tensor=False)
        self.doa_trans_track2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads, dim_feedforward=self.d_ff, dropout=0.2),
            num_layers=self.n_layers, enable_nested_tensor=False)
        self.doa_trans_track3 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads, dim_feedforward=self.d_ff, dropout=0.2),
            num_layers=self.n_layers, enable_nested_tensor=False)

        self.fc_sed_track1 = nn.Linear(self.d_model, self.target_embed_size, bias=True)
        self.fc_sed_track2 = nn.Linear(self.d_model, self.target_embed_size, bias=True)
        self.fc_sed_track3 = nn.Linear(self.d_model, self.target_embed_size, bias=True)
        self.fc_doa_track1 = nn.Linear(self.d_model, 3, bias=True)
        self.fc_doa_track2 = nn.Linear(self.d_model, 3, bias=True)
        self.fc_doa_track3 = nn.Linear(self.d_model, 3, bias=True)
        self.final_act_sed = nn.Sequential()  # nn.Sigmoid()
        self.final_act_doa = nn.Tanh()

    def forward(self, x_a):
        x_a = x_a.transpose(2, 3)
        b, c, t, f = x_a.size()  # (N, C, T, F); N = batch_size, C = in_channels, T = time_frames, F = freq_bins

        x_sed = x_a[:, :4]
        x_doa = x_a

        # cnn
        x_sed = self.sed_conv_block1(x_sed)
        x_doa = self.doa_conv_block1(x_doa)
        x_sed = torch.einsum('c, nctf -> nctf', self.stitch1[:, 0, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch1[:, 0, 1], x_doa)
        x_doa = torch.einsum('c, nctf -> nctf', self.stitch1[:, 1, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch1[:, 1, 1], x_doa)
        x_sed = self.sed_conv_block2(x_sed)
        x_doa = self.doa_conv_block2(x_doa)
        x_sed = torch.einsum('c, nctf -> nctf', self.stitch2[:, 0, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch2[:, 0, 1], x_doa)
        x_doa = torch.einsum('c, nctf -> nctf', self.stitch2[:, 1, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch2[:, 1, 1], x_doa)
        x_sed = self.sed_conv_block3(x_sed)
        x_doa = self.doa_conv_block3(x_doa)
        x_sed = torch.einsum('c, nctf -> nctf', self.stitch3[:, 0, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch3[:, 0, 1], x_doa)
        x_doa = torch.einsum('c, nctf -> nctf', self.stitch3[:, 1, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch3[:, 1, 1], x_doa)
        x_sed = self.sed_conv_block4(x_sed)
        x_doa = self.doa_conv_block4(x_doa)
        x_sed = x_sed.mean(dim=3)  # (N, C, T)
        x_doa = x_doa.mean(dim=3)  # (N, C, T)

        # transformer
        if self.pe_enable:
            x_sed = self.pe(x_sed)
        if self.pe_enable:
            x_doa = self.pe(x_doa)
        x_sed = x_sed.permute(2, 0, 1)  # (T, N, C)
        x_doa = x_doa.permute(2, 0, 1)  # (T, N, C)

        x_sed_1 = self.sed_trans_track1(x_sed).transpose(0, 1)  # (N, T, C)
        x_sed_2 = self.sed_trans_track2(x_sed).transpose(0, 1)  # (N, T, C)
        x_sed_3 = self.sed_trans_track3(x_sed).transpose(0, 1)  # (N, T, C)
        x_doa_1 = self.doa_trans_track1(x_doa).transpose(0, 1)  # (N, T, 3)
        x_doa_2 = self.doa_trans_track2(x_doa).transpose(0, 1)  # (N, T, 3)
        x_doa_3 = self.doa_trans_track3(x_doa).transpose(0, 1)  # (N, T, 3)

        # fc
        x_sed_1 = self.final_act_sed(self.fc_sed_track1(x_sed_1))  # (N, T, C)
        x_sed_2 = self.final_act_sed(self.fc_sed_track2(x_sed_2))
        x_sed_3 = self.final_act_sed(self.fc_sed_track3(x_sed_3))
        x_doa_1 = self.final_act_doa(self.fc_doa_track1(x_doa_1))  # (N, T, 3)
        x_doa_2 = self.final_act_doa(self.fc_doa_track2(x_doa_2))
        x_doa_3 = self.final_act_doa(self.fc_doa_track3(x_doa_3))

        # normalize
        if self.normalize_enable:
            x_sed_1 = F.normalize(x_sed_1, dim=-1)
            x_sed_2 = F.normalize(x_sed_2, dim=-1)
            x_sed_3 = F.normalize(x_sed_3, dim=-1)

        # interpolate
        x_sed_1 = interpolate(x_sed_1, self.interp_ratio)
        x_sed_1 = x_sed_1.transpose(1, 2)  # (N, C, T)
        x_sed_2 = interpolate(x_sed_2, self.interp_ratio)
        x_sed_2 = x_sed_2.transpose(1, 2)  # (N, C, T)
        x_sed_3 = interpolate(x_sed_3, self.interp_ratio)
        x_sed_3 = x_sed_3.transpose(1, 2)  # (N, C, T)
        x_doa_1 = interpolate(x_doa_1, self.interp_ratio)
        x_doa_1 = x_doa_1.transpose(1, 2)  # (N, 3, T)
        x_doa_2 = interpolate(x_doa_2, self.interp_ratio)
        x_doa_2 = x_doa_2.transpose(1, 2)  # (N, 3, T)
        x_doa_3 = interpolate(x_doa_3, self.interp_ratio)
        x_doa_3 = x_doa_3.transpose(1, 2)  # (N, 3, T)

        x_sed = torch.stack((x_sed_1, x_sed_2, x_sed_3), 1)  # (N, 3, target_embed_size, T)
        x_doa = torch.stack((x_doa_1, x_doa_2, x_doa_3), 1)  # (N, 3, 3, T)

        return x_doa, x_sed


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                 dilation=1, bias=False):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x):
        x = self.double_conv(x)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, pos_len, d_model=512, pe_type='t', dropout=0.0):
        """ Positional encoding using sin and cos

        Args:
            pos_len: positional length
            d_model: number of feature maps
            pe_type: 't' | 'f' , time domain, frequency domain
            dropout: dropout probability
        """
        super().__init__()

        self.pe_type = pe_type
        pe = torch.zeros(pos_len, d_model)
        pos = torch.arange(0, pos_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = 0.1 * torch.sin(pos * div_term)
        pe[:, 1::2] = 0.1 * torch.cos(pos * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2)  # (N, C, T)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x is (N, C, T, F) or (N, C, T) or (N, C, F)
        if x.ndim == 4:
            if self.pe_type == 't':
                pe = self.pe.unsqueeze(3)
                x += pe[:, :, :x.shape[2]]
            elif self.pe_type == 'f':
                pe = self.pe.unsqueeze(2)
                x += pe[:, :, :, :x.shape[3]]
        elif x.ndim == 3:
            x += self.pe[:, :, :x.shape[2]]
        return self.dropout(x)
