"""Temporal Convolutional Network used by torque_estimator and trajectory_estimator."""

import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout, dropout_type, activation):
        super().__init__()
        self.chomp1 = Chomp1d(padding)
        self.af1 = getattr(nn, activation)()
        self.dropout1 = getattr(nn, dropout_type)(dropout)
        self.chomp2 = Chomp1d(padding)
        self.af2 = getattr(nn, activation)()
        self.dropout2 = getattr(nn, dropout_type)(dropout)
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.af1, self.dropout1,
            self.conv2, self.chomp2, self.af2, self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.af = getattr(nn, activation)()

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        return self.af(self.net(x) + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout, dropout_type, activation):
        super().__init__()
        layers = []
        for i, out_channels in enumerate(num_channels):
            dilation = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, 1, dilation, (kernel_size - 1) * dilation,
                              dropout, dropout_type, activation)
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, ksize, dropout, eff_hist,
                 spatial_dropout=False, activation="ReLU", norm="weight_norm", center=0.0, scale=1.0):
        super().__init__()
        dropout_type = "Dropout2d" if spatial_dropout else "Dropout"
        self.tcn = TemporalConvNet(input_size, num_channels, ksize, dropout, dropout_type, activation)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.eff_hist = eff_hist
        self.center = center
        self.scale = scale

    def forward(self, x):
        y = self.tcn((x - self.center) / self.scale)
        return self.linear(y[:, :, -1].contiguous())
