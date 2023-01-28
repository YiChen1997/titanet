import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1dSamePadding(nn.Conv1d):
    """
    1D convolutional layer with "same" padding (no downsampling),
    that is also compatible with strides > 1
    """

    def __init__(self, *args, **kwargs):
        super(Conv1dSamePadding, self).__init__(*args, **kwargs)

    def forward(self, inputs):
        """
        Given an input of size [B, CI, WI], return an output
        [B, CO, WO], where WO = [CI + 2P - K - (K - 1) * (D - 1)] / S + 1,
        by computing P on-the-fly ay forward time

        B: batch size
        CI: input channels
        WI: input width
        CO: output channels
        WO: output width
        P: padding
        K: kernel size
        D: dilation
        S: stride
        """
        padding = (
                          self.stride[0] * (inputs.shape[-1] - 1)
                          - inputs.shape[-1]
                          + self.kernel_size[0]
                          + (self.dilation[0] - 1) * (self.kernel_size[0] - 1)
                  ) // 2
        return self._conv_forward(
            F.pad(inputs, (padding, padding)),
            self.weight,
            self.bias,
        )


class DepthwiseConv1d(nn.Module):
    """
    Compute a depth-wise separable convolution, by performing
    a depth-wise convolution followed by a point-wise convolution

    "Xception: Deep Learning with Depthwise Separable Convolutions",
    Chollet, https://arxiv.org/abs/1610.02357
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=1,
            bias=True,
            device=None,
            dtype=None,
    ):
        super(DepthwiseConv1d, self).__init__()
        self.conv = nn.Sequential(
            Conv1dSamePadding(  # 深度可分离卷积是让output_channels与input_channels相同，同时groups等于input_channels
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=in_channels,
                bias=bias,
                device=device,
                dtype=dtype,
            ),  # 这时输出通道与输入通道相同，如果需要改变输出通道数，则再跟一个kernel等于1的1D卷积即可。
            Conv1dSamePadding(
                in_channels, out_channels, kernel_size=1, device=device, dtype=dtype
            ),
        )

    def forward(self, inputs):
        """
        Given an input of size [B, CI, WI], return an output
        [B, CO, WO], where CO is given as a parameter and WO
        depends on the convolution operation attributes

        B: batch size
        CI: input channels
        WI: input width
        CO: output channels
        WO: output width
        """
        return self.conv(inputs)


class ConvBlock1d(nn.Module):
    """
    Standard convolution, normalization, activation block
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=1,
            activation="relu",
            dropout=0,
            depthwise=False,
    ):
        super(ConvBlock1d, self).__init__()
        assert activation is None or activation in (
            "relu",
            "tanh",
        ), "Incompatible activation function"

        # Define architecture
        conv_module = DepthwiseConv1d if depthwise else Conv1dSamePadding
        modules = [
            conv_module(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_channels),
        ]
        if activation is not None:
            modules += [nn.ReLU() if activation == "relu" else nn.Tanh()]
        if dropout > 0:
            modules += [nn.Dropout(p=dropout)]
        self.conv_block = nn.Sequential(*modules)

    def forward(self, inputs):
        """
        Given an input of size [B, CI, WI], return an output
        [B, CO, WO], where CO is given as a parameter and WO
        depends on the convolution operation attributes

        B: batch size
        CI: input channels
        WI: input width
        CO: output channels
        WO: output width
        """
        return self.conv_block(inputs)


class SqueezeExcitation(nn.Module):
    """
    The SE layer squeezes a sequence of local feature vectors into
    a single global context vector, broadcasts this context back to
    each local feature vector, and merges the two via multiplications

    "Squeeze-and-Excitation Networks", Hu et al.,
    https://arxiv.org/abs/1709.01507
    """

    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()

        # Define architecture
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        """
        Given an input of shape [B, C, W], returns an
        output of the same shape

        B: batch size
        C: number of channels
        W: input width
        """
        # [B, C, W] -> [B, C]
        squeezed = self.squeeze(inputs).squeeze(-1)

        # [B, C] -> [B, C]
        excited = self.excitation(squeezed).unsqueeze(-1)

        # [B, C] -> [B, C, W]
        return inputs * excited.expand_as(inputs)


class Bottle2neck(nn.Module):
    """
    ResNet + TDNN + SE-Block
    """

    def __init__(self, channels, reduction=16, kernel_size=3, dilation=2):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(channels / reduction))
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * reduction)  # dim, 输入与输出维度
        self.nums = reduction - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * reduction, channels, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SqueezeExcitation(channels, reduction=reduction)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        out += residual  # input + se_block的输出
        return out


class Squeeze(nn.Module):
    """
    Remove dimensions of size 1 from the input tensor
    """

    def __init__(self, dim=None):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        return inputs.squeeze(self.dim)


class CoordAtt(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        mid_channels = max(8, in_channels // reduction)
        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm1d(mid_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        identity = inputs
        b, c, w = inputs.size()
        inputs_pool = self.pool(inputs)
        y = self.conv1(inputs_pool)
        y = self.bn(y)
        y = self.relu(y)
        ca = self.conv2(y).sigmoid()
        return identity * ca
