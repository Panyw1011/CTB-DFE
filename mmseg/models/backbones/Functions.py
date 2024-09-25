import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function, Variable
from torch.nn import Module, parameter


import warnings
try:
    from queue import Queue
except ImportError:
    from Queue import Queue

from torch.nn.modules.batchnorm import _BatchNorm
from functools import partial


from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD



class Encoding(nn.Module):
    def __init__(self, in_channels, num_codes):
        super(Encoding, self).__init__()
        self.in_channels, self.num_codes = in_channels, num_codes
        num_codes = 16       
        self.cw_conv = nn.Conv2d(in_channels, num_codes, 1, bias=False)
        self.cw_bn = nn.BatchNorm2d(num_codes)
        self.cw_act = nn.ReLU(inplace=True)
        

    @staticmethod
    def scaled_l2(x, x2, codewords):
        b, c, h, w = x.size()
        codewords_b, codewords_c, codewords_h, codewords_w =codewords.size()
        codewords = codewords.view((codewords_b, codewords_c, -1))
        codewords_hw = codewords.size(2)
        scale = codewords.mean(dim=2)
        x2_b, x2_hw, x2_c = x2.size()
        expanded_x = x2.unsqueeze(2).expand((x2_b, x2_hw, codewords_c, x2_c))
        reshaped_codewords = codewords.view((codewords_b, codewords_hw, codewords_c, 1))
        reshaped_scale = scale.view((codewords_b, codewords_c, 1))
        scaled_l2_norm = reshaped_scale * (expanded_x - reshaped_codewords).pow(2).sum(dim=1)
        return scaled_l2_norm, codewords

    @staticmethod
    def aggregate(assignment_weights, x, x2, codewords):
        codewords_b, codewords_c, codewords_hw =codewords.size()
        x2_b, x2_hw, x2_c = x2.size()
        reshaped_codewords = codewords.view((codewords_b, codewords_hw, codewords_c, 1))
        expanded_x = x2.unsqueeze(2).expand((x2_b, x2_hw, codewords_c, x2_c))
        assignment_weights = assignment_weights.unsqueeze(1)
        encoded_feat = (assignment_weights * (expanded_x - reshaped_codewords)).sum(1)
        return encoded_feat

    def forward(self, x):
        assert x.dim() == 4 and x.size(1) == self.in_channels
        b, in_channels, w, h = x.size()
        codewords = self.cw_conv(x) 
        codewords = self.cw_bn(codewords)
        codewords = self.cw_act(codewords)
        x2 = x.view(b, self.in_channels, -1).transpose(1, 2).contiguous()
        scaled_l2, codewords = self.scaled_l2(x, x2, codewords)
        assignment_weights = F.softmax(scaled_l2, dim=1)
        encoded_feat = self.aggregate(assignment_weights, x, x2, codewords)
        return encoded_feat


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        expansion = 4
        c = out_channels // expansion

        self.conv1 = nn.Conv2d(in_channels, c, kernel_size=1, stride=1, padding=0, bias=False)  # [64, 256, 1, 1]
        self.bn1 = norm_layer(c)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(c, c, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(c)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(c, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(out_channels)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.residual_bn = norm_layer(out_channels)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) #if x_t_r is None else self.conv2(x + x_t_r)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x


class Mean(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)


class Mlp2(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions. Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)