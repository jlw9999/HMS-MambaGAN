import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class MultiscaleChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(MultiscaleChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        # 应用多尺度池化
        scales = [F.avg_pool2d(x, s, stride=s) for s in [2, 4, 8]]
        scales = [F.upsample_nearest(s, x.size()[2:]) for s in scales]
        scales.append(x)
        x_combined = torch.cat(scales, dim=1)

        channel_att_sum = None
        for pool_type in self.pool_types:
            for scale in scales:
                if scale.size(2) != x.size(2) or scale.size(3) != x.size(3):
                    scale = F.interpolate(scale, size=x.size()[2:], mode='nearest')

                if pool_type == 'avg':
                    avg_pool = F.avg_pool2d(scale, (scale.size(2), scale.size(3)),
                                            stride=(scale.size(2), scale.size(3)))
                    channel_att_raw = self.mlp(avg_pool)
                elif pool_type == 'max':
                    max_pool = F.max_pool2d(scale, (scale.size(2), scale.size(3)),
                                            stride=(scale.size(2), scale.size(3)))
                    channel_att_raw = self.mlp(max_pool)
                elif pool_type == 'lp':
                    lp_pool = F.lp_pool2d(scale, 2, (scale.size(2), scale.size(3)),
                                          stride=(scale.size(2), scale.size(3)))
                    channel_att_raw = self.mlp(lp_pool)
                elif pool_type == 'lse':
                    lse_pool = logsumexp_2d(scale)
                    channel_att_raw = self.mlp(lse_pool)

                if channel_att_sum is None:
                    channel_att_sum = channel_att_raw
                else:
                    channel_att_sum = channel_att_sum + channel_att_raw
        # print(f"Shape of channel_att_sum before unsqueeze: {channel_att_sum.shape}")
        if channel_att_sum.dim() == 1:
            channel_att_sum = channel_att_sum.unsqueeze(0)
        if channel_att_sum.dim() == 2:
            channel_att_sum = channel_att_sum.unsqueeze(2).unsqueeze(3)

        scale = F.sigmoid(channel_att_sum).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + x
        return out

class MultiscaleSelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(MultiscaleSelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim * 4, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim * 4, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim * 4, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        # 应用多尺度池化
        scales = [F.avg_pool2d(x, s, stride=s) for s in [2, 4, 8]]
        scales = [F.interpolate(s, size=x.size()[2:], mode='nearest') for s in scales]
        scales.append(x)
        x_combined = torch.cat(scales, dim=1)  # 通道数变成了原来的4倍
        # Query, Key, Value 计算
        proj_query = self.query_conv(x_combined).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x_combined).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x_combined).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + x

        return out

class MultiscaleSpatialGate(nn.Module):
    def __init__(self):
        super(MultiscaleSpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        # 应用多尺度池化
        scales = [F.avg_pool2d(x, s, stride=s) for s in [2, 4, 8]]
        scales = [F.upsample_nearest(s, x.size()[2:]) for s in scales]
        scales.append(x)
        x_combined = torch.cat(scales, dim=1)
        x_compress = self.compress(x_combined)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale
class HMSBlock(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(HMSBlock, self).__init__()
        self.ChannelGate = MultiscaleChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        # self.SelfAttention = SelfAttention(gate_channels)
        self.SelfAttention = MultiscaleSelfAttention(gate_channels)
        if not no_spatial:
            self.SpatialGate = MultiscaleSpatialGate()

    def forward(self, x):
        # print("我是 HMSBlock！")
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        x_out = self.SelfAttention(x_out)
        return x_out

class HMSModule(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(HMSModule, self).__init__()
        self.hms1 = HMSBlock(gate_channels, reduction_ratio, pool_types, no_spatial)
        self.hms2 = HMSBlock(gate_channels, reduction_ratio, pool_types, no_spatial)
        self.hms3 = HMSBlock(gate_channels, reduction_ratio, pool_types, no_spatial)

        self.downsample1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downsample2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x):
        # Original scale
        x1 = self.hms1(x)
        # Downscale once
        x_down1 = self.downsample1(x)
        x2 = self.hms2(x_down1)
        x2_up = self.upsample1(x2)
        # Downscale twice
        x_down2 = self.downsample2(x_down1)
        x3 = self.hms3(x_down2)
        x3_up = self.upsample2(x3)
        # Fusion of multi-scale features
        x_out = x1 + x2_up + x3_up

        return x_out
