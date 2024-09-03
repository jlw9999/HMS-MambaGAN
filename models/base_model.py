import torch
import torch.nn as nn
import numpy as np
from torch.nn import init
from abc import ABC, abstractmethod
from collections import OrderedDict
from torch.optim import lr_scheduler
import functools
import os
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from networks import get_scheduler

class BaseModel(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.isTrain = conf.isTrain
        self.gpu_ids = conf.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(conf.save_dir, conf.task + conf.model)
        self.output_save_dir = conf.output_save_path
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.output_save_dir):
            os.makedirs(self.output_save_dir)
        self.loss_names = []
        self.visual_names = []
        self.optimizers = []
        self.model_names = []
        self.image_paths = []
        self.metric = 0
        self.emaG = None

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def set_input(self, input):
        pass

    @abstractmethod
    def optimize_parameters(self):
        pass
    def load_network(self, network, network_label, epoch_label, save_dir=''):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        state_dict = torch.load(save_path)
        if isinstance(state_dict, dict):
            network.load_state_dict(state_dict)
        else:
            raise RuntimeError(f'Expected state_dict but got {type(state_dict)}')

    def setup(self, opt):
        if opt.isTrain:
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            if opt.load_iter == 'latest':
                load_suffix = 'latest'
            else:
                load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            print(f'Loading load_suffix from {load_suffix}')
            self.load_network(self.netG, 'G', load_suffix)

    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        pass

    def update_learning_rate(self):
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.conf.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        errors_ret = OrderedDict()

        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def save_networks(self, epoch):
        if epoch == 'latest' and self.emaG:
            self.emaG.apply_shadow()
            print('The latest using EMA.')
        for name in self.model_names:
            if isinstance(name, str):
                
                load_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)

                # if len(self.gpu_ids)>0 and torch.cuda.is_available():
                #     torch.save(net.module.cpu().state_dict(), save_path)
                #     net.cuda(self.gpu_ids[0])
                # else:
                torch.save(net.state_dict(), save_path)


    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    # def load_networks(self, epoch):
    #     for name in self.model_names:
    #         if isinstance(name, str):
    #             load_filename = '%s_net_%s.pth' % (epoch, name)
    #             load_path = os.path.join(self.save_dir, load_filename)
    #             net = getattr(self, 'net' + name)
    #             if isinstance(net, torch.nn.DataParallel):
    #                 net = net.module
    #             print('loading the model from %s' % load_path)
    #
    #             # 如果找不到文件则跳过
    #             if not os.path.isfile(load_path):
    #                 print(f'File {load_path} not found, skipping.')
    #                 continue
    #
    #             # 尝试加载模型
    #             try:
    #                 state_dict = torch.load(load_path, map_location=str(self.device))
    #                 net.load_state_dict(state_dict)
    #             except RuntimeError as e:
    #                 print(f'Error loading {load_filename}: {e}')
    #                 # 加载部分模型参数
    #                 state_dict = torch.load(load_path, map_location=str(self.device))
    #                 new_state_dict = net.state_dict()
    #                 for key, value in state_dict.items():
    #                     if key in new_state_dict and value.size() == new_state_dict[key].size():
    #                         new_state_dict[key] = value
    #                 net.load_state_dict(new_state_dict)
    #                 print(f'Partially loaded {load_filename}')

#     def load_networks(self, epoch):
#         """Load all the networks from the disk.
    
#         Parameters:
#             epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
#         """
#         for name in self.model_names:
#             if isinstance(name, str):
#                 load_filename = '%s_net_%s.pth' % (epoch, name)
#                 load_path = os.path.join(self.save_dir, load_filename)
#                 net = getattr(self, 'net' + name)
#                 if isinstance(net, torch.nn.DataParallel):
#                     net = net.module
#                 print('loading the model from %s' % load_path)
#                 state_dict = torch.load(load_path, map_location=str(self.device))
    
#                 # Check if the state_dict keys start with 'module.' to determine if DataParallel was used
#                 if list(state_dict.keys())[0].startswith('module.'):
#                     new_state_dict = OrderedDict()
#                     for k, v in state_dict.items():
#                         new_state_dict[k[7:]] = v  # remove `module.` prefix
#                     state_dict = new_state_dict
    
#                 net.load_state_dict(state_dict, strict=True)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


class Identity(nn.Module):
    def forward(self, x):
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class Norm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class UnetGenerator(nn.Module):
    def __init__(self, in_channel, out_channel, num_downsample, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        unet_block = UnetBlock(None, ngf * 8, ngf * 8, pre_module=None, norm_layer=norm_layer, use_dropout=use_dropout,
                               inner=True)
        for _ in range(num_downsample - 5):
            unet_block = UnetBlock(None, ngf * 8, ngf * 8, pre_module=unet_block, norm_layer=norm_layer,
                                   use_dropout=use_dropout)
        unet_block = UnetBlock(None, ngf * 4, ngf * 8, pre_module=unet_block, norm_layer=norm_layer)
        unet_block = UnetBlock(None, ngf * 2, ngf * 4, pre_module=unet_block, norm_layer=norm_layer)
        unet_block = UnetBlock(None, ngf, ngf * 2, pre_module=unet_block, norm_layer=norm_layer)
        self.model = UnetBlock(in_channel, out_channel, ngf, pre_module=unet_block, outer=True, norm_layer=norm_layer)

    def forward(self, x):
        return self.model(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super().__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class UnetBlock(nn.Module):
    def __init__(self, in_channel=None, out_channel=1, hidden_channel=1, pre_module=None, inner=False, outer=False,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetBlock, self).__init__()
        self.outer = outer
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if in_channel == None:
            in_channel = out_channel

        downconv = nn.Conv2d(in_channel, hidden_channel, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(hidden_channel)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(out_channel)

        if outer:
            upconv = nn.ConvTranspose2d(hidden_channel * 2, out_channel, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [pre_module] + up
        elif inner:
            upconv = nn.ConvTranspose2d(hidden_channel, out_channel, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(hidden_channel * 2, out_channel, kernel_size=4, stride=2, padding=1,
                                        bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [pre_module] + up + [nn.Dropout(0.5)]
            else:
                model = down + [pre_module] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outer:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], dim=1)


class UnetBlock_with_z(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel, nz=0, pre_module=None, inner=False, outer=False,
                 norm_layer=None, use_dropout=False):
        super(UnetBlock_with_z, self).__init__()
        downconv = []
        self.inner = inner
        self.outer = outer
        self.nz = nz
        in_channel = in_channel + nz
        downconv += [nn.Conv2d(in_channel, hidden_channel, kernel_size=4, stride=2, padding=1)]
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nn.ReLU(True)

        if self.outer:
            upconv = [nn.ConvTranspose2d(hidden_channel * 2, out_channel, kernel_size=4, stride=2, padding=1)]
            down = downconv
            up = [uprelu] + upconv + [nn.Tanh()]
        elif self.inner:
            upconv = [nn.ConvTranspose2d(hidden_channel, out_channel, kernel_size=4, stride=2, padding=1)]
            down = [downrelu] + downconv
            up = [uprelu] + upconv + [norm_layer(out_channel)]
        else:
            upconv = [nn.ConvTranspose2d(hidden_channel * 2, out_channel, kernel_size=4, stride=2, padding=1)]
            down = [downrelu] + downconv + [norm_layer(hidden_channel)]
            up = [uprelu] + upconv + [norm_layer(out_channel)]
            if use_dropout:
                up += [nn.Dropout(0.5)]
        self.down = nn.Sequential(*down)
        self.pre_module = pre_module
        self.up = nn.Sequential(*up)

    def forward(self, x, z):
        if self.nz > 0:
            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
            x_and_z = torch.cat([x, z_img], dim=1)
        else:
            x_and_z = x

        if self.outer:
            x1 = self.down(x_and_z)
            x2 = self.pre_module(x1, z)
            return self.up(x2)
        elif self.inner:
            x1 = self.up(self.down(x_and_z))
            return torch.cat([x1, x], dim=1)
        else:
            x1 = self.down(x_and_z)
            x2 = self.pre_module(x1, z)
            return torch.cat([self.up(x2), x], dim=1)


class D_NLayersMulti(nn.Module):
    def __init__(self, in_channel, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, num_D=1):
        super(D_NLayersMulti, self).__init__()
        self.num_D = num_D
        if num_D == 1:
            layers = self.get_layers(in_channel, ndf, n_layers, norm_layer)
            self.model = nn.Sequential(*layers)
        else:
            layers = self.get_layers(in_channel, ndf, n_layers, norm_layer)
            self.add_module("model_0", nn.Sequential(*layers))
            self.down = nn.AvgPool2d(kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)

            for i in range(1, num_D):
                ndf_i = int(round(ndf / (2 ** i)))
                layers = self.get_layers(in_channel, ndf_i, n_layers, norm_layer)
                self.add_module("model_%d" % i, nn.Sequential(*layers))

    def get_layers(self, in_channel, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(in_channel, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_multi = 1
        nf_multi_prev = 1
        for n in range(1, n_layers):
            nf_multi_prev = nf_multi
            nf_multi = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_multi_prev, ndf * nf_multi, kw, 2, padw),
                norm_layer(ndf * nf_multi),
                nn.LeakyReLU(0.2, True)
            ]

        nf_multi_prev = nf_multi
        nf_multi = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_multi_prev, ndf * nf_multi, kw, 1, padw),
            norm_layer(ndf * nf_multi),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [
            nn.Conv2d(ndf * nf_multi, 1, kw, 1, padw)
        ]
        return sequence

    def forward(self, x):
        if self.num_D == 1:
            return self.model(x)
        result = []
        down = x
        for i in range(self.num_D):
            model = getattr(self, "model_%d" % i)
            result.append(model(down))
            if i != self.num_D - 1:
                down = self.down(down)
        return result


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, norm_layer=None):
        super(ResBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(in_channel)]
        layers += [
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=True)
        ]
        if norm_layer is not None:
            layers += [norm_layer(in_channel)]
        layers += [
            nn.ReLU(True),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=True),
            nn.AvgPool2d(2, 2)
        ]

        self.conv = nn.Sequential(*layers)
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=True)
        )

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


class E_ResNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, ndf=64, n_blocks=4, norm_layer=None, vaeLike=False):
        super(E_ResNet, self).__init__()
        self.vaeLike = vaeLike
        max_ndf = 4
        conv_layers = [
            nn.Conv2d(in_channel, ndf, 4, 2, 1, bias=True)
        ]
        for n in range(1, n_blocks):
            in_ndf = ndf * min(max_ndf, n)
            out_ndf = ndf * min(max_ndf, n + 1)
            conv_layers += [
                ResBlock(in_ndf, out_ndf, norm_layer)
            ]
        conv_layers += [
            nn.ReLU(True),
            nn.AvgPool2d(8)
        ]

        if vaeLike:
            self.fc = nn.Sequential(*[nn.Linear(out_ndf, out_channel)])
            self.fcVar = nn.Sequential(*[nn.Linear(out_ndf, out_channel)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(out_ndf, out_channel)])
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        else:
            return output


# # ***************************************** 原始代码 ***********************************************
class DilatedConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(1, 1), stride=1, groups=1, dilated=(1, 1),
                 norm_layer=None):
        super().__init__()
        padding = tuple(
            [(k - 1) // 2 * d for k, d in zip(kernel_size, dilated)]
        )
        conv = []
        conv += [
            norm_layer(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                      dilation=dilated, bias=False)
        ]
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, groups=1, norm_layer=None):
        super().__init__()
        padding = (kernel_size - 1) // 2
        conv = []
        conv += [
            norm_layer(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                      bias=False)
        ]

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

# ****************************************** 修改的代码 ***********************************************
# class ConvBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, norm_layer=None):
#         super().__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         padding = kernel_size // 2
#         self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=False)
#         self.bn = norm_layer(out_channel)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.relu(self.bn(self.conv(x)))
#         return x


# class DilatedConvBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, kernel_size=(3, 3), stride=1, groups=1, dilated=(1, 1),
#                  norm_layer=None):
#         super().__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         padding = (dilated[0] * (kernel_size[0] - 1) // 2, dilated[1] * (kernel_size[1] - 1) // 2)
#         self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation=dilated, groups=groups,
#                               bias=False)
#         self.bn = norm_layer(out_channel)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.relu(self.bn(self.conv(x)))
#         return x


# class SelfAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(SelfAttention, self).__init__()
#
#         self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
#         self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
#         self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x):
#         batch_size, channels, height, width = x.size()
#
#         query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # (batch_size, HW, C//8)
#         key = self.key(x).view(batch_size, -1, height * width)  # (batch_size, C//8, HW)
#         value = self.value(x).view(batch_size, -1, height * width)  # (batch_size, C, HW)
#
#         attention_weights = torch.bmm(query, key)  # (batch_size, HW, HW)
#         attention_weights = self.softmax(attention_weights)
#
#         out = torch.bmm(value, attention_weights.permute(0, 2, 1))  # (batch_size, C, HW)
#         out = out.view(batch_size, channels, height, width)
#
#         return out


# *************************************** ShuffleNet块 ***********************************************
# class ShuffleNetBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, groups=3, stride=1):
#         super(ShuffleNetBlock, self).__init__()
#         self.groups = groups
#         self.stride = stride
#         self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, groups=groups, bias=False)
#         self.bn1 = nn.BatchNorm2d(in_channel)
#         self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=stride, padding=1, groups=in_channel, bias=False)
#         self.bn2 = nn.BatchNorm2d(in_channel)
#         self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=groups, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_channel)
#
#     def forward(self, x):
#         identity = x
#         out = nn.functional.relu(self.bn1(self.conv1(x)))
#         out = nn.functional.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#
#         if self.stride == 2:
#             identity = nn.functional.avg_pool2d(identity, 3, stride=2, padding=1)
#
#         out = torch.cat([out, identity], 1)
#         out = nn.functional.shuffle(out, self.groups)
#         return out

def shuffle_channels(x, groups):
    """shuffle channels of a 4-D Tensor"""
    batch_size, channels, height, width = x.size()
    assert channels % groups == 0
    channels_per_group = channels // groups
    # split into groups
    x = x.view(batch_size, groups, channels_per_group,
               height, width)
    # transpose 1, 2 axis
    x = x.transpose(1, 2).contiguous()
    # reshape into orignal
    x = x.view(batch_size, channels, height, width)
    return x

class ShuffleNetUnitA(nn.Module):
    """ShuffleNet unit for stride=1"""
    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnitA, self).__init__()
        assert in_channels == out_channels
        assert out_channels % 4 == 0
        bottleneck_channels = out_channels // 4
        self.groups = groups
        self.group_conv1 = nn.Conv2d(in_channels, bottleneck_channels,
                                        1, groups=groups, stride=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.depthwise_conv3 = nn.Conv2d(bottleneck_channels,
                                         bottleneck_channels,
                                         3, padding=1, stride=1,
                                         groups=bottleneck_channels)
        self.bn4 = nn.BatchNorm2d(bottleneck_channels)
        self.group_conv5 = nn.Conv2d(bottleneck_channels, out_channels,
                                     1, stride=1, groups=groups)
        self.bn6 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.group_conv1(x)
        out = F.relu(self.bn2(out))
        out = shuffle_channels(out, groups=self.groups)
        out = self.depthwise_conv3(out)
        out = self.bn4(out)
        out = self.group_conv5(out)
        out = self.bn6(out)
        out = F.relu(x + out)
        return out

class ShuffleNetUnitB(nn.Module):
    """ShuffleNet unit for stride=2"""
    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnitB, self).__init__()
        out_channels -= in_channels
        assert out_channels % 4 == 0
        bottleneck_channels = out_channels // 4
        self.groups = groups
        self.group_conv1 = nn.Conv2d(in_channels, bottleneck_channels,
                                     1, groups=groups, stride=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.depthwise_conv3 = nn.Conv2d(bottleneck_channels,
                                         bottleneck_channels,
                                         3, padding=1, stride=2,
                                         groups=bottleneck_channels)
        self.bn4 = nn.BatchNorm2d(bottleneck_channels)
        self.group_conv5 = nn.Conv2d(bottleneck_channels, out_channels,
                                     1, stride=1, groups=groups)
        self.bn6 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.group_conv1(x)
        out = F.relu(self.bn2(out))
        out = shuffle_channels(out, groups=self.groups)
        out = self.depthwise_conv3(out)
        out = self.bn4(out)
        out = self.group_conv5(out)
        out = self.bn6(out)
        x = F.avg_pool2d(x, 3, stride=2, padding=1)
        out = F.relu(torch.cat([x, out], dim=1))
        return out


class ShuffleNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, groups=1, dilated=[1, 3, 5], norm_layer=None):
        super(ShuffleNetBlock, self).__init__()
        mid = in_channel if in_channel <= out_channel else out_channel

        self.conv_1 = nn.Conv2d(in_channel, in_channel // 4, kernel_size=1, stride=1, bias=False)
        self.bn_1 = norm_layer(in_channel // 4) if norm_layer is not None else None
        self.conv_2 = nn.Conv2d(in_channel // 4, mid, kernel_size=1, stride=1, bias=False)
        self.bn_2 = norm_layer(mid) if norm_layer is not None else None

        self.d_conv = nn.ModuleList()
        for i in range(3):
            self.d_conv.append(
                nn.Conv2d(mid, out_channel, kernel_size=3, stride=stride, padding=dilated[i], dilation=dilated[i],
                          groups=groups, bias=False)
            )
            self.bn_d = norm_layer(out_channel) if norm_layer is not None else None

        self.gconv_3 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), stride=(1, 1), groups=groups,
                                 bias=False)
        self.bn_3 = norm_layer(out_channel) if norm_layer is not None else None

        self.res = nn.Identity() if in_channel == out_channel and stride == 1 else nn.Conv2d(in_channel, out_channel,kernel_size=stride+1,stride=stride,padding=1,bias=False)
        self.bn_res = norm_layer(out_channel) if norm_layer is not None else None

    def forward(self, x):
        res = x
        x = nn.functional.relu(x)
        x = self.conv_1(x)
        if self.bn_1 is not None:
            x = self.bn_1(x)
        x = nn.functional.relu(x)
        x = self.conv_2(x)
        if self.bn_2 is not None:
            x = self.bn_2(x)

        xd = torch.cat([d_conv(x) for d_conv in self.d_conv], dim=1)
        if self.bn_d is not None:
            xd = self.bn_d(xd)
        x = x + self.res(x)
        x = self.gconv_3(x)
        if self.bn_3 is not None:
            x = self.bn_3(x)

        if self.bn_res is not None:
            res = self.bn_res(res)
        return x + res

# ***************************************************************************************************

class SelfAttention(nn.Module):
    def __init__(self, in_channel, reduction=8):
        super().__init__()
        print("*** hello,SelfAttention! ***")

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channel // reduction, in_channel, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        y = self.pool(x)
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.sigmoid(y)
        y = y.view(batch_size, channels, 1, 1)
        out = x * y.expand_as(x)
        return out

class ADBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, groups=1, dilated=[1, 3, 5], norm_layer=None):
        # 输入通道数（in_channel）、输出通道数（out_channel）、步长（stride）、分组数（groups）、空洞卷积的膨胀率（dilated）和标准化层（norm_layer）
        super().__init__()
        # super(ADBlock, self).__init__()
        # self.dilated_convs = nn.ModuleList()
        # for dilation in dilated:
        #     self.dilated_convs.append(
        #         nn.Conv2d(in_channel, out_channel, kernel_size=3, dilation=dilation, padding=dilation))
        #
        # self.self_attention = SelfAttention(out_channel)

        #在初始化过程中，首先确定了一个中间通道数mid，如果输入通道数小于等于输出通道数，则将mid设置为输入通道数，否则设置为输出通道数。
        mid = in_channel if in_channel <= out_channel else out_channel
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w3 = nn.Parameter(torch.ones(1), requires_grad=True)

        # 两个卷积块self.conv_1和self.conv_2。self.conv_1是一个1x1卷积操作，将输入通道数减少到输入通道数的四分之一
        # self.conv_2也是一个1x1卷积操作，将通道数进一步减少到中间通道数mid
        self.conv_1 = ConvBlock(in_channel, in_channel // 4, kernel_size=1, stride=1, norm_layer=norm_layer)
        self.conv_2 = ConvBlock(in_channel // 4, mid, kernel_size=1, stride=1, norm_layer=norm_layer)

        # 这部分代码创建了一个包含三个子模块的列表self.d_conv。对于每个索引i，都创建了一个DilatedConvBlock模块，并将其添加到self.d_conv列表中。
        # 这些DilatedConvBlock模块使用中间通道数mid作为输入通道数，输出通道数为out_channel，卷积核大小为3x3，步长为stride，分组数为groups，
        # 膨胀率为(dilated[i], dilated[i])，标准化层为norm_layer。
        self.d_conv = nn.ModuleList()
        for i in range(3):
            self.d_conv.append(
                DilatedConvBlock(mid, out_channel, kernel_size=(3, 3), stride=stride, groups=groups,
                                 dilated=(dilated[i], dilated[i]), norm_layer=norm_layer)
            )

        # 注意力模块
        self.self_attention = SelfAttention(out_channel)

        # 这部分代码创建了另一个名为 self.gconv_3 的DilatedConvBlock模块。它将输出通道数设置为out_channel，并使用1x3的卷积核进行卷积操作
        # 步长为(1, 1)，分组数为groups，标准化层为norm_layer。
        self.gconv_3 = DilatedConvBlock(out_channel, out_channel, kernel_size=(1, 3), groups=groups, stride=(1, 1),
                                        norm_layer=norm_layer)

        # 这部分代码根据输入的步长stride创建了self.res子模块。如果步长为1，则创建一个1x1的卷积块，将中间通道数mid转换为输出通道数out_channel
        # 如果步长为2，则创建一个2x2的卷积块，同样将中间通道数mid转换为输出通道数out_channel
        if stride == 1:
            self.res = ConvBlock(mid, out_channel, kernel_size=1, stride=1, norm_layer=norm_layer)
        if stride == 2:
            self.res = ConvBlock(mid, out_channel, kernel_size=2, stride=2, norm_layer=norm_layer)

        # skip connection
        # 如果输入通道数in_channel不等于输出通道数out_channel或者步长stride不等于1，那么需要进行跳跃连接操作。
        # 如果步长为1，则创建一个1x1的卷积块self.conv_res，将输入通道数in_channel转换为输出通道数out_channel。如果步长为2，则创建一个2x2的卷积块。
        if in_channel != out_channel or stride != 1:
            if stride == 1:
                self.conv_res = ConvBlock(in_channel, out_channel, kernel_size=1, stride=1, norm_layer=norm_layer)
            if stride == 2:
                self.conv_res = ConvBlock(in_channel, out_channel, kernel_size=2, stride=2, norm_layer=norm_layer)

    def forward(self, x):
        res = x
        # 将x分别通过self.conv_1 和 self.conv_2进行卷积操作
        x = self.conv_1(x)
        x = self.conv_2(x)

        # 对self.d_conv列表中的每个DilatedConvBlock模块进行卷积操作，并对结果进行加权求和。
        # 使用self.w1、self.w2和self.w3作为权重，分别乘以self.d_conv0、self.d_conv1和self.d_conv2，得到xd。
        xd = self.w1 * self.d_conv[0](x) + self.w2 * self.d_conv[1](x) + self.w3 * self.d_conv[2](x)
        # 注意力模块
        xd = self.self_attention(xd)
        x = xd + self.res(x)  # residual connection
        x = self.gconv_3(x)

        # 如果存在self.conv_res模块，则对res进行卷积操作
        if hasattr(self, 'conv_res'):
            res = self.conv_res(res)
        return x + res


class UpADBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, groups=1, dilated=[1, 3, 5], norm_layer=None):
        super().__init__()
        mid = in_channel if in_channel <= out_channel else out_channel

        # 创建一个名为conv_1的卷积块，输入通道数为in_channel，输出通道数为in_channel//4，卷积核大小为1×1，步幅为1。
        self.conv_1 = ConvBlock(in_channel, in_channel // 4, kernel_size=1, stride=1, norm_layer=norm_layer)
        self.conv_2 = ConvBlock(in_channel // 4, mid, kernel_size=1, stride=1, norm_layer=norm_layer)

        # 注意力模块
        self.attention = SelfAttention(mid)

        # 创建一个名为g_conv的序列模块，用于连续应用多个卷积块。
        # 序列中的第一个卷积块的输入通道数为mid，输出通道数为out_channel，卷积核大小为3×3，步幅为stride，分组数为groups，规范化层类型为norm_layer。
        # 第二个卷积块的输入通道数和输出通道数都为out_channel，其他参数与第一个卷积块相同。
        self.g_conv = nn.Sequential(
            ConvBlock(mid, out_channel, kernel_size=3, stride=stride, groups=groups, norm_layer=norm_layer),
            ConvBlock(out_channel, out_channel, kernel_size=3, stride=1, groups=groups, norm_layer=norm_layer)
        )

        # 创建一个名为 gconv_3的空洞卷积块，其中的DilatedConvBlock函数用于定义该卷积块。该卷积块的输入通道数和输出通道数都为out_channel
        # 卷积核大小为(1, 3)，分组数为groups，步幅为(1, 1)，规范化层类型为norm_layer。
        self.gconv_3 = DilatedConvBlock(out_channel, out_channel, kernel_size=(1, 3), groups=groups, stride=(1, 1),
                                        norm_layer=norm_layer)
        # skip connection
        if in_channel != out_channel or stride != 1:
            if stride == 1:
                self.conv_res = ConvBlock(in_channel, out_channel, kernel_size=1, stride=1, norm_layer=norm_layer)
            if stride == 2:
                self.conv_res = ConvBlock(in_channel, out_channel, kernel_size=2, stride=2, norm_layer=norm_layer)

    def forward(self, x):
        res = x
        x = self.conv_1(x)
        x = self.conv_2(x)

        # 调用注意力模块
        # 调用注意力模块attention，对特征图x进行加权融合操作，并将结果赋给变量x
        x = self.attention(x)

        x = self.g_conv(x)

        if hasattr(self, 'conv_res'):
            res = self.conv_res(res)
        return x + res


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, conf):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if conf.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + conf.epoch_count - conf.n_epochs) / float(conf.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif conf.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=conf.lr_decay_iters, gamma=0.1)
    elif conf.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif conf.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=conf.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', conf.lr_policy)
    return scheduler
