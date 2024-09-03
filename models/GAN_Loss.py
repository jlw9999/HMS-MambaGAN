import sys
sys.path.insert(0, r'/HOME/scw6d2x/run/jlw')
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import math
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import vgg19
from torchvision.models import inception_v3, vgg16
from distutils.version import LooseVersion
from pytorch_msssim import SSIM  # 需要安装 pytorch_msssim 库
from torch.autograd import Variable
from torch import Tensor
import warnings
from typing import List, Optional, Tuple, Union
import torchvision.models as models
# 手动下载预训练模型文件，并存储到指定路径
model_weights_path = '/HOME/scw6d2x/run/jlw/1232/pretraining/vgg16-397923af.pth'
# 加载预训练模型
vgg = models.vgg16()
vgg.load_state_dict(torch.load(model_weights_path))

def _fspecial_gauss_1d(size: int, sigma: float) -> Tensor:
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input: Tensor, win: Tensor) -> Tensor:
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


def _ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float,
    win: Tensor,
    size_average: bool = True,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03)
) -> Tuple[Tensor, Tensor]:
    K1, K2 = K
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

# 假设 _fspecial_gauss_1d 和 _ssim 函数已经定义

class MS_SSIM_Loss(nn.Module):
    def __init__(
        self,
        data_range: float = 255,
        size_average: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        weights: Optional[List[float]] = None,
        K: Union[Tuple[float, float], List[float]] = (0.01, 0.03)
    ):
        super(MS_SSIM_Loss, self).__init__()
        self.data_range = data_range
        self.size_average = size_average
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.weights = weights if weights is not None else [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        self.K = K

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        if not X.shape == Y.shape:
            raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

        for d in range(len(X.shape) - 1, 1, -1):
            X = X.squeeze(dim=d)
            Y = Y.squeeze(dim=d)

        if len(X.shape) == 4:
            avg_pool = F.avg_pool2d
        elif len(X.shape) == 5:
            avg_pool = F.avg_pool3d
        else:
            raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

        if not (self.win_size % 2 == 1):
            raise ValueError("Window size should be odd.")

        smaller_side = min(X.shape[-2:])
        assert smaller_side > (self.win_size - 1) * (2 ** 4), \
            "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((self.win_size - 1) * (2 ** 4))

        weights_tensor = X.new_tensor(self.weights)

        win = _fspecial_gauss_1d(self.win_size, self.win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

        levels = weights_tensor.shape[0]
        mcs = []
        for i in range(levels):
            ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=self.data_range, size_average=False, K=self.K)

            if i < levels - 1:
                mcs.append(torch.relu(cs))
                padding = [s % 2 for s in X.shape[2:]]
                X = avg_pool(X, kernel_size=2, padding=padding)
                Y = avg_pool(Y, kernel_size=2, padding=padding)

        ssim_per_channel = torch.relu(ssim_per_channel)  # type: ignore  # (batch, channel)
        mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
        ms_ssim_val = torch.prod(mcs_and_ssim ** weights_tensor.view(-1, 1, 1), dim=0)

        if self.size_average:
            return ms_ssim_val.mean()
        else:
            return ms_ssim_val.mean(1)



def abs_criterion(in_, target):
    return torch.mean(torch.abs(in_ - target))

class ContrastLoss(nn.Module):
    def __init__(self):
        super(ContrastLoss, self).__init__()

    def forward(self, synthesized_T2, real_T2):
        # 计算真实T2图像的对比度
        real_T2_gray = torch.mean(real_T2, dim=1, keepdim=True)
        real_T2_std = torch.std(real_T2_gray)

        # 计算合成T2图像的对比度
        synthesized_T2_gray = torch.mean(synthesized_T2, dim=1, keepdim=True)
        synthesized_T2_std = torch.std(synthesized_T2_gray)

        # 计算对比度损失，这里使用绝对值来衡量对比度差异
        contrast_loss = torch.abs(real_T2_std - synthesized_T2_std)

        return contrast_loss

class ShapeConsistencyLoss(nn.Module):
    def __init__(self):
        super(ShapeConsistencyLoss, self).__init__()
        # Sobel filter for edge detection
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view((1, 1, 3, 3))
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view((1, 1, 3, 3))
        self.sobel_x.weight = nn.Parameter(sobel_kernel_x, requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_kernel_y, requires_grad=False)

    def forward(self, input, target):
        edge_input_x = self.sobel_x(input)
        edge_input_y = self.sobel_y(input)
        edge_target_x = self.sobel_x(target)
        edge_target_y = self.sobel_y(target)
        loss = F.mse_loss(edge_input_x, edge_target_x) + F.mse_loss(edge_input_y, edge_target_y)
        return loss

class FeatureReconstructionLoss(nn.Module):
    def __init__(self, feature_layer=8):
        super(FeatureReconstructionLoss, self).__init__()
        self.vgg = vgg19(pretrained=True).features[:feature_layer].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        # 检查输入图像是否为单通道，如果是，则复制到三个通道
        if input.size(1) == 1:
            input = input.repeat(1, 3, 1, 1)
        if target.size(1) == 1:
            target = target.repeat(1, 3, 1, 1)

        input_features = self.vgg(input)
        target_features = self.vgg(target)
        loss = nn.functional.mse_loss(input_features, target_features)
        return loss


class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()

    def forward(self, input_features, target_features):
        # 计算内容损失为特征之间的均方误差
        loss = nn.MSELoss()(input_features, target_features)
        return loss

class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=批次大小，b=特征图数量，(c,d)=特征图的维度
        features = input.view(a * b, c * d)  # 将特征图转换为二维矩阵
        G = torch.mm(features, features.t())  # 计算格拉姆矩阵
        return G.div(a * b * c * d)

    def forward(self, input_features, target_features):
        G_input = self.gram_matrix(input_features)
        G_target = self.gram_matrix(target_features)
        loss = nn.MSELoss()(G_input, G_target)
        return loss


class HistogramAndEdgeLoss(nn.Module):
    def __init__(self, window_size=8, bins=256):
        super(HistogramAndEdgeLoss, self).__init__()
        self.compute_histograms = ComputeHistograms(window_size, bins)
        self.compute_edge_attributes = ComputeEdgeAttributes(window_size)
        self.l1_loss = nn.L1Loss()  # 使用L1Loss

    def forward(self,fake_images, real_images):
        # 计算真实图像和生成图像的强度直方图和边缘属性直方图
        real_hist = self.compute_histograms(real_images)
        fake_hist = self.compute_histograms(fake_images)
        real_edge = self.compute_edge_attributes(real_images)
        fake_edge = self.compute_edge_attributes(fake_images)

        # 计算直方图和边缘属性直方图的L1损失
        hist_loss = self.l1_loss(fake_hist, real_hist)
        edge_loss = self.l1_loss(fake_edge, real_edge)

        # 综合两种损失
        total_loss = hist_loss + edge_loss
        return total_loss
1
# 计算强度直方图
class ComputeHistograms(nn.Module):
    def __init__(self, window_size=8, bins=256, min_value=None, max_value=None):
        super(ComputeHistograms, self).__init__()
        self.window_size = window_size
        self.bins = bins
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, input_tensor):
        # if self.min_value is None or self.max_value is None:
        #     min_value, max_value = input_tensor.min(), input_tensor.max()
        # else:
        #     min_value, max_value = self.min_value, self.max_value
        if self.min_value is None or self.max_value is None:
            min_value = input_tensor.min().item()  # 转换为数值
            max_value = input_tensor.max().item()  # 转换为数值
        else:
            # 假设self.min_value和self.max_value已经是数值类型，或者在类的构造器中正确处理
            min_value = self.min_value
            max_value = self.max_value

        batch_size, channels, height, width = input_tensor.shape
        histograms = torch.zeros((batch_size, height // self.window_size, width // self.window_size, self.bins), device=input_tensor.device)

        for i in range(0, height, self.window_size):
            for j in range(0, width, self.window_size):
                window = input_tensor[:, :, i:i + self.window_size, j:j + self.window_size]
                window = window.reshape(batch_size, -1)
                # hist = window.histc(bins=self.bins, min=min_value, max=max_value, max_bins=self.bins)
                hist = window.histc(bins=self.bins, min=min_value, max=max_value)

                histograms[:, i // self.window_size, j // self.window_size, :] = hist

        return histograms

# 计算边缘属性直方图
class ComputeEdgeAttributes(nn.Module):
    def __init__(self, window_size=8):
        super(ComputeEdgeAttributes, self).__init__()
        self.window_size = window_size
        self.compute_histograms = ComputeHistograms(window_size=window_size)
        # Sobel算子定义为不需要梯度的参数
        self.sobel_x = nn.Parameter(torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3), requires_grad=False)
        self.sobel_y = nn.Parameter(torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3), requires_grad=False)

    def forward(self, input_tensor):

        edge_x = F.conv2d(input_tensor, self.sobel_x, padding=1)
        edge_y = F.conv2d(input_tensor, self.sobel_y, padding=1)
        edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        edge_direction = torch.atan2(edge_y, edge_x)

        edge_attributes = self.compute_histograms(edge_magnitude) + self.compute_histograms(edge_direction)

        return edge_attributes


# 加载预训练模型
vgg = models.vgg16()
vgg.load_state_dict(torch.load(model_weights_path))

class Lambda(nn.Module):
    """Wraps a callable in an :class:`nn.Module` without registering it."""

    def __init__(self, func):
        super().__init__()
        object.__setattr__(self, 'forward', func)

    def extra_repr(self):
        return getattr(self.forward, '__name__', type(self.forward).__name__) + '()'


class WeightedLoss(nn.ModuleList):
    """A weighted combination of multiple loss functions."""

    def __init__(self, losses, weights, verbose=False):
        super().__init__()
        for loss in losses:
            self.append(loss if isinstance(loss, nn.Module) else Lambda(loss))
        self.weights = weights
        self.verbose = verbose

    def _print_losses(self, losses):
        for i, loss in enumerate(losses):
            print(f'({i}) {type(self[i]).__name__}: {loss.item()}')

    def forward(self, *args, **kwargs):
        losses = []
        for loss, weight in zip(self, self.weights):
            losses.append(loss(*args, **kwargs) * weight)
        if self.verbose:
            self._print_losses(losses)
        return sum(losses)


class TVLoss(nn.Module):
    """Total variation loss (Lp penalty on image gradient magnitude).

    The input must be 4D. If a target (second parameter) is passed in, it is ignored.
    ``p=1`` yields the vectorial total variation norm. It is a generalization
    of the originally proposed (isotropic) 2D total variation norm (see
    (see https://en.wikipedia.org/wiki/Total_variation_denoising) for color
    images. On images with a single channel it is equal to the 2D TV norm.

    ``p=2`` yields a variant that is often used for smoothing out noise in
    reconstructions of images from neural network feature maps (see Mahendran
    and Vevaldi, "Understanding Deep Image Representations by Inverting
    Them", https://arxiv.org/abs/1412.0035)

    :attr:`reduction` can be set to ``'mean'``, ``'sum'``, or ``'none'``
    similarly to the loss functions in :mod:`torch.nn`. The default is
    ``'mean'``.
    """

    def __init__(self, p, reduction='mean', eps=1e-8):
        super().__init__()
        if p not in {1, 2}:
            raise ValueError('p must be 1 or 2')
        if reduction not in {'mean', 'sum', 'none'}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.p = p
        self.reduction = reduction
        self.eps = eps

    def forward(self, input, target=None):
        input = F.pad(input, (0, 1, 0, 1), 'replicate')
        x_diff = input[..., :-1, :-1] - input[..., :-1, 1:]
        y_diff = input[..., :-1, :-1] - input[..., 1:, :-1]
        diff = x_diff**2 + y_diff**2
        if self.p == 1:
            diff = (diff + self.eps).mean(dim=1, keepdims=True).sqrt()
        if self.reduction == 'mean':
            return diff.mean()
        if self.reduction == 'sum':
            return diff.sum()
        return diff


class VGGLoss(nn.Module):
    models = {'vgg16': models.vgg16}
    model_paths = {'vgg16':  '/HOME/scw6d2x/run/jlw/1232/pretraining/vgg16-397923af.pth'}

    def __init__(self, model='vgg16', layer=8, shift=0, reduction='mean'):
        super().__init__()
        self.shift = shift
        self.reduction = reduction
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        # 加载预训练模型
        vgg = self.models[model]()
        model_weights_path = self.model_paths[model]
        vgg.load_state_dict(torch.load(model_weights_path))

        # 截断到指定层
        self.model = vgg.features[:layer + 1]
        self.model.eval()
        self.model.requires_grad_(False)

        # self.model = self.models[model](pretrained=True).features[:layer+1]
        # self.model.eval()
        # self.model.requires_grad_(False)

    def get_features(self, input):
        if input.shape[1] == 1:  # If input is single-channel
            input = input.repeat(1, 3, 1, 1)  # Repeat the single channel to create 3 channels
        return self.model(self.normalize(input))

    def train(self, mode=True):
        self.training = mode

    def forward(self, input, target, target_is_features=False):
        if target_is_features:
            input_feats = self.get_features(input)
            target_feats = target
        else:
            sep = input.shape[0]
            batch = torch.cat([input, target])
            if self.shift and self.training:
                padded = F.pad(batch, [self.shift] * 4, mode='replicate')
                batch = transforms.RandomCrop(batch.shape[2:])(padded)
            feats = self.get_features(batch)
            input_feats, target_feats = feats[:sep], feats[sep:]
        return F.mse_loss(input_feats, target_feats, reduction=self.reduction)


class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss

    # def __call__(self, prediction, target_is_real, generated_images):
    #     target_tensor = self.get_target_tensor(prediction, target_is_real)
    #     loss = self.loss(prediction, target_tensor)
    #     # 添加正则项
    #     reg_loss = torch.mean(torch.abs(generated_images[:, :, :, :-1] - generated_images[:, :, :, 1:])) + \
    #                torch.mean(torch.abs(generated_images[:, :, :-1, :] - generated_images[:, :, 1:, :]))
    #     loss += 0.1 * reg_loss  # 调整正则项的权重
    #
    #     return loss

class HistLoss(nn.Module):
    def __init__(self, bins=100):
        super().__init__()
        self.bins = bins
        self.loss = nn.L1Loss()

    def _min_max(self, img1, img2):
        self.minv = float(min(img1.min(), img2.min()))
        self.maxv = float(max(img1.max(), img2.max()))

    def _histc(self, img):
        if LooseVersion(torch.__version__) >= LooseVersion("1.10.0"):
            # 将数据从 GPU 移动到 CPU
            img = img.cpu()
            histc, bins = torch.histogram(img, bins=self.bins,
                                          range=(self.minv + 0.1, self.maxv))  # for PyTorch>=1.10 version
            return histc, bins
        else:
            histc = torch.histc(img, bins=self.bins, min=self.minv + 0.1, max=self.maxv)
            return histc, None

    def forward(self, prediction, target):
        self._min_max(prediction, target)
        histc_p, _ = self._histc(prediction.detach())
        histc_t, _ = self._histc(target.detach())
        loss = self.loss(histc_p, histc_t)
        return loss

def compute_gradient_penalty(Discriminator, real_sample, fake_sample):
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    alpha = Tensor(np.random.random((real_sample.size(0), 1, 1, 1)))
    interpolates = (alpha * real_sample + (1 - alpha) * fake_sample).requires_grad_(True)
    d_interpolates = Discriminator(interpolates)
    grad_tensor = Variable(Tensor(d_interpolates.size(0), 1, d_interpolates.size(2), d_interpolates.size(3)).fill_(1.0),
                           requires_grad=False)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_tensor,
        create_graph=True,      # 设置为True可以计算更高阶的导数
        retain_graph=True ,    # 设置为True可以重复调用backward
        only_inputs=True,       # 默认为True，如果为True，则只会返回指定input的梯度值。 若为False，则会计算所有叶子节点的梯度，
    )[0]  # return a tensor list, get index 0.
    gradients = gradients.view(real_sample.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class KLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, prediction, target):
        prediction = prediction.view(prediction.size(0), -1)
        target = target.view(target.size(0), -1)

        prediction = F.log_softmax(prediction, dim=1)
        target = F.log_softmax(target, dim=1)

        loss = self.loss(prediction, target)
        return loss


class GDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.pairwise_p_distance = torch.nn.PairwiseDistance(p=1.0)

    def forward(self, correct_images, generated_images):
        correct_images_gradient_x = self.calculate_x_gradient(correct_images)
        generated_images_gradient_x = self.calculate_x_gradient(generated_images)
        correct_images_gradient_y = self.calculate_y_gradient(correct_images)
        generated_images_gradient_y = self.calculate_y_gradient(generated_images)

        distances_x_gradient = self.pairwise_p_distance(
            correct_images_gradient_x, generated_images_gradient_x
        )
        distances_y_gradient = self.pairwise_p_distance(
            correct_images_gradient_y, generated_images_gradient_y
        )
        loss_x_gradient = torch.mean(distances_x_gradient)
        loss_y_gradient = torch.mean(distances_y_gradient)
        loss = 0.5 * (loss_x_gradient + loss_y_gradient)
        return loss

    def calculate_x_gradient(self, images):
        x_gradient_filter = torch.Tensor(
            [
                [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
            ]
        ).cuda()
        x_gradient_filter = x_gradient_filter.view(1, 1, 3, 3)
        result = torch.functional.F.conv2d(
            images, x_gradient_filter, groups=1, padding=(1, 1)
        )
        return result

    def calculate_y_gradient(self, images):
        y_gradient_filter = torch.Tensor(
            [
                [[0, 1, 0], [0, 0, 0], [0, -1, 0]],
            ]
        ).cuda()
        y_gradient_filter = y_gradient_filter.view(1, 1, 3, 3)
        result = torch.functional.F.conv2d(
            images, y_gradient_filter, groups=1, padding=(1, 1)
        )
        return result

class DiffusionLoss(nn.Module):
    def __init__(self):
        super(DiffusionLoss, self).__init__()

    def forward(self, real_B, fake_B):
        # 计算扩散过程中的差异损失
        loss_diff = ((fake_B - real_B)**2).mean()
        return loss_diff

class SingleChannelVGG(nn.Module):
    def __init__(self, feature_layer=8):
        super(SingleChannelVGG, self).__init__()
        # 加载预训练的VGG16模型
        # vgg_pretrained_features = vgg16(pretrained=True).features

        model_weights_path = '/HOME/scw6d2x/run/jlw/1232/pretraining/vgg16-397923af.pth'
        vgg = models.vgg16()
        vgg.load_state_dict(torch.load(model_weights_path))
        vgg_pretrained_features = vgg.features

        # 修改第一个卷积层以接受单通道输入
        # self.features = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=3, padding=1),
        #     *vgg_pretrained_features[1:feature_layer]  # 使用VGG的后续层
        # )
        self.features = nn.Sequential(
            # 修改第一个卷积层以接受单通道输入
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # 添加Batch Normalization层
            *vgg_pretrained_features[1:feature_layer]  # 使用VGG的后续层
        )

    def forward(self, x):
        return self.features(x)

class PerceptualLoss(nn.Module):
    def __init__(self, feature_layer=8, device='cuda'):
        super(PerceptualLoss, self).__init__()
        self.vgg = SingleChannelVGG(feature_layer).to(device).eval()

    def forward(self, generated_image, target_image):
        generated_features = self.vgg(generated_image)
        target_features = self.vgg(target_image)
        return nn.functional.mse_loss(generated_features, target_features)
