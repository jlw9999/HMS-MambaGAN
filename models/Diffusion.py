import sys
import torch
import functools
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torchvision.transforms.functional as TF
from PIL import Image
import os
sys.path.insert(0, r'/')
from HMS_MambaGAN.models.Diff_Loss import KLoss, PerceptualLoss, DiffusionLoss, BoundaryLoss, ContrastSensitiveLoss

from networks import ResnetGenerator

class SelfAttention(nn.Module):
    def __init__(self, in_channel, reduction=8):
        super().__init__()
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
    
class DownSample(nn.Module):
    def __init__(self, in_channels):
        super(DownSample, self).__init__()
        # 保持通道数不变
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # 根据需要调整输出通道数
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x

class DiffusionModel(nn.Module):
    def __init__(self, conf):
        super(DiffusionModel, self).__init__()
        self.device = torch.device(conf.device if torch.cuda.is_available() else "cpu")

        self.time_steps = conf.time_steps  # 添加时间步长
        self.k = conf.k  # 较大时间步的步长
        self.beta_min = conf.beta_min  # 噪声方差调度的下限
        self.beta_max = conf.beta_max  # 噪声方差调度的上限
        in_channel = conf.in_channel  # 输入通道数，根据您的实际情况设置
        out_channel = conf.out_channel  # 输出通道数，根据您的实际情况设置
        self.save_file = conf.output_save_path

        image_name = None  # 添加一个新的成员变量来存储图像名称

        self.attention = SelfAttention(in_channel)  # 添加注意力机制
        # 添加上采样和下采样层
        self.downsample = DownSample(in_channel)
        self.upsample = UpSample(in_channel, out_channel)
        # 初始化 ResnetGenerator
        self.generator = ResnetGenerator(in_channel, out_channel, n_blocks=6).to(self.device)

        # 初始化损失函数权重
        self.lambda_mse = 20.0         # MSE损失的权重
        self.lambda_diff = 10.0        # Diff损失的权重
        self.lambda_kl = 10.0          # KL损失的权重
        self.lambda_sensitive = 10      # Sensitive损失的权重
        self.lambda_perceptual = 100.0  # Perceptual损失的权重
        self.lambda_Boundary = 10.0    # Boundary损失的权重

        # 损失函数
        self.criterionMseLoss = nn.MSELoss()
        self.criterionKLoss = KLoss().to(self.device)
        # self.criterionL1 = nn.L1Loss()
        self.criterionDiffLoss = DiffusionLoss().to(self.device)
        self.criterionPerceptual = PerceptualLoss().to(self.device)
        self.criterionSensitive = ContrastSensitiveLoss().to(self.device)
        self.criterionBoundaryL = BoundaryLoss().to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=conf.lr)


    def diffusion_step(self, z, t,apply_gaussian_noise, apply_random_flip, apply_gradient_perturbation,apply_random_rotation):
        if apply_gaussian_noise:
            # noise = torch.randn_like(z) * 0.1  # 调整噪声的幅度
            # z = z + noise
            beta_t = self.beta_min + (self.beta_max - self.beta_min) * t / self.time_steps
            noise = torch.randn_like(z) * beta_t
            z = z + noise
        if apply_random_flip:
            # 明确指定 flip_dims 的设备
            flip_dims = torch.randint(0, 2, size=z.shape[1:], device=z.device) * 2 - 1
            z = z * flip_dims
        if apply_gradient_perturbation:
            z = z
        if apply_random_rotation:
            rotation_angles = torch.randn(z.shape[1], device=z.device) * 0.1
            rotation_matrix = torch.eye(z.shape[1], device=z.device)
            for i in range(z.shape[1]):
                rotation_matrix[i, i] = torch.cos(rotation_angles[i])
                for j in range(i + 1, z.shape[1]):
                    rotation_matrix[i, j] = -torch.sin(rotation_angles[j])
                    rotation_matrix[j, i] = torch.sin(rotation_angles[j])
            z = torch.matmul(rotation_matrix, z.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        return z

    def diffusion_process(self, z, apply_gaussian_noise, apply_random_flip, apply_gradient_perturbation, apply_random_rotation):
        # z = z.to(self.device)
        for t in range(self.time_steps):
            if t % self.k == 0:
                z = self.diffusion_step(z, t, apply_gaussian_noise, apply_random_flip, apply_gradient_perturbation, apply_random_rotation)
        return z

    def save_generated_images(self, upsample_image, image_names, save_dir):
        """
        保存生成的图像到指定目录。
        参数:
        upsample_image -- 生成的图像张量，形状为 [batch_size, 1, H, W]
        image_names -- 生成图像的名称列表
        save_dir -- 图像保存的目录
        """
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        # 转换到 CPU 并转换为 numpy，如果已经在CPU上可以省略 `.cpu()`
        upsample_image = upsample_image.cpu().detach()
        # 保存每个图像
        for i, img_tensor in enumerate(upsample_image):
            # 由于图像只有一个通道，使用 'L' 模式进行保存
            img = TF.to_pil_image(img_tensor.squeeze(0)).convert('L')
            # 使用提供的图像名称作为文件名
            img.save(os.path.join(save_dir, f"{image_names[i]}.png"))

    def forward(self, input_noise, real_image,file_name):
        print("我是扩散模型的forward()方法！")
        input_noise = input_noise.to(self.device)
        real_image = real_image.to(self.device)
        input_noise = input_noise.requires_grad_(True)
        real_image = real_image.requires_grad_(True)
        image_name = file_name  # 存储图像名称
        # 应用下采样
        downsampled_noise = self.downsample(input_noise)

        # 执行扩散过程
        diffused_noise = self.diffusion_process(downsampled_noise, apply_gaussian_noise=True, apply_random_flip=False,
                                                apply_gradient_perturbation=False, apply_random_rotation=False)

        diffused_noise = diffused_noise.to(self.device)
        generated_image = self.generator(diffused_noise)

        self.is_training = 'train' in sys.argv
        if not self.is_training:
            # 在训练模式下执行的操作
            generated_image.requires_grad = True

        upsample_image = self.upsample(generated_image)

        return upsample_image

    def backward(self, upsample_image, real_image):
        # 计算损失
        # 检查是否 loss_mse = loss_diff
        loss_mse = self.criterionMseLoss(upsample_image, real_image) * self.lambda_mse
        loss_diff = self.criterionDiffLoss(real_image, upsample_image) * self.lambda_diff
        kl_div_loss = self.criterionKLoss(upsample_image, real_image) * self.lambda_kl * 5
        sensitive_loss = self.criterionSensitive(upsample_image, real_image) * self.lambda_sensitive * 5
        Boundary_Loss = self.criterionBoundaryL(upsample_image, real_image) * self.lambda_Boundary
        loss_total = loss_diff + kl_div_loss + sensitive_loss  + Boundary_Loss
        if not self.is_training:
            loss_mse.requires_grad = True
            loss_diff.requires_grad = True
            kl_div_loss.requires_grad = True
            sensitive_loss.requires_grad = True
            # perceptual_loss.requires_grad = True
            Boundary_Loss.requires_grad = True
            loss_total.requires_grad = True

        # 反向传播
        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()

    def optimize_parameters(self, input_noise, real_image, image_name):
        print(" strat optimize! ")
        upsample_image = self.forward(input_noise, real_image,image_name)
        self.backward(upsample_image, real_image)
        print(" end optimize! ")


