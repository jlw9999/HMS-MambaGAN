import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

model_weights_path = '/HOME/scw6d2x/run/jlw/1232/pretraining/vgg16-397923af.pth'

# 加载预训练模型
vgg = models.vgg16()
vgg.load_state_dict(torch.load(model_weights_path))

# 边界保护损失函数
class BoundaryLoss(nn.Module):
    def __init__(self, lambda_boundary=1.0):
        super(BoundaryLoss, self).__init__()
        self.lambda_boundary = lambda_boundary
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, fake_images, real_images):
        # 计算生成图像的梯度
        fake_gradients_x = torch.abs(
            F.conv2d(fake_images, torch.ones(1, fake_images.shape[1], 1, 3).to(fake_images.device), padding=(0, 1)))
        fake_gradients_y = torch.abs(
            F.conv2d(fake_images, torch.ones(1, fake_images.shape[1], 3, 1).to(fake_images.device), padding=(1, 0)))
        fake_gradients = fake_gradients_x + fake_gradients_y
        # 计算真实图像的梯度
        real_gradients_x = torch.abs(
            F.conv2d(real_images, torch.ones(1, real_images.shape[1], 1, 3).to(real_images.device), padding=(0, 1)))
        real_gradients_y = torch.abs(
            F.conv2d(real_images, torch.ones(1, real_images.shape[1], 3, 1).to(real_images.device), padding=(1, 0)))
        real_gradients = real_gradients_x + real_gradients_y
        # 确保 fake_gradients 和 real_gradients 在同一设备上
        fake_gradients = fake_gradients.to(real_gradients.device)
        # 边界保护损失为梯度的差异
        boundary_loss = F.mse_loss(fake_gradients, real_gradients)
        # 加权
        boundary_loss = self.lambda_boundary * boundary_loss

        return boundary_loss

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
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
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

class DiffusionLoss(nn.Module):
    def __init__(self):
        super(DiffusionLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, real_B, fake_B):
        # 计算扩散过程中的差异损失
        loss_diff = ((fake_B - real_B) ** 2).mean() + self.mse_loss(fake_B, real_B)
        return loss_diff


class ContrastSensitiveLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(ContrastSensitiveLoss, self).__init__()
        self.alpha = alpha

    def forward(self, predicted, target):
        pixel_difference = torch.abs(predicted - target)
        loss = torch.mean(torch.pow((pixel_difference + self.alpha), 2)) - self.alpha**2
        return loss