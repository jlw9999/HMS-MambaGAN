import torch
import torch.nn as nn
import cv2
import numpy as np

class OutlineROI(nn.Module):
    def __init__(self):
        super(OutlineROI, self).__init__()

    def contours_list_to_tensor(self, contours_list):
        # 将列表转换为张量，并添加通道维度
        contours_tensor = torch.tensor(contours_list, dtype=torch.float32).unsqueeze(1)
        return contours_tensor
    def compute_outline_roi(self, tensor):
        contours_list = []
        for i in range(tensor.shape[0]):
            image = tensor[i]
            image = image.squeeze().cpu().detach().numpy()
            image = (image * 255).astype(np.uint8)
            blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
            edges = cv2.Canny(blurred_image, 30, 100)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            outline_roi = np.zeros_like(image, dtype=np.uint8)  # 确保 dtype 是 uint8
            cv2.drawContours(outline_roi, contours, -1, (255), thickness=cv2.FILLED)
            contours_list.append(outline_roi)
        return contours_list

    def forward(self, tensor):
        outline_roi = self.compute_outline_roi(tensor)
        return self.contours_list_to_tensor(outline_roi)


class GLGCM_Loss(nn.Module):
    def __init__(self, num_bins=256):
        super().__init__()
        self.num_bins = num_bins
        self.distance = 1

    def tensor_to_numpy(tensor):
        # 将Tensor转换为NumPy数组
        return tensor.cpu().detach().numpy()

    def numpy_to_tensor(numpy_array):
        # 将NumPy数组转换为Tensor
        return torch.from_numpy(numpy_array).to(torch.float32)

    def compute_gradient(self, image):
        # 将卷积核移动到与输入图像相同的设备上
        device = image.device
        kernel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32, device=device)
        kernel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32, device=device)

        # 计算 x 和 y 方向的梯度
        gradient_x = torch.nn.functional.conv2d(image, kernel_x, padding=1)
        gradient_y = torch.nn.functional.conv2d(image, kernel_y, padding=1)

        # 计算梯度幅值
        gradient = torch.sqrt(gradient_x ** 2 + gradient_y ** 2)
        gradient = (gradient * 255).byte()

        return gradient

    def compute_glgcm(self, gradient):
        # 由于batch中有多个图像，我们需要为每个图像创建一个单独的GLCM
        glgcm_list = []
        for i in range(gradient.shape[0]):
            # 获取当前图像的梯度数据
            current_gradient = gradient[i].cpu()  # 将张量移动到CPU上
            # 创建一个空的GLCM
            glgcm = np.zeros((self.num_bins, self.num_bins))
            # 计算当前图像的GLCM
            for y in range(current_gradient.shape[0]):
                for x in range(current_gradient.shape[1]):
                    current_pixel = current_gradient[y, x]
                    neighbor_x = x + self.distance
                    neighbor_y = y
                    if neighbor_x < current_gradient.shape[1]:
                        neighbor_pixel = current_gradient[neighbor_y, neighbor_x]
                        # 增加共生矩阵对应位置的共生次数
                        glgcm[current_pixel, neighbor_pixel] += 1
                        glgcm[neighbor_pixel, current_pixel] += 1
            # 计算共生概率
            glgcm /= np.sum(glgcm)

            # 将NumPy数组转换为PyTorch张量
            glgcm_tensor = torch.from_numpy(glgcm).unsqueeze(0)  # 添加一个额外的维度
            glgcm_tensor = glgcm_tensor.to(gradient.device)  # 将张量移回原始设备

            # 将当前图像的GLCM添加到列表中
            glgcm_list.append(glgcm_tensor)

        # 将所有图像的GLCM合并成一个单一的张量
        if len(glgcm_list) > 1:
            # 如果有多于一个GLCM，将它们拼接起来
            glgcm_tensor = torch.stack(glgcm_list, dim=0)
        else:
            # 只有一个GLCM时，直接返回它
            glgcm_tensor = glgcm_list[0]

        return glgcm_tensor

    def compute_energy(self, glgcm_tensor):
        energy = torch.sum(glgcm_tensor ** 2)
        return energy

    def compute_correlation(self, glgcm_tensor):
        mean_i, mean_j = torch.meshgrid(torch.arange(glgcm_tensor.shape[-2]).to(glgcm_tensor.device),
                                        torch.arange(glgcm_tensor.shape[-1]).to(glgcm_tensor.device))
        mean_i = torch.sum(mean_i * glgcm_tensor)
        mean_j = torch.sum(mean_j * glgcm_tensor)
        std_i = torch.sqrt(
            torch.sum((mean_i - torch.arange(glgcm_tensor.shape[-2], device=glgcm_tensor.device)) ** 2 * glgcm_tensor))
        std_j = torch.sqrt(
            torch.sum((mean_j - torch.arange(glgcm_tensor.shape[-1], device=glgcm_tensor.device)) ** 2 * glgcm_tensor))

        # Avoid division by zero
        if std_i == 0 or std_j == 0:
            return torch.tensor(0.0, device=glgcm_tensor.device)

        correlation = torch.sum(
            (glgcm_tensor * (mean_i - torch.arange(glgcm_tensor.shape[-2], device=glgcm_tensor.device)).unsqueeze(1) *
             (mean_j - torch.arange(glgcm_tensor.shape[-1], device=glgcm_tensor.device)) / (std_i * std_j)))
        return correlation

    def compute_entropy(self, glgcm_tensor):
        epsilon = 1e-8
        entropy = -torch.sum(glgcm_tensor * torch.log(glgcm_tensor + epsilon))
        return entropy

    def compute_glcm_features(self, glcm):
        # 计算能量、相关性和熵值
        energy = self.compute_energy(glcm)
        correlation = self.compute_correlation(glcm)
        entropy = self.compute_entropy(glcm)

        return energy, correlation, entropy

    def forward(self, real_image, synthesized_image, real_outline, synthesized_outline):
        # 计算真实图像和合成图像的梯度
        real_gradient = self.compute_gradient(real_image)
        synthesized_gradient = self.compute_gradient(synthesized_image)

        # 计算轮廓的梯度
        real_outline_gradient = self.compute_gradient(real_outline)
        synthesized_outline_gradient = self.compute_gradient(synthesized_outline)

        # 计算真实图像和合成图像的 GLGCM
        real_glgcm = self.compute_glgcm(real_gradient)
        synthesized_glgcm = self.compute_glgcm(synthesized_gradient)

        # 计算轮廓的 GLGCM
        real_outline_glgcm = self.compute_glgcm(real_outline_gradient)
        synthesized_outline_glgcm = self.compute_glgcm(synthesized_outline_gradient)

        # 计算真实图像和合成图像的 GLCM 特征
        real_features = self.compute_glcm_features(real_glgcm)
        synthesized_features = self.compute_glcm_features(synthesized_glgcm)

        # 计算轮廓的 GLCM 特征
        real_outline_features = self.compute_glcm_features(real_outline_glgcm)
        synthesized_outline_features = self.compute_glcm_features(synthesized_outline_glgcm)

        # 计算总损失
        energy_diff = torch.mean(torch.abs(real_features[0] - synthesized_features[0]))
        correlation_diff = torch.mean(torch.abs(real_features[1] - synthesized_features[1]))
        entropy_diff = torch.mean(torch.abs(real_features[2] - synthesized_features[2]))

        outline_energy_diff = torch.mean(torch.abs(real_outline_features[0] - synthesized_outline_features[0]))
        outline_correlation_diff = torch.mean(torch.abs(real_outline_features[1] - synthesized_outline_features[1]))
        outline_entropy_diff = torch.mean(torch.abs(real_outline_features[2] - synthesized_outline_features[2]))
        total_loss = energy_diff + correlation_diff + entropy_diff + outline_energy_diff + outline_correlation_diff + outline_entropy_diff

        return total_loss