import cv2
import random
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
from abc import ABC, abstractmethod
from torchvision import transforms
from PIL import Image, ImageOps, ImageEnhance

class CLAHE:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        img_np = np.array(img)
        if len(img_np.shape) == 3:  # 如果是彩色图像
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)  # 转换为灰度图像
        img_clahe = self.clahe.apply(img_np.astype(np.uint8))  # 确保图像为uint8类型
        return Image.fromarray(img_clahe)


class BaseDataset(data.Dataset):
    def __init__(self, conf):
        self.conf = conf
        self.root = conf.dataroot

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, idx):
        pass


def get_transform_face(conf, grayscale=False):
    # 定义图像变换操作
    transform_list = []
    if grayscale:
        print("grayscale为 True ！！！")
        transform_list.append(transforms.Grayscale(1))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    else:
        print("grayscale为 False ！！！")

        transforms.RandomHorizontalFlip()
        transforms.RandomRotation(10)
        # transforms.RandomResizedCrop((256, 256), scale=(0.8, 1.0))
        transforms.ColorJitter(contrast=0.5)
        # transform_list.append(ApplyCLAHE(conf.clahe_clip_limit, conf.clahe_tile_grid_size))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    return transforms.Compose(transform_list)


def get_transform(conf, is2d=True, grayscale=False, method=Image.BICUBIC):
    transforms_list = []
    if grayscale:
        transforms_list.append(transforms.Grayscale(1))
    if 'resize' in conf.preprocess:
        osize = [conf.load_size, conf.load_size]
        transforms_list.append(transforms.Resize(osize, method))
    if 'centercrop' in conf.preprocess:
        transforms_list.append(transforms.CenterCrop(conf.crop_size))
    if 'crop' in conf.preprocess:
        transforms_list.append(transforms.RandomCrop(conf.crop_size))

    if conf.preprocess == 'none':
        transforms_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if conf.flip:
        transforms_list.append(transforms.RandomHorizontalFlip())

    if conf.use_grayscale_normalization:
        transforms_list.append(transforms.Lambda(lambda img: normalize(img)))
    # 添加对比度增强
    if conf.use_contrast_enhancement:
        transforms_list.append(
            transforms.Lambda(lambda img: ImageEnhance.Contrast(img).enhance(conf.contrast_factor)))

    # 添加锐化
    if conf.use_sharpening:
        transforms_list.append(
            transforms.Lambda(lambda img: ImageEnhance.Sharpness(img).enhance(conf.sharpening_factor)))

    # 添加CLAHE处理
    if conf.use_clahe:
        transforms_list.append(ApplyCLAHE(conf.clahe_clip_limit, conf.clahe_tile_grid_size))

    transforms_list.append(transforms.ToTensor())
    if is2d:
        transforms_list.append(transforms.Normalize((0.5,), (0.5,)))
    else:
        transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    return transforms.Compose(transforms_list)


def transform_resize(conf, method=Image.BICUBIC):
    transforms_list = []
    osize = [conf.load_size, conf.load_size]
    transforms_list.append(transforms.Resize(osize, method))
    return transforms.Compose(transforms_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img
    return img.resize((w, h), method)


def normalize(image):
    return (image - image.mean()) / image.std()


def standard(image):  # range in [-1, 1]
    return (image - image.mean()) / (image.max() - image.min())


# 定义CLAHE转换
class ApplyCLAHE:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        # 将PIL图像转换为NumPy数组
        img_np = np.array(img)
        # 如果图像是RGB，先转换为灰度
        if img_np.ndim == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        # 创建并应用CLAHE对象
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        img_clahe = clahe.apply(img_np)
        # 将NumPy数组转回PIL图像
        img = Image.fromarray(img_clahe)
        return img

