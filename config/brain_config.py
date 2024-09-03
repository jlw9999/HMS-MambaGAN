import torch as t
import warnings
# from config.base_config import BaseConfig
from .base_config import BaseConfig

class BrainConfig(BaseConfig):
    load_size = 256
    crop_size = 240
    flip = False
    serial_batch = False

    time_steps = 50
    k=10
    beta_min=0.1
    beta_max=1.0

    device = 'cuda'
    use_contrast_enhancement = False  # True 或 False
    contrast_factor = 2.0  # 对比度增强的强度，>1为增强，<1为减弱
    use_sharpening = False  # 或 True 或 False
    sharpening_factor = 1.2  # 锐化的程度，通常 >1

    # 添加CLAHE配置
    use_clahe = False
    clahe_clip_limit = 1.5
    clahe_tile_grid_size = (4, 4)
    
    # 灰度范围归一化
    use_grayscale_normalization = False  # True 或 False

    preprocess = 'resize'
    dataroot = '/autodl-fs/data/MR_CT_Validation'
    model = 'HMS_GANModel'
    # A = 'flair'
    # B = 't1ce'
    # A = 't1'
    # B = 't2'
    A = 'MR'
    B = 'CT'

    task = 'AtoB'
    in_channel = 1
    out_channel = 1

    apply_gaussian_noise = False       # 添加高斯噪声
    apply_random_flip = False         # 随机翻转某些维度
    apply_gradient_perturbation = False    # 使用梯度信息对潜在空间进行微小扰动
    apply_random_rotation = False      # 随机旋转潜在空间中的某些维度

    save_dir = '/HOME/scw6d2x/run/jlw/experiment/ckpt_t2_t1_HMS_MambaGAN'
    output_save_path = '/HOME/scw6d2x/run/jlw/experiment/ckpt_t2_t1_HMS_MambaGAN/'











