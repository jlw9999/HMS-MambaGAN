import torch
from math import exp
from PIL import Image
import numpy as np

def load_image_as_numpy(path):
    image = Image.open(path).convert('L')
    image_np = np.array(image)
    return image_np

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()
# ------------------------------------- MAE  -------------------------------------------
def calculate_mae_numpy(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("The two images must have the same size.")

    mae = np.mean(np.abs(image1 - image2))
    return mae
# ------------------------------------- ALSE -------------------------------------------
def ALSE_solve(X, Y):
    N = X.size  # 获取数据点个数
    # 计算每个数据点的平方距离
    mean_X = np.mean(X)
    mean_Y = np.mean(Y)
    dist_X = np.linalg.norm(mean_X - X, axis=1)
    dist_Y = np.linalg.norm(mean_Y - Y, axis=1)

    # 添加防御性编程，将距离为零的值替换为一个很小的正数
    epsilon = 1e-8
    dist_X[dist_X == 0] = epsilon
    dist_Y[dist_Y == 0] = epsilon

    # 计算 log 差的绝对值
    abs_log_diff = np.abs(np.log(dist_X) - np.log(dist_Y))
    # 计算平均值
    alse_value = 1 / N * np.sum(abs_log_diff)
    return alse_value

# -----------------------------------MSE -----------------------------------------------
def MSE(img1,img2):
    mse = np.mean( (img1 - img2) ** 2 )
    return mse
