import os
import sys
import torch
import time
import models
import Calculation

from models import *
from sklearn import metrics
from torch.utils.data import DataLoader
from data.brain_dataset import BrainDataset
from config.brain_config import BrainConfig
from torch.nn.utils import clip_grad_norm_
from skimage.metrics import structural_similarity as SSIM_function
from skimage.metrics import mean_squared_error as mse_function
from PIL import Image
from HMS_MambaGAN.metrics.metrics_compute import *


def train(**kwargs):
    brain_config = BrainConfig()
    brain_config._parse(kwargs)
    brain_dataset = BrainDataset(brain_config)
    brain_dataset = DataLoader(brain_dataset, batch_size = 12, shuffle=True, num_workers=0, pin_memory=True)
    save_dir = brain_config.output_save_path
    model = HMS_GANModel(brain_config)
    # 用于计算平均训练时间和内存负载
    total_training_time = 0
    total_iterations = 0
    max_memory_load = 0

    for epoch in range(brain_config.epoch_count, brain_config.n_epochs + brain_config.n_epochs_decay + 1):
        model.update_learning_rate()
        if hasattr(model, 'alpha'):
            model.alpha = [0.0]
            print('Alpha updated.')
        for i, data in enumerate(brain_dataset):
            start_time = time.time()  # 开始时间
            model.set_input(data)
            model.optimize_parameters()
            total_norm = clip_grad_norm_(model.parameters(), max_norm=float('inf'))

            end_time = time.time()  # 结束时间
            iteration_time = end_time - start_time
            total_training_time += iteration_time
            total_iterations += 1

            max_memory_load = max(max_memory_load, torch.cuda.max_memory_allocated() / (1024 ** 3))  # 以GB为单位

            with open(save_dir + 'gradient_norms.txt', 'a') as file:
                file.write("Total gradient norm for epoch {}, iteration {}: {}\n".format(epoch, i, total_norm))
                print("Total gradient norm for epoch {}, iteration {}: {}".format(epoch, i, total_norm))
            if i % 10 == 0:
                # print('Epoch: {}, i: {}, loss: {}'.format(epoch, i, model.get_current_losses()))
                with open(save_dir + 'output_Epoch.txt', 'a') as f:
                    # 将标准输出重定向到文件
                    original_stdout = sys.stdout
                    sys.stdout = f
                    # 执行print语句
                    print('Epoch: {}, i: {}, loss: {}'.format(epoch, i, model.get_current_losses()))
                    # 恢复标准输出
                    sys.stdout = original_stdout

        # if epoch % 2 == 0:
        print('saving model')
        model.save_networks('iter_{}'.format(epoch))
    model.save_networks('latest')
    average_training_time = total_training_time / total_iterations
    print(f"Average training time per iteration: {average_training_time:.4f} sec")
    print(f"Maximum memory load: {max_memory_load:.4f} GB")

def predict(**kwargs):
    print('kwargs: {}'.format(kwargs))
    brain_config = BrainConfig()
    brain_config._parse(kwargs)
    brain_config.isTrain = False
    brain_dataset = BrainDataset(brain_config)
    brain_dataset = DataLoader(brain_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    save_dir = brain_config.output_save_path

    model = getattr(models, brain_config.model)(brain_config)
    model.setup(brain_config)
    model.eval()

    if brain_config.task == 'AtoB':
        task = '{}_to_{}'.format(brain_config.A, brain_config.B)
    else:
        task = '{}_to_{}'.format(brain_config.B, brain_config.A)

    output_path = os.path.join(save_dir, '{}_{}'.format(brain_config.model, task))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    total_inference_time = 0
    max_memory_load = 0

    mae_synthrad2023_values = []
    ssim_synthrad2023_values = []
    psnr_synthrad2023_values = []
    num = 0
    for i, data in enumerate(brain_dataset):
        start_time = time.time()  # 开始时间

        i = data['name'][0]
        model.set_input(data)
        file_name = i.split("_")[0]
        num += 1

        image_path = output_path + '/image/' + file_name
        npy_path = output_path + '/npy/' + file_name

        if not os.path.exists(image_path):
            os.makedirs(image_path)
        if not os.path.exists(npy_path):
            os.makedirs(npy_path)

        model.test()
        visuals = model.get_current_visuals()

        end_time = time.time()  # 结束时间
        inference_time = end_time - start_time
        total_inference_time += inference_time

        max_memory_load = max(max_memory_load, torch.cuda.max_memory_allocated() / (1024 ** 3))  # 以GB为单位

        real_A = visuals['real_A'].permute(0, 2, 3, 1)[0, :, :, 0].data.detach().cpu().numpy()
        real_B = visuals['real_B'].permute(0, 2, 3, 1)[0, :, :, 0].data.detach().cpu().numpy()
        fake_B = visuals['fake_B'].permute(0, 2, 3, 1)[0, :, :, 0].data.detach().cpu().numpy()

        # Normalize images to 0-255
        real_A = (real_A + 1) / 2.0 * 255.0
        real_B = (real_B + 1) / 2.0 * 255.0
        fake_B = (fake_B + 1) / 2.0 * 255.0

        # Save images as required
        np.save(npy_path + '/{}_real_A.npy'.format(i), real_A)
        np.save(npy_path + '/{}_real_B.npy'.format(i), real_B)
        np.save(npy_path + '/{}_fake_B.npy'.format(i), fake_B)

        Image.fromarray(real_A.astype(np.uint8)).convert('L').save(image_path + '/{}_real_A.png'.format(i))
        Image.fromarray(real_B.astype(np.uint8)).convert('L').save(image_path + '/{}_real_B.png'.format(i))
        Image.fromarray(fake_B.astype(np.uint8)).convert('L').save(image_path + '/{}_fake_B.png'.format(i))
        # 图片路径
        real_A_path = image_path + '/{}_real_A.png'.format(i)
        real_B_path = image_path + '/{}_real_B.png'.format(i)
        fake_B_path = image_path + '/{}_fake_B.png'.format(i)
        # 加载图片
        real_A_np = Calculation.load_image_as_numpy(real_A_path)  # <class 'numpy.ndarray'>   (256, 256)
        real_B_np = Calculation.load_image_as_numpy(real_B_path)  # <class 'numpy.ndarray'>   (256, 256)
        fake_B_np = Calculation.load_image_as_numpy(fake_B_path)  # <class 'numpy.ndarray'>   (256, 256)

        real_B_array = torch.from_numpy(real_B_np)  # 转换为 PyTorch 张量
        fake_B_array = torch.from_numpy(fake_B_np)  # 转换为 PyTorch 张量
        real_B_tensor = real_B_array.unsqueeze(0).unsqueeze(0)
        fake_B_tensor = fake_B_array.unsqueeze(0).unsqueeze(0)

        mae_SynthRAD2023_value = mae_SynthRAD2023(real_B_np, fake_B_np)
        mae_synthrad2023_values.append(mae_SynthRAD2023_value)
        mae_SynthRAD2023_value = f'{mae_SynthRAD2023_value:.6f}'
        print('***** mae_SynthRAD2023_value *****', mae_SynthRAD2023_value)

        ssim_SynthRAD2023_value = ssim_SynthRAD2023(real_B_np, fake_B_np)
        ssim_synthrad2023_values.append(ssim_SynthRAD2023_value)
        ssim_SynthRAD2023_value = f'{ssim_SynthRAD2023_value:.6f}'
        print('***** ssim_SynthRAD2023_value *****', ssim_SynthRAD2023_value)

        psnr_SynthRAD2023_value = psnr_SynthRAD2023(real_B_np, fake_B_np)
        psnr_synthrad2023_values.append(psnr_SynthRAD2023_value)
        psnr_SynthRAD2023_value = f'{psnr_SynthRAD2023_value:.6f}'
        print('***** psnr_SynthRAD2023_value *****', psnr_SynthRAD2023_value)

    def calculate_statistics(values):
        mean_value = np.mean(values)
        std_value = np.std(values)
        return mean_value, std_value

    average_inference_time = total_inference_time / num
    print(f"Average inference time per sample: {average_inference_time:.4f} sec")
    print(f"Maximum memory load: {max_memory_load:.4f} GB")

    average_mae_SynthRAD2023, std_mae_SynthRAD2023 = calculate_statistics(mae_synthrad2023_values)
    average_ssim_SynthRAD2023, std_ssim_SynthRAD2023 = calculate_statistics(ssim_synthrad2023_values)
    average_psnr_SynthRAD2023, std_psnr_SynthRAD2023 = calculate_statistics(psnr_synthrad2023_values)

    print(f"Average MAE_SynthRAD2023: {average_mae_SynthRAD2023:.4f}±{std_mae_SynthRAD2023:.4f}")
    print(f"Average SSIM_SynthRAD2023: {average_ssim_SynthRAD2023:.4f}±{std_ssim_SynthRAD2023:.4f}")
    print(f"Average PSNR_SynthRAD2023: {average_psnr_SynthRAD2023:.4f}±{std_psnr_SynthRAD2023:.4f}")

if __name__ == '__main__':
    import fire
    fire.Fire()


