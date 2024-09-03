import os
import sys
sys.path.insert(0, r'/HOME/scw6d2x/run/jlw')
import cv2
import torch
import numpy as np
from PIL import Image
from .utils import load_path
from .base_dataset import get_transform, BaseDataset, normalize, standard, transform_resize, get_transform_face
from torchvision import transforms

class BrainDataset(BaseDataset):
    def __init__(self, conf):
        BaseDataset.__init__(self, conf)
        self.dir_A = os.path.join(conf.dataroot, conf.A)
        self.dir_B = os.path.join(conf.dataroot, conf.B)
        self.A_paths = load_path(self.dir_A)  # MR
        self.B_paths = load_path(self.dir_B)  # CT
        self.len_A = len(self.A_paths)
        self.len_B = len(self.B_paths)

        self.transform_A = get_transform(self.conf, grayscale=False)
        self.transform_B = get_transform(self.conf, grayscale=False)

        self.transform_A_face = get_transform_face(self.conf, grayscale=True)
        self.transform_B_face = get_transform_face(self.conf, grayscale=True)

        self.transform_resize = transform_resize(self.conf)

    def __len__(self):
        return max(self.len_A, self.len_B)

    def __getitem__(self, idx):

        if self.conf.B == 'CT':
            A_path = self.A_paths[(idx % self.len_A)]
            name = A_path.split('/')[-1]
            parts = name.split('_')
            for k in range(len(parts)):
                if parts[k] == 'mr':
                    parts[k] = 'ct'
                elif parts[k] == 'MR':
                    parts[k] = 'CT'
            name_ct = "_".join(parts)
            B_path = os.path.join(self.dir_B, name_ct)
            name = B_path.split('/')[-1][:-4]

        elif self.conf.B == 't2':
            A_path = self.A_paths[(idx % self.len_A)]
            print("%%% A_path %%%", A_path)
            name = A_path.split('/')[-1]
            parts = name.split('_')
            for k in range(len(parts)):
                if parts[k] == 't1':
                    parts[k] = 't2'
                elif parts[k] == 'T1':
                    parts[k] = 'T2'
            name_t2 = "_".join(parts)
            B_path = os.path.join(self.dir_B, name_t2)
            name = B_path.split('/')[-1][:-4]

        elif self.conf.B == 't1ce':
            A_path = self.A_paths[(idx % self.len_A)]
            name = A_path.split('/')[-1]
            parts = name.split('_')
            for k in range(len(parts)):
                if parts[k] == 'flair':
                    parts[k] = 't1ce'
                elif parts[k] == 'FLAIR':
                    parts[k] = 'T1CE'
            name_t2 = "_".join(parts)
            B_path = os.path.join(self.dir_B, name_t2)
            name = B_path.split('/')[-1][:-4]

        elif self.conf.B == 't1':
            if self.conf.A == 't2':
                A_path = self.A_paths[(idx % self.len_A)]
                print("%%% A_path %%%", A_path)
                name = A_path.split('/')[-1]
                parts = name.split('_')
                for k in range(len(parts)):
                    if parts[k] == 't2':
                        parts[k] = 't1'
                    elif parts[k] == 'T2':
                        parts[k] = 'T1'
                name_t2 = "_".join(parts)
                B_path = os.path.join(self.dir_B, name_t2)
                name = B_path.split('/')[-1][:-4]

            elif self.conf.A == 'flair':  # # flair ——> t1
                A_path = self.A_paths[(idx % self.len_A)]
                name = A_path.split('/')[-1]
                parts = name.split('_')
                for k in range(len(parts)):
                    if parts[k] == 'flair':
                        parts[k] = 't1'
                    elif parts[k] == 'FLAIR':
                        parts[k] = 'T1'
                name_t2 = "_".join(parts)
                B_path = os.path.join(self.dir_B, name_t2)
                name = B_path.split('/')[-1][:-4]

        elif self.conf.B == 'flair':
            if self.conf.A == 't1ce':
                A_path = self.A_paths[(idx % self.len_A)]
                print("%%% A_path %%%", A_path)
                name = A_path.split('/')[-1]
                # print("#### name_prior ####", name)
                parts = name.split('_')
                for k in range(len(parts)):
                    if parts[k] == 't1ce':
                        parts[k] = 'flair'
                    elif parts[k] == 'T1CE':
                        parts[k] = 'FLAIR'
                # print("----------------------------")
                name_t2 = "_".join(parts)
                B_path = os.path.join(self.dir_B, name_t2)
                name = B_path.split('/')[-1][:-4]

            elif self.conf.A == 't1':  # # t1 ——> flair
                A_path = self.A_paths[(idx % self.len_A)]
                print("%%% A_path %%%", A_path)
                name = A_path.split('/')[-1]
                # print("#### name_prior ####", name)
                parts = name.split('_')
                for k in range(len(parts)):
                    if parts[k] == 't1':
                        parts[k] = 'flair'
                    elif parts[k] == 'T1':
                        parts[k] = 'FLAIR'
                name_t2 = "_".join(parts)
                B_path = os.path.join(self.dir_B, name_t2)
                name = B_path.split('/')[-1][:-4]

        A_img = np.load(A_path)
        B_img = np.load(B_path)
        A_img = A_img / A_img.max() * 255
        B_img = B_img / B_img.max() * 255
        A_img = Image.fromarray(np.uint8(A_img)).convert('L')
        B_img = Image.fromarray(np.uint8(B_img)).convert('L')

        img_np_A = np.array(A_img)
        img_np_B = np.array(B_img)

        original_A = transforms.ToTensor()(img_np_A)
        original_B = transforms.ToTensor()(img_np_B)
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'original_A': original_A, 'original_B': original_B, 'A_paths': A_path, 'B_paths': B_path, 'name': name}


