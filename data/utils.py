import torch.utils.data as data
import glob
from PIL import Image
import os

IMG_EXTENSIONS = [
    '.jpg', '.png', '.jpeg', '.tiff'
]

def load_path(root):
    paths = []
    for file_type in IMG_EXTENSIONS:
        path = glob.glob(os.path.join(root, '*{}'.format(file_type)))
        if path:
            paths += path
    if len(paths) == 0:
        path = glob.glob(os.path.join(root, '*.npy'))
        if path:
            paths += path
    return paths
