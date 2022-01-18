import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
# TODO: specify the return type
def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
class dataset_retina(Dataset):
    def __init__(self, directory, transform=None,sex=True,loader= default_loader):
        super().__init__()
        sex_dict, age_dict = {}, {}
        with open(f'{directory}/label.txt') as f:
            for i, line in enumerate(f):
                idx, sex, age = line.strip().split()
                if int(idx) not in sex_dict.keys():
                    sex_dict[int(idx)] = int(sex) % 2
                    continue

                if int(idx) not in age_dict.keys():
                    sex_dict[int(idx)] = int(age)
                    continue
        listfiles = []
        files = os.listdir(f'{directory}/images')
        for i, item in enumerate(files):
            listfiles.append('{}/images/{}'.format(directory, item))
        self.directory = directory
        self.listfiles = listfiles
        self.search_dict = sex_dict if sex else age_dict
        self.transform = transform
        self.loader = loader
    def __len__(self):
        return len(self.listfiles)

    def __getitem__(self, idx):
        path = self.listfiles[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        target = self.search_dict[idx]
        return sample,target
import random
class dataset_retina30k(Dataset):
    def __init__(self, directory, transform=None, loader= default_loader,train=True,train_ratio=0.8):
        super().__init__()

        listfiles = []
        labels = []
        files = os.listdir(directory)
        files.sort()
        for i, item in enumerate(files):
            cur_dir = os.path.join(directory,item)
            allfiles = os.listdir(cur_dir)
            random.seed(888)
            random.shuffle(allfiles)
            if train:
                select_files = allfiles[:int(len(allfiles)*train_ratio)]
            else:
                select_files = allfiles[int(len(allfiles)*train_ratio):]
            for item2 in select_files:
                listfiles.append(os.path.join(cur_dir,item2))
                labels.append(i)

        self.directory = directory
        self.listfiles = listfiles
        self.labels = labels
        self.transform = transform
        self.loader = loader
    def __len__(self):
        return len(self.listfiles)

    def __getitem__(self, idx):
        path = self.listfiles[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        target = self.labels[idx]
        return sample,target
