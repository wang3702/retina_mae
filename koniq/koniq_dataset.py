import os
import torch
import numpy as np
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

class IQADataset(torch.utils.data.Dataset):
    def __init__(self, db_path, txt_file_name, transform, train_mode,  train_size=0.8,loader= default_loader):
        super(IQADataset, self).__init__()

        self.db_path = db_path
        self.txt_file_name = txt_file_name
        self.transform = transform
        self.train_mode = train_mode
        self.train_size = train_size

        self.data_dict = IQADatalist(
            txt_file_name = self.txt_file_name,
            train_mode = self.train_mode,
            train_size = self.train_size
        ).load_data_dict()

        self.n_images = len(self.data_dict['d_img_list'])
        self.loader = loader

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # d_img_org: H x W x C
        d_img_name = self.data_dict['d_img_list'][idx]
        path = os.path.join(self.db_path, d_img_name)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        score = self.data_dict['score_list'][idx]
        return sample,score
import random

class IQADatalist():
        def __init__(self, txt_file_name, train_mode,train_size=0.8):
            self.txt_file_name = txt_file_name
            self.train_mode = train_mode
            self.train_size = train_size


        def load_data_dict(self):
            scn_idx_list, d_img_list, score_list = [], [], []



            # list append
            with open(self.txt_file_name, 'r') as listFile:
                for line in listFile:
                    scn_idx, dis, score = line.split()
                    scn_idx = int(scn_idx)
                    score = float(score)

                    scn_idx_list.append(scn_idx)
                    d_img_list.append(dis)
                    score_list.append(score)
            select_index = np.arange(len(d_img_list))
            random.seed(42)
            random.shuffle(select_index)
            if self.train_mode==1:
                select_index = select_index[:int(self.train_size*len(select_index))]
            else:
                select_index = select_index[int(self.train_size*len(select_index)):]
            final_img_list = []
            final_score_list = []
            for k in range(len(d_img_list)):
                if k not in select_index:
                    continue
                final_img_list.append(d_img_list[k])
                final_score_list.append(score_list[k])
            # reshape score_list (1xn -> nx1)
            final_score_list = np.array(final_score_list)
            final_score_list = final_score_list.astype('float').reshape(-1, 1)
            print("remained %d examples"%len(final_score_list))
            data_dict = {'d_img_list': final_img_list, 'score_list':final_score_list}

            return data_dict
