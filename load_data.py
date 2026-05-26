from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import scipy.io as scio

from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop
from utils.image_path import *


class BaseDataset(Dataset):
    def __init__(self,
                 indexs: dict,
                 labels: dict,
                 img_path: str,
                 is_train=True,
                 imageResolution=224
                 ):

        self.indexs = indexs
        self.labels = labels
        self.img_path = img_path
        self.is_train = is_train
        self.transform = Compose([
            Resize((256,256), interpolation=Image.BICUBIC),
            RandomHorizontalFlip(),
            RandomCrop(imageResolution),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]) if is_train else Compose([
            Resize((256,256), interpolation=Image.BICUBIC),
            CenterCrop(imageResolution),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.__length = len(self.indexs)

    def __len__(self):
        return self.__length

    def _load_image(self, index: int) -> torch.Tensor:
        image_path = os.path.join(self.img_path, self.indexs[index].strip())
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image

    def _load_label(self, index: int) -> torch.Tensor:
        label = self.labels[index]
        label = torch.from_numpy(label)
        return label

    def get_all_label(self):
        labels = torch.zeros([self.__length, len(self.labels[0])], dtype=torch.int64)
        for i, item in enumerate(self.labels):
            labels[i] = torch.from_numpy(item)
        return labels

    def __getitem__(self, index):
        image = self._load_image(index)
        label = self._load_label(index)
        return image, label, index

def generate_dataset(
        dataset_name: str,
        imageResolution=224,
):
    DB_indexs = scio.loadmat(f"dataset/{dataset_name}/index_DB.mat")["index"]
    DB_labels = scio.loadmat(f"dataset/{dataset_name}/label_DB.mat")["label"]

    Query_indexs = scio.loadmat(f"dataset/{dataset_name}/index_Query.mat")["index"]
    Query_labels = scio.loadmat(f"dataset/{dataset_name}/label_Query.mat")["label"]

    Train_indexs = scio.loadmat(f"dataset/{dataset_name}/index_Train.mat")["index"]
    Train_labels = scio.loadmat(f"dataset/{dataset_name}/label_Train.mat")["label"]

    if dataset_name == 'imagenet':
        img_path = imagenet_img_path
    elif dataset_name == 'nuswide':
        img_path = nuswide_img_path
    else:
        img_path = coco_img_path

    query_data = BaseDataset(indexs=Query_indexs, labels=Query_labels,
                             img_path=img_path, imageResolution=imageResolution,
                              is_train=False)
    train_data = BaseDataset(indexs=Train_indexs, labels=Train_labels,
                             img_path=img_path, imageResolution=imageResolution)
    retrieval_data = BaseDataset(indexs=DB_indexs, labels=DB_labels,
                                 img_path=img_path, imageResolution=imageResolution,
                                  is_train=False)

    return train_data, query_data, retrieval_data
