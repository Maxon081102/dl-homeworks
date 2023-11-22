import os
import cv2
import torch
import xmltodict
import pytorch_lightning as L
import numpy as np

import torch.nn.functional as F

from typing import Callable

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from PIL import Image
    
def get_img_names(base_dir):
    imgs = []
    for dir in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, dir)):
            for img in os.listdir(os.path.join(base_dir, dir)):
                if img[-4:] == '.jpg':
                    imgs.append(os.path.join(dir, img))
    return imgs

image_transform = transforms.ToTensor()

def collate_fn(batch):
    img_batch = torch.stack([elem[0] for elem in batch])
    annotation_batch = [elem[1] for elem in batch]
    return img_batch, annotation_batch

def prepare_annotation(annotation):
    if "object" not in annotation:
        return torch.zeros(0,4), torch.zeros(0).type(torch.long)
    objects = []
    if isinstance(annotation["object"], dict):
        objects = torch.Tensor(list(map(int, annotation["object"]["bndbox"].values())))[None]
        return objects, torch.ones(objects.shape[0]).type(torch.long)
    else:
        for item in annotation["object"]:
            objects.append(torch.Tensor(list(map(int, item["bndbox"].values()))))
    objects = torch.stack(objects)
    return objects, torch.ones(objects.shape[0]).type(torch.long)


class ImgData(Dataset):
    def __init__(self, imgs, annotations, transform=None):
        self.img_source = os.path.join('PeopleArt-master', 'JPEGImages')
        super(ImgData).__init__()
        self.imgs = imgs
        self.annotations = annotations
        self.transform = transform
    
    def __getitem__(self, index):
        img = image_transform(Image.open(os.path.join(self.img_source, self.imgs[index])))
        img = F.pad(img, (0, 500 - img.shape[2], 0, 500 - img.shape[1], 0, 0))
        if self.transform is not None:
            img = self.transform(img)
        
        annotation = prepare_annotation(self.annotations[index])
        return img, {"boxes": annotation[0], "labels": annotation[1]}
    
    def __len__(self):
        return len(self.imgs)

class ImgDatamodule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        transform: Callable[[Image.Image], torch.Tensor] = transforms.ToTensor(),
        num_workers: int = 8,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transform
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        np.random.seed(777)
        img_source = os.path.join('PeopleArt-master', 'JPEGImages')
        annotation_source = os.path.join('PeopleArt-master', 'Annotations')
        img_names = np.array(get_img_names(img_source))
        self.img_names = []
        self.annotation_data = []
        for img_name in img_names:
            annotation_location = os.path.join(annotation_source, img_name + '.xml')
            if not os.path.exists(annotation_location):
                continue
            with open(annotation_location) as xml:
                self.annotation_data.append(xmltodict.parse(xml.read())["annotation"])
                self.img_names.append(img_name)
                
                # # TODO: remove
                # if len(self.annotation_data) == 100:
                #     break
                
        self.annotation_data = np.array(self.annotation_data)
        self.img_names = np.array(self.img_names)
        
        idxs = list(np.arange(len(self.img_names)))
        np.random.shuffle(idxs)
        self.train_idx = idxs[:int(0.8 * len(idxs))]
        self.val_idx = idxs[int(0.8 * len(idxs)):]
        
        # # TODO: remove
        # self.train_idx = self.train_idx[:100]
        # self.val_idx = self.val_idx[:10]

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = ImgData(self.img_names[self.train_idx], self.annotation_data[self.train_idx], self.transform)
            self.val_dataset = ImgData(self.img_names[self.val_idx], self.annotation_data[self.val_idx], self.transform)
        elif stage == "validate":
            self.val_dataset = ImgData(self.img_names[self.val_idx], self.annotation_data[self.val_idx], self.transform)
        else:
            raise NotImplementedError

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=self.num_workers,
        )
