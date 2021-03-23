# dataset.py
import torch
import numpy as np
import os
import torchvision.transforms as trn
from torch.utils.data import Dataset
from PIL import Image
from skimage.filters import gaussian as gblur
from data.augment import Rot
from collections import OrderedDict


class ImageDataset(Dataset):
    def __init__(self, root_dir, class_file, transforms=None):
        self.root_dir = root_dir
        self.ClassInfo = self._GetClassInfo(class_file)
        self.ImageInfo = self._GetImageInfo(root_dir)
        self.transforms = transforms

    def __len__(self):
        return len(self.ImageInfo["path"])

    def __getitem__(self, ix):
        img_path = self.ImageInfo["path"][ix]
        img = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.ImageInfo["label"][ix], dtype=torch.int64)

        if self.transforms:
            img = self.transforms(img)

        sample = {"image": img, "label": label}

        return sample

    def _GetImageInfo(self, Dir):
        imgInfo = {"path": [], "label": []}  # all paths and labels
        folders = os.listdir(Dir)
        for folder in folders:
            files = os.listdir(Dir + "/" + folder)
            for file in files:
                imgInfo["path"].append(Dir + "/" + folder + "/" + file)
                imgInfo["label"].append(self.ClassInfo[folder])

        # check #path and #label
        assert len(imgInfo["path"]) == len(imgInfo["label"]), "check #path and #label"

        # random shuffle
        imgInfo2 = {"path": [], "label": []}  # all paths and labels
        index = np.arange(len(imgInfo["path"]))
        np.random.shuffle(index)
        for ix in index:
            imgInfo2["path"].append(imgInfo["path"][ix])
            imgInfo2["label"].append(imgInfo["label"][ix])
        return imgInfo2

    def _GetClassInfo(self, file):
        ClassInfo = {}
        with open(file, 'r') as f:
            chars = f.read().split("\n")[:-1]
            for char in chars:
                if char != '':
                    c, L = char.split(" ")[0], int(char.split(" ")[1])
                    ClassInfo[c] = L
        return ClassInfo


class MultiAugmentDataset(Dataset):
    def __init__(self, root_dir, class_file, transforms=None, mode='train', size=32):
        self.root_dir = root_dir
        self.ClassInfo = self._GetClassInfo(class_file)
        self.ImageInfo = self._GetImageInfo(root_dir)
        self.transforms = transforms  # transform for multiple copy image
        self.mode = mode
        self.size = size

    def __len__(self):
        return len(self.ImageInfo["path"])

    def __getitem__(self, ix):
        img_path = self.ImageInfo["path"][ix]
        img = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.ImageInfo["label"][ix], dtype=torch.int64)
        sample = OrderedDict()

        # resize and centercrop (for img.size != 32 * 32)
        img_w, img_h = img.size
        if not (img_w == self.size and img_h == self.size):
            img = trn.Resize(self.size)(img)
            img = trn.CenterCrop(self.size)(img)

        # random crop (for training)
        if self.mode == 'train':
            img = trn.RandomCrop(self.size, padding=4)(img)

        # rotation
        if self.transforms:
            for t in self.transforms:
                img_clone = self.transforms[t](img.copy())
                img_clone = trn.ToTensor()(img_clone)
                img_clone = trn.Normalize(mean=[0.5] * 3, std=[0.5] * 3)(img_clone)
                sample[t] = {"image": img_clone, "label": label}

        return sample

    def _GetImageInfo(self, Dir):
        imgInfo = {"path": [], "label": []}  # all paths and labels
        folders = os.listdir(Dir)
        for folder in folders:
            files = os.listdir(Dir + "/" + folder)
            for file in files:
                imgInfo["path"].append(Dir + "/" + folder + "/" + file)
                imgInfo["label"].append(self.ClassInfo[folder])
        # check number
        assert len(imgInfo["path"]) == len(imgInfo["label"]), "# path != # label"

        # random shuffle path and label
        imgInfo2 = {"path": [], "label": []}  # all paths and labels
        index = np.arange(len(imgInfo["path"]))
        np.random.shuffle(index)
        for ix in index:
            imgInfo2["path"].append(imgInfo["path"][ix])
            imgInfo2["label"].append(imgInfo["label"][ix])
        return imgInfo2

    def _GetClassInfo(self, file):
        ClassInfo = {}
        with open(file, 'r') as f:
            chars = f.read().split("\n")[:-1]
            for char in chars:
                if char != '':
                    c, L = char.split(" ")[0], int(char.split(" ")[1])
                    ClassInfo[c] = L
        return ClassInfo


class SyntheticDataset(Dataset):
    def __init__(self, dataName, num_examples, transforms=None, size=32):
        self.dataName = dataName
        self.num_examples = num_examples
        self.transforms = transforms
        self.size = size
        self.data, self.label = self._DataGenerator(dataName, num_examples)
        assert self.data.shape[0] == self.label.shape[0], "# path != # label"

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, ix):
        label = torch.tensor(self.label[ix], dtype=torch.int64)
        img = Image.fromarray(self.data[ix], 'RGB')
        sample = {}

        if self.transforms:
            for t in self.transforms:
                img_clone = self.transforms[t](img.copy())
                img_clone = trn.ToTensor()(img_clone)
                img_clone = trn.Normalize(mean=[0.5] * 3, std=[0.5] * 3)(img_clone)
                sample[t] = {"image": img_clone, "label": label}

        return sample

    def _DataGenerator(self, dataName, num_examples):
        """
        create synthetic data using numpy

        """
        if dataName == "gaussian":
            # prepare gaussian data
            data = np.clip(np.random.normal(size=(num_examples, self.size, self.size, 3), scale=0.5), -1, 1).astype(np.float32)

        elif dataName == "rademacher":
            # prepare rademacher data
            data = np.random.binomial(n=1, p=0.5, size=(num_examples, self.size, self.size, 3)).astype(np.float32) * 2 - 1

        elif dataName == "blob":
            data = np.random.binomial(n=1, p=0.7, size=(num_examples, self.size, self.size, 3)).astype(np.float32)
            for i in range(num_examples):
                data[i] = gblur(data[i], sigma=1.5, multichannel=False)
                data[i][data[i] < 0.75] = 0.0

        label = np.random.randint(low=0, high=10, size=num_examples, dtype=np.int64)

        return data, label


if __name__ == "__main__":
    test_aug = {"rot_0": Rot(0), "rot_90": Rot(90), "rot_180": Rot(180), "rot_270": Rot(270)}
    dst = MultiAugmentDataset(
        root_dir="/usr/share/dataset/cifar10" + '/' + 'test',
        class_file="/usr/share/dataset/cifar10" + '/' + 'class.txt',
        mode='test',
        transforms=test_aug
    )
    sample = dst[100]
    print(sample["rot_0"]["image"].dtype, sample["rot_0"]["image"].size())
    print(sample["rot_0"]["label"])
