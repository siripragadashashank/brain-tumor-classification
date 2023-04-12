import torch
import numpy as np
from torch.utils.data import Dataset

from utils import load_dicom_images_3d


class MRIDataset(Dataset):
    def __init__(self, paths, targets=None, mri_type=None, num_imgs=64, img_size=128,
                 label_smoothing=0.01, split="train", augment=False):
        self.paths = paths
        self.targets = targets
        self.mri_type = mri_type
        self.num_imgs = num_imgs
        self.img_size = img_size
        self.label_smoothing = label_smoothing
        self.split = split
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        scan_id = self.paths[index]
        if self.targets is None:
            data = load_dicom_images_3d(str(scan_id).zfill(5),
                                        num_imgs=self.num_imgs,
                                        img_size=self.img_size,
                                        mri_type=self.mri_type[index],
                                        split=self.split)
        else:
            if self.augment:
                rotation = np.random.randint(0, 4)
            else:
                rotation = 0

            data = load_dicom_images_3d(str(scan_id).zfill(5),
                                        num_imgs=self.num_imgs,
                                        img_size=self.img_size,
                                        mri_type=self.mri_type[index],
                                        split="train",
                                        rotate=rotation)

        if self.targets is None:
            return {"X": torch.tensor(data).float(), "id": scan_id}
        else:
            y = torch.tensor(abs(self.targets[index] - self.label_smoothing), dtype=torch.float)
            return {"X": torch.tensor(data).float(), "y": y}



