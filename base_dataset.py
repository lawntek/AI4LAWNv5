

import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

import random
import importlib
import albumentations as A
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
from collections import namedtuple
import os
import numpy as np
import torch



label = namedtuple('label', ['name', 'color', 'trainId'])

labels = [label('Building', [255,106,77], 0), #orange
          label('Driveway', [61,61,245], 1), #blue
          label('Landscaping Bed', [204,153,51], 2), #brown
          label('Pool', [51,221,255], 3), #light blue
          label('Sidewalk', [250,250,55], 4), #yellow
          label('turf', [61,245,61], 5), #green
          label('background', [0,0,0], 6)] #black



class SegmentationDataset(Dataset):
    def __init__(self, root_dir, labelmap_path, transform=None, crop_size=256):
        self.transform = transform
        self.crop_size = crop_size
        self.rgb_to_class = self._parse_labelmap(labelmap_path)

        image_dir = os.path.join(root_dir, "images")
        mask_dir  = os.path.join(root_dir, "masks")

        self.image_paths = sorted([
            os.path.join(image_dir, f) 
            for f in os.listdir(image_dir) 
            if f.lower().endswith((".jpeg", ".jpg", ".png"))
        ])
        self.mask_paths = []
        for img_path in self.image_paths:
            fname = os.path.splitext(os.path.basename(img_path))[0] + ".png"
            mpath = os.path.join(mask_dir, fname)
            if os.path.exists(mpath):
                self.mask_paths.append(mpath)
            else:
                raise FileNotFoundError(f"Mask not found for image {img_path}: {mpath}")
        # self.image_paths=self.image_paths[:100]p
        # self.mask_paths=self.mask_paths[:100]p
        print(f"Total samples: {len(self.image_paths)}")

    def _parse_labelmap(self, filepath):
        rgb_to_class = {}
        with open(filepath, 'r') as f:
            idx = 0
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(':')
                if len(parts) < 2 or not parts[1].strip():
                    continue
                try:
                    rgb = tuple(map(int, parts[1].split(',')))
                    rgb_to_class[rgb] = idx
                    idx += 1
                except:
                    continue
        return rgb_to_class

    def _mask_rgb_to_class(self, mask):
        h, w, _ = mask.shape
        class_mask = np.zeros((h, w), dtype=np.int64)
        for rgb, idx in self.rgb_to_class.items():
            class_mask[np.all(mask == rgb, axis=-1)] = idx
        return class_mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask_rgb = np.array(Image.open(self.mask_paths[idx]).convert("RGB"))

        mask = self._mask_rgb_to_class(mask_rgb)

        image = cv2.resize(image, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask,  (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"]

        mask = np.eye(len(self.rgb_to_class), dtype=np.float32)[mask]

        return image, mask
