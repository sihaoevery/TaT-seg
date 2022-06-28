from __future__ import absolute_import, print_function

import os.path as osp
from glob import glob

# import cv2
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data
import random
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr
from mypath import Path

class _BaseDataset(data.Dataset):
    """
    Base dataset class
    """

    def __init__(
        self,
        root,
        split,
        ignore_label,
        mean_bgr,
        augment=True,
        base_size=None,
        crop_size=321,
        scales=(1.0),
        flip=True,
    ):
        self.root = root
        self.split = split
        self.ignore_label = ignore_label
        self.mean_bgr = np.array(mean_bgr)
        self.augment = augment
        self.base_size = base_size
        self.crop_size = crop_size
        self.scales = scales
        self.flip = flip
        self.files = []
        self._set_files()

        # cv2.setNumThreads(0)

    def _set_files(self):
        """
        Create a file path/image id list.
        """
        raise NotImplementedError()

    def _load_data(self, image_id):
        """
        Load the image and label in numpy.ndarray
        """
        raise NotImplementedError()

    def _augmentation(self, image, label):
        # Scaling
        h, w = label.shape
        if self.base_size:
            if h > w:
                h, w = (self.base_size, int(self.base_size * w / h))
            else:
                h, w = (int(self.base_size * h / w), self.base_size)
        scale_factor = random.choice(self.scales)
        h, w = (int(h * scale_factor), int(w * scale_factor))
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        label = Image.fromarray(label).resize((w, h), resample=Image.NEAREST)
        label = np.asarray(label, dtype=np.int64)

        # Padding to fit for crop_size
        h, w = label.shape
        pad_h = max(self.crop_size - h, 0)
        pad_w = max(self.crop_size - w, 0)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT,
        }
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, value=self.mean_bgr, **pad_kwargs)
            label = cv2.copyMakeBorder(label, value=self.ignore_label, **pad_kwargs)

        # Cropping
        h, w = label.shape
        start_h = random.randint(0, h - self.crop_size)
        start_w = random.randint(0, w - self.crop_size)
        end_h = start_h + self.crop_size
        end_w = start_w + self.crop_size
        image = image[start_h:end_h, start_w:end_w]
        label = label[start_h:end_h, start_w:end_w]

        if self.flip:
            # Random flipping
            if random.random() < 0.5:
                image = np.fliplr(image).copy()  # HWC
                label = np.fliplr(label).copy()  # HW
        return image, label

    # def __getitem__(self, index):
    #     image_id, image, label = self._load_data(index)
    #     if self.augment:
    #         image, label = self._augmentation(image, label)
    #     # Mean subtraction
    #     image -= self.mean_bgr
    #     # HWC -> CHW
    #     image = image.transpose(2, 0, 1)
    #     return image_id, image.astype(np.float32), label.astype(np.int64)

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str


class CocoStuff10k(_BaseDataset):
    """COCO-Stuff 10k dataset"""
    NUM_CLASSES=182

    def __init__(self, 
                warp_image=True, 
                **kwargs):
        self.warp_image = warp_image
        super(CocoStuff10k, self).__init__(**kwargs)
        
    def _set_files(self):
        # Create data list via {train, test, all}.txt
        if self.split in ["train", "test", "all"]:
            file_list = osp.join(self.root, "imageLists", self.split + ".txt")
            file_list = tuple(open(file_list, "r"))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files = file_list
        else:
            raise ValueError("Invalid split name: {}".format(self.split))

    def _load_data(self, index):
        # Set paths
        image_id = self.files[index]
        image_path = osp.join(self.root, "images", image_id + ".jpg")
        label_path = osp.join(self.root, "annotations", image_id + ".mat")
        # Load an image and label
        # image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        image = Image.open(image_path).convert('RGB') 
        label = sio.loadmat(label_path)["S"]
        label -= 1  # unlabeled (0 -> -1)
        label[label == -1] = 255
        label = Image.fromarray(label,mode='L') # align the custom_transform.py, need Image object
        # Warping: this is just for reproducing the official scores on GitHub
        # if self.warp_image:
        #     image = cv2.resize(image, (513, 513), interpolation=cv2.INTER_LINEAR)
        #     label = Image.fromarray(label).resize((513, 513), resample=Image.NEAREST)
        #     label = np.asarray(label)
        return image_id, image, label

    def __getitem__(self, index):
        image_id, image, label = self._load_data(index)
        sample = {'image': image, 'label': label}

        # for split in self.split:
        #     if split == "train":
        #         return self.transform_tr(sample)
        #     elif split == 'test':
        #         return self.transform_val(sample)

        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == 'test':
            return self.transform_val(sample)

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.base_size, crop_size=self.crop_size,fill=255),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

class CocoStuff164k(_BaseDataset):
    """COCO-Stuff 164k dataset"""

    def __init__(self, **kwargs):
        super(CocoStuff164k, self).__init__(**kwargs)

    def _set_files(self):
        # Create data list by parsing the "images" folder
        if self.split in ["train2017", "val2017"]:
            file_list = sorted(glob(osp.join(self.root, "images", self.split, "*.jpg")))
            assert len(file_list) > 0, "{} has no image".format(
                osp.join(self.root, "images", self.split)
            )
            file_list = [f.split("/")[-1].replace(".jpg", "") for f in file_list]
            self.files = file_list
        else:
            raise ValueError("Invalid split name: {}".format(self.split))

    def _load_data(self, index):
        # Set paths
        image_id = self.files[index]
        image_path = osp.join(self.root, "images", self.split, image_id + ".jpg")
        label_path = osp.join(self.root, "annotations", self.split, image_id + ".png")
        # Load an image and label
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        return image_id, image, label


def get_parent_class(value, dictionary):
    # Get parent class with COCO-Stuff hierarchy
    for k, v in dictionary.items():
        if isinstance(v, list):
            if value in v:
                yield k
        elif isinstance(v, dict):
            if value in list(v.keys()):
                yield k
            else:
                for res in get_parent_class(value, v):
                    yield res


# if __name__ == "__main__":
#     import matplotlib
#     import matplotlib.pyplot as plt
#     import matplotlib.cm as cm
#     import torchvision
#     import yaml
#     from torchvision.utils import make_grid
#     from tqdm import tqdm

#     kwargs = {"nrow": 10, "padding": 50}
#     batch_size = 100

#     dataset = CocoStuff164k(
#         root="/media/kazuto1011/Extra/cocostuff/cocostuff-164k",
#         split="train2017",
#         ignore_label=255,
#         mean_bgr=(104.008, 116.669, 122.675),
#         augment=True,
#         crop_size=321,
#         scales=(0.5, 0.75, 1.0, 1.25, 1.5),
#         flip=True,
#     )
#     print(dataset)

#     loader = data.DataLoader(dataset, batch_size=batch_size)

#     for i, (image_ids, images, labels) in tqdm(
#         enumerate(loader), total=np.ceil(len(dataset) / batch_size), leave=False
#     ):
#         if i == 0:
#             mean = torch.tensor((104.008, 116.669, 122.675))[None, :, None, None]
#             images += mean.expand_as(images)
#             image = make_grid(images, pad_value=-1, **kwargs).numpy()
#             image = np.transpose(image, (1, 2, 0))
#             mask = np.zeros(image.shape[:2])
#             mask[(image != -1)[..., 0]] = 255
#             image = np.dstack((image, mask)).astype(np.uint8)

#             labels = labels[:, np.newaxis, ...]
#             label = make_grid(labels, pad_value=255, **kwargs).numpy()
#             label_ = np.transpose(label, (1, 2, 0))[..., 0].astype(np.float32)
#             label = cm.jet_r(label_ / 182.0) * 255
#             mask = np.zeros(label.shape[:2])
#             label[..., 3][(label_ == 255)] = 0
#             label = label.astype(np.uint8)

#             tiled_images = np.hstack((image, label))
#             # cv2.imwrite("./docs/datasets/cocostuff.png", tiled_images)
#             plt.imshow(np.dstack((tiled_images[..., 2::-1], tiled_images[..., 3])))
#             plt.show()
#             break

#     class_hierarchy = "./data/datasets/cocostuff/cocostuff_hierarchy.yaml"
#     data = yaml.load(open(class_hierarchy))
#     key = "person"

#     for _ in range(3):
#         key = get_parent_class(key, data)
#         key = list(key)[0]
#         print(key)
