import os
import pickle
from functools import partial
from math import ceil, floor

import numpy as np
import torch
import torchvision.transforms.functional as tf
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageSR(Dataset):
    def __init__(
        self,
        size: int | None = None,
        downscale_f=1,
        min_crop_f=0.5,
        max_crop_f=1.0,
        gray=False,
        random_crop=True,
        class_labels=False,
        validation=False,
    ):
        """
        Super-resolution Dataloader
        Performs following ops in order:
        1.  crops a crop of size s from image either as random or center crop
        2.  resizes crop to size with cv2.area_interpolation
        3.  degrades resized crop

        :param size: resizing to size after cropping
        :param downscale_f: Low Resolution Downsample factor
        :param min_crop_f: determines crop size s,
          where s = c * min_img_side_len with c sampled from interval (min_crop_f, max_crop_f)
        :param max_crop_f: ""
        :param data_root:
        :param random_crop:
        """
        assert size
        assert (size / downscale_f).is_integer()
        self.size = size
        self.LR_size = int(size / downscale_f)
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        assert max_crop_f <= 1.0
        self.center_crop = not random_crop
        self.pil_interpolation = downscale_f != 1  # gets reset later if incase interp_op is from pillow
        self.degradation_process = partial(tf.resize, size=self.LR_size, interpolation=tf.InterpolationMode.NEAREST)  # type: ignore
        # dataset_path = "/media/data/dataset-nako/images/" + ("train" if not validation else "val")
        dataset_path = "/media/data/robert/datasets/nako_jpg/" + ("train" if not validation else "val")

        buffer_path = os.path.join(dataset_path, ".buffer.pkl")

        # Check if the buffer file exists
        if os.path.exists(buffer_path):
            # Load the buffer file
            with open(buffer_path, "rb") as f:
                file_list = pickle.load(f)
            print("Loaded file list from buffer.")
        else:
            # Traverse the directory and collect all file paths
            file_list = []
            for root, _, files in os.walk(dataset_path):
                for file in files:
                    if ".png" in file or ".jpg" in file:
                        file_list.append(os.path.join(root, file))

            # Save the list to a pickle file
            with open(buffer_path, "wb") as f:
                pickle.dump(file_list, f)
            print("Buffer file created and file list saved:", buffer_path)
        self.file_list = file_list

        # Pad to the larger dimension and then resize
        self.image_rescale = transforms.Resize(self.size)
        self.cropper = transforms.CenterCrop(self.size) if self.center_crop else transforms.RandomCrop(self.size)
        self.class_labels = class_labels
        self.keys = {"HWS": 0, "BWS": 1, "LWS": 2}
        self.gray = gray

    def padd(self, target: torch.Tensor, edge=10, padding=None):
        if padding is None:
            w, h = target.shape[-2], target.shape[-1]
            hp = max((self.size + edge - w) / 2, 0)
            vp = max((self.size + edge - h) / 2, 0)
            padding = [int(floor(vp)), int(floor(hp)), int(ceil(vp)), int(ceil(hp))]
        return tf.pad(target, padding, padding_mode="constant"), padding

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, i):
        path: str = self.file_list[i]
        image_pil = Image.open(path)  # type: ignore

        if image_pil.mode != "RGB":
            if not self.gray:
                image_pil = image_pil.convert("RGB")
        else:
            assert not self.gray
        image = tf.pil_to_tensor(image_pil).to(torch.uint8)

        # Add padding
        image, _ = self.padd(image)
        image = self.cropper(image)
        image: torch.Tensor = self.image_rescale(image)

        if self.pil_interpolation:
            lr_image = image.clone()
            lr_image = self.degradation_process(lr_image)  # type: ignore
            lr_image = tf.pil_to_tensor(lr_image).to(torch.uint8)
        else:
            lr_image = image

        example = {}
        example["image"] = (image / 127.5 - 1.0).to(torch.float32).permute((1, 2, 0))
        example["LR_image"] = (lr_image / 127.5 - 1.0).to(torch.float32).permute((1, 2, 0))  # type: ignore
        if self.class_labels:
            for key, value in self.keys.items():
                if key in path[-15:]:
                    example["class_label"] = value
                    example["human_label"] = key
                    break
            else:
                print("NoCLASS", path)
        return example


class ImageSRTrain(ImageSR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ImageSRValidation(ImageSR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, validation=True)


if __name__ == "__main__":
    c = ImageSR(256, gray=True)
    print(c[0]["image"].shape)
