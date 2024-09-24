import os
import pickle
import random
from math import ceil, floor

import numpy as np
import torch
import torchvision.transforms.functional as tf
from torch.utils.data import Dataset
from torchvision import transforms


class ImageSR(Dataset):
    def __init__(
        self,
        size: int | tuple[int, int] | None = None,
        # downscale_f=1,
        min_crop_f=0.5,
        max_crop_f=1.0,
        gray=False,
        random_crop=True,
        padding="constant",  # constant, edge, reflect or symmetric
        validation=False,
        norm=False,
        gauss=False,
        image_dropout=0.1,
        vflip=True,
        hflip=True,
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
        # assert (size / downscale_f).is_integer()
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        # self.LR_size = int(size / downscale_f)
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        assert max_crop_f <= 1.0
        self.center_crop = not random_crop
        # self.pil_interpolation = downscale_f != 1  # gets reset later if incase interp_op is from pillow
        # self.degradation_process = partial(tf.resize, size=self.LR_size, interpolation=tf.InterpolationMode.NEAREST)  # type: ignore
        # dataset_path = "/media/data/dataset-nako/images/" + ("train" if not validation else "val")
        dataset_path = "/media/data/robert/datasets/t2w_ct/" + ("train" if not validation else "val")

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
                    if ".npz" in file or ".npy" in file:
                        file_list.append(os.path.join(root, file))

            # Save the list to a pickle file
            with open(buffer_path, "wb") as f:
                pickle.dump(file_list, f)
            print("Buffer file created and file list saved:", buffer_path)
        self.file_list = file_list

        # Pad to the larger dimension and then resize
        # self.class_labels = class_labels
        self.gray = gray
        self.mri_transform = transforms.Compose([transforms.ColorJitter(brightness=0.2, contrast=0.2)])
        self.norm = norm
        self.gauss = gauss
        self.image_dropout = image_dropout
        self.vflip = vflip
        self.hflip = hflip
        self.train = not validation
        self.padding = padding
        self.condition_types = ["T2", "CT"]

    def load_file(self, name):
        dict_mods = {}
        if name.endswith(".npz"):
            f = np.load(name)
            for k, v in f.items():  # type: ignore
                dict_mods[k] = v.astype("f")
                if self.norm:
                    dict_mods[k] /= max(float(np.max(dict_mods[k])), 0.0000001)
            f.close()  # type: ignore
            return dict_mods
        raise AssertionError("Expected a .npz file")

    def gauss_filter(self, img_data) -> torch.Tensor | np.ndarray:
        if self.gauss:
            to_tensor = False
            if isinstance(img_data, torch.Tensor):
                img_data = img_data.detach().cpu().numpy()
                to_tensor = True
            from scipy import ndimage

            out: np.ndarray = ndimage.median_filter(img_data, size=3)  # type: ignore
            if to_tensor:
                return torch.from_numpy(out)
            return out
        return img_data

    @torch.no_grad()
    def transform(self, dict_mods):
        condition_types = self.condition_types
        if len(condition_types) == 1:
            key = condition_types[0]
            ct = dict_mods[key]  # self.gauss_filter()
            target = tf.to_tensor(ct)
        else:
            key = condition_types[0]
            condition_types = condition_types[1:]
            target = dict_mods[key]
            target = tf.to_tensor(target)
        if key not in ("CT", "SG") and self.mri_transform is not None:
            target = self.mri_transform(torch.cat([target, target, target], dim=0))[1:2]  # type: ignore

        second_img_list: list[torch.Tensor] = []
        for key in condition_types:
            img = tf.to_tensor(dict_mods[key])
            if key == "CT":
                img = self.gauss_filter(img)
            elif key == "SG":
                pass
            elif self.mri_transform is not None:
                img = self.mri_transform(torch.cat([img, img, img], dim=0))[1:2]  # type: ignore
            second_img_list.append(img)  # type: ignore

        if self.image_dropout > 0 and self.train and self.image_dropout > random.random():
            second_img_list[0] = second_img_list[0] * 0
        for _, i in enumerate(second_img_list):
            assert second_img_list[0].shape == i.shape, f"Shape mismatch {second_img_list[0].shape} {i.shape} "

        second_img = torch.cat(second_img_list, dim=0)
        # Padding
        w, h = target.shape[-2], target.shape[-1]
        hp = max((self.size[0] - w) / 2, 0)
        vp = max((self.size[1] - h) / 2, 0)
        padding = [int(floor(vp)), int(floor(hp)), int(ceil(vp)), int(ceil(hp))]

        target = tf.pad(target, padding, padding_mode=self.padding)
        second_img = tf.pad(second_img, padding, padding_mode=self.padding)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(target, output_size=self.size)  # type: ignore
        target = tf.crop(target, i, j, h, w)
        second_img = tf.crop(second_img, i, j, h, w)

        # Random horizontal flipping
        if self.hflip and random.random() > 0.5:
            target = tf.hflip(target)
            second_img = tf.hflip(second_img)

        # Random vertical flipping
        if self.vflip and random.random() > 0.5:
            target = tf.vflip(target)
            second_img = tf.vflip(second_img)

        # Normalize to -1, 1
        target = target * 2 - 1
        second_img = second_img * 2 - 1

        return target.swapaxes(-1, 0), second_img.swapaxes(-1, 0)

    def __len__(self):
        return len(self.file_list) * 2

    def __getitem__(self, i):
        flip = len(self.file_list) > i

        dict_mods = self.load_file(self.file_list[i % len(self.file_list)])
        target, condition = self.transform(dict_mods)
        if flip:
            a = condition
            condition = target
            target = a
        example = {"image": target, "LR_image": target, "c_concat": condition, "n_crossattn": condition}
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
    print(c[0].keys())
