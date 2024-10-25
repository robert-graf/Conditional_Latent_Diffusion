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
        gauss=0,
        image_dropout=0.1,
        vflip=True,
        hflip=True,
        dflip=False,
        rotation=None,
        noise_factor=0,
        noise=0,
        blur=0,
        random_zoom=False,
        zoom_min=0.8,
        zoom_max=1.2,
        use_multi_class_class_label=False,
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
        self.use_multi_class_class_label = use_multi_class_class_label
        if use_multi_class_class_label:
            labels_pkl = os.path.join("/media/data/robert/datasets/multimodal_large/", "labels.pkl")
            with open(labels_pkl, "rb") as f:
                self.labels_name: list = pickle.load(f)
            self.class_idx = [self.labels_name.index("T2w"), self.labels_name.index("ct")]
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
        self.dflip = dflip
        self.train = not validation
        self.padding = padding
        self.rotation = rotation
        self.blur = blur
        self.noise = noise
        self.noise_factor = noise_factor
        self.random_zoom = random_zoom
        self.zoom_min = zoom_min
        self.zoom_max = zoom_max
        self.condition_types = ["T2", "CT"]

    def load_file(self, i):
        name = self.file_list[i % len(self.file_list)]
        dict_mods = {}
        if name.endswith(".npz"):
            f = np.load(name)
            for k, v in f.items():  # type: ignore
                dict_mods[k] = v.astype("f")
                if dict_mods[k].sum() <= 5:
                    self.file_list.remove(i)
                    print("Remove empty image", i, name)
                    return self.load_file(i + 123)
                if self.norm:
                    dict_mods[k] /= max(float(np.max(dict_mods[k])), 0.0000001)
                assert float(np.max(dict_mods[k])) <= 1, (np.min(dict_mods[k]), np.max(dict_mods[k]))
                assert float(np.min(dict_mods[k])) >= 0, (np.min(dict_mods[k]), np.max(dict_mods[k]))
                # TODO round to 255?
            f.close()  # type: ignore
            return dict_mods
        raise AssertionError("Expected a .npz file")

    def gauss_filter(self, img_data) -> torch.Tensor | np.ndarray:
        if self.gauss > random.random():
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
    def transform(self, dict_mods, flip):
        condition_types = self.condition_types
        if flip:
            condition_types = [condition_types[-1], *condition_types[:-1]]

        if len(condition_types) == 1:
            key = condition_types[0]
            ct = dict_mods[key]  # self.gauss_filter()
            target = tf.to_tensor(ct)
        else:
            key = condition_types[0]
            condition_types = condition_types[1:]
            target = dict_mods[key]
            target = tf.to_tensor(target)
        # if key not in ("CT", "SG") and self.mri_transform is not None:
        #    target = self.mri_transform(torch.cat([target, target, target], dim=0))[1:2]  # type: ignore

        cond_img_list: list[torch.Tensor] = []
        for key in condition_types:
            img = tf.to_tensor(dict_mods[key])
            if key == "CT" or self.gauss != 0:
                img = self.gauss_filter(img)
            elif key == "SG":
                pass
            elif self.mri_transform is not None:
                img = self.mri_transform(torch.cat([img, img, img], dim=0))[1:2]  # type: ignore
            cond_img_list.append(img)  # type: ignore
        if self.image_dropout > 0 and self.train and self.image_dropout > random.random():
            cond_img_list[0] = cond_img_list[0] * 0
        for _, i in enumerate(cond_img_list):
            assert cond_img_list[0].shape == i.shape, f"Shape mismatch {cond_img_list[0].shape} {i.shape} "

        second_img = torch.cat(cond_img_list, dim=0)
        # Random zoom (RandomResizedCrop)
        if self.random_zoom:
            scale = (self.zoom_min, self.zoom_max)
            target = transforms.RandomResizedCrop(self.size, scale=scale)(target)
            second_img = transforms.RandomResizedCrop(self.size, scale=scale)(second_img)
        # Padding
        w, h = target.shape[-2], target.shape[-1]
        hp = max((self.size[0] - w) / 2, 0)
        vp = max((self.size[1] - h) / 2, 0)
        padding = [int(floor(vp)), int(floor(hp)), int(ceil(vp)), int(ceil(hp))]
        if self.rotation:
            angle = random.uniform(-self.rotation, self.rotation)  # Random rotation within range
            target = tf.rotate(target, angle, tf.InterpolationMode.BILINEAR)
            second_img = tf.rotate(second_img, angle)
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
        # Random vertical flipping
        if self.dflip and random.random() > 0.5:
            target = target.swapaxes(-1, -2)
            second_img = second_img.swapaxes(-1, -2)

        if self.blur > random.random():
            blur_transform = transforms.GaussianBlur(random.choice([3, 5]))
            target = blur_transform(target)
            second_img = blur_transform(second_img)

        if self.noise_factor != 0 and random.random() < self.noise:
            noise = torch.randn_like(target) * self.noise_factor * random.random()
            target = target + noise * 0.1
            second_img = second_img + noise
        # Normalize to -1, 1
        target = target * 2 - 1
        second_img = second_img * 2 - 1

        return target.swapaxes(-1, 0), second_img.swapaxes(-1, 0)

    def __len__(self):
        return len(self.file_list) * 2

    def __getitem__(self, i):
        flip = len(self.file_list) > i

        dict_mods = self.load_file(i)
        target, condition = self.transform(dict_mods, flip)
        class_label = int(flip)
        if self.use_multi_class_class_label:
            class_label = self.class_idx[flip]
        example = {
            "image": target,
            "condition": condition,
            # "class_label": condition,
            "class_label": class_label,
            # "c_crossattn": condition,
            "human_label": self.condition_types[-1] if flip else self.condition_types[0],
        }

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
