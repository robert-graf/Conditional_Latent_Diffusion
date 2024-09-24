import argparse
import csv
import datetime
import glob
import importlib
import os
import random
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from .data.base import Txt2ImgIterableBaseDataset
from .util import instantiate_from_config


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()  # type: ignore

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size : (worker_id + 1) * split_size]  # type: ignore
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)  # type: ignore
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)  # type: ignore
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)  # type: ignore


@dataclass()
class TrainSettings:
    name: str
    resume: str
    resume_from_checkpoint: str
    base: list[str]
    seed: int
    postfix: str
    logdir: str
    scale_lr: bool
    train: bool
    no_test: bool
    debug: bool


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-n", "--name", type=str, const=True, default="", nargs="?", help="postfix for logdir")
    parser.add_argument("-r", "--resume", type=str, const=True, default="", nargs="?", help="resume from logdir or checkpoint in logdir")
    help_ = "paths to base configs. Loaded from left-to-right. " "Parameters can be overwritten or added with command-line options of the form `--key value`."
    parser.add_argument("-b", "--base", nargs="*", metavar="base_config.yaml", help=help_, default=[])
    parser.add_argument("-t", "--train", type=str2bool, const=True, default=False, nargs="?", help="train")
    parser.add_argument("--no-test", type=str2bool, const=True, default=False, nargs="?", help="disable test")
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument("-d", "--debug", type=str2bool, nargs="?", const=True, default=False, help="enable post-mortem debugging")
    parser.add_argument("-s", "--seed", type=int, default=23, help="seed for seed_everything")
    parser.add_argument("-f", "--postfix", type=str, default="", help="post-postfix for default name")
    parser.add_argument("-l", "--logdir", type=str, default="logs", help="directory for logging dat shit")
    parser.add_argument("--scale_lr", type=str2bool, nargs="?", const=True, default=True, help="scale base-lr by n_gpu * batch_size * n_accumulate")
    return parser


def validate_resume(opt: TrainSettings, now):
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError(f"Cannot find {opt.resume}")
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)
    return nowname, logdir


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    # parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        train=None,
        validation=None,
        test=None,
        predict=None,
        wrap=False,
        num_workers=None,
        shuffle_test_loader=False,
        use_worker_init_fn=False,
        shuffle_val_dataloader=False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = {}
        self.num_workers = num_workers if num_workers is not None else (batch_size * 2)
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):  # noqa: ARG002
        self.datasets = {k: instantiate_from_config(self.dataset_configs[k]) for k in self.dataset_configs}
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets["train"], Txt2ImgIterableBaseDataset)
        init_fn = worker_init_fn if is_iterable_dataset or self.use_worker_init_fn else None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=not is_iterable_dataset, worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        init_fn = worker_init_fn if isinstance(self.datasets["validation"], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn else None
        return DataLoader(self.datasets["validation"], batch_size=self.batch_size, num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets["train"], Txt2ImgIterableBaseDataset)
        init_fn = worker_init_fn if is_iterable_dataset or self.use_worker_init_fn else None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size, num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, _shuffle=False):
        init_fn = worker_init_fn if isinstance(self.datasets["predict"], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn else None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size, num_workers=self.num_workers, worker_init_fn=init_fn)
