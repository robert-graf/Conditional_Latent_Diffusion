import argparse
import datetime
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.trainer import Trainer

from ldm.arguments import DataModuleFromConfig, TrainSettings, get_parser, nondefault_trainer_args, validate_resume
from ldm.callbacks import CUDACallback, ImageLogger, SetupCallback
from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config

torch.multiprocessing.set_sharing_strategy("file_system")


def default_config(trainer_kwargs, lightning_config, ckptdir):
    # default logger configs
    default_logger_cfgs = {
        # "wandb": {
        #    "target": "pytorch_lightning.loggers.WandbLogger",
        #    "params": {"name": nowname, "save_dir": logdir, "offline": opt.debug, "id": nowname},
        # },
        "testtube": {"target": "pytorch_lightning.loggers.TensorBoardLogger", "params": {"name": "testtube", "save_dir": logdir}},
    }
    default_logger_cfg = default_logger_cfgs["testtube"]
    logger_cfg = lightning_config.logger if "logger" in lightning_config else OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)  # type: ignore

    # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
    # specify which metric is used to determine best models
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {"dirpath": ckptdir, "filename": "{epoch:06}", "verbose": True, "save_last": True},
    }
    if hasattr(model, "monitor"):
        print(f"Monitoring {model.monitor} as checkpoint metric.")
        default_modelckpt_cfg["params"]["monitor"] = model.monitor
        default_modelckpt_cfg["params"]["save_top_k"] = 3

    modelckpt_cfg = lightning_config.modelcheckpoint if "modelcheckpoint" in lightning_config else OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
    print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
    if version.parse(pl.__version__) < version.parse("1.4.0"):
        trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)  # type: ignore

    # add callback which sets up log directory
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "main.SetupCallback",
            "params": {"resume": opt.resume, "now": now, "logdir": logdir, "ckptdir": ckptdir, "cfgdir": cfgdir, "config": config, "lightning_config": lightning_config},
        },
        "image_logger": {"target": "main.ImageLogger", "params": {"batch_frequency": 1000, "max_images": 8, "clamp": True}},
        "learning_rate_logger": {"target": "main.LearningRateMonitor", "params": {"logging_interval": "step"}},
        "cuda_callback": {"target": "main.CUDACallback"},
    }
    if version.parse(pl.__version__) >= version.parse("1.4.0"):
        default_callbacks_cfg.update({"checkpoint_callback": modelckpt_cfg})

    callbacks_cfg = lightning_config.callbacks if "callbacks" in lightning_config else OmegaConf.create()

    if "metrics_over_trainsteps_checkpoint" in callbacks_cfg:
        print("Caution: Saving checkpoints every n train steps without deleting. This might require some free space.")
        default_metrics_over_trainsteps_ckpt_dict = {
            "metrics_over_trainsteps_checkpoint": {
                "target": "pytorch_lightning.callbacks.ModelCheckpoint",
                "params": {
                    "dirpath": os.path.join(ckptdir, "trainstep_checkpoints"),
                    "filename": "{epoch:06}-{step:09}",
                    "verbose": True,
                    "save_top_k": -1,
                    "every_n_train_steps": 10000,
                    "save_weights_only": True,
                },
            }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    if "ignore_keys_callback" in callbacks_cfg and hasattr(trainer_opt, "resume_from_checkpoint"):
        callbacks_cfg.ignore_keys_callback.params["ckpt_path"] = trainer_opt.resume_from_checkpoint
    elif "ignore_keys_callback" in callbacks_cfg:
        del callbacks_cfg["ignore_keys_callback"]  # type: ignore

    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]  # type: ignore


if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()

    opt, unknown = parser.parse_known_args()  # type: ignore
    opt: TrainSettings
    nowname, logdir = validate_resume(opt, now)
    print(Path(logdir).absolute())
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    if not os.path.exists(cfgdir):
        Path(cfgdir).mkdir(exist_ok=True, parents=True)
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        OmegaConf.save(config, cfgdir + "/config.yaml")
        lightning_config = config.pop("lightning", OmegaConf.create())  # type: ignore
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp
        trainer_config["accelerator"] = "ddp"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if "gpus" not in trainer_config:
            del trainer_config["accelerator"]
            cpu = True
        else:
            gpu_info = trainer_config["gpus"]
            print(f"Running on GPUs {gpu_info}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config
        # config.model["params"]["ckpt_path"] = "logs/2024-09-03T08-58-26_nako/checkpoints/last.ckpt"
        # model
        model = instantiate_from_config(config.model)
        # trainer and callbacks
        trainer_kwargs = {}
        default_config(trainer_kwargs, lightning_config, ckptdir)

        trainer = Trainer(**trainer_opt.__dict__, **trainer_kwargs)

        trainer.logdir = logdir  # type: ignore ###

        # data
        data: DataModuleFromConfig = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()
        print("#### Data #####")
        for k in data.datasets:
            print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        n_gpu = len(str(lightning_config.trainer.gpus).strip(",").split(",")) if not cpu else 1
        accumulate_grad_batches = int(lightning_config.trainer.accumulate_grad_batches) if "accumulate_grad_batches" in lightning_config.trainer else 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * n_gpu * bs * base_lr
            print(
                f"Setting learning rate to {model.learning_rate:.2e} = {accumulate_grad_batches} (accumulate_grad_batches) * {n_gpu} (num_gpus) * {bs} (batchsize) * {base_lr:.2e} (base_lr)"
            )
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")

        # allow checkpointing via USR1
        def melk(*_args, **_kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*_args, **_kwargs):
            if trainer.global_rank == 0:
                import pudb  # type: ignore

                pudb.set_trace()

        import signal

        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger  # type: ignore
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        try:
            if trainer.global_rank == 0:
                print(trainer.profiler.summary())  # type: ignore
        except Exception:
            pass
