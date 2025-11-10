import os
from math import ceil
from pathlib import Path

import fire
import torch
import pytorch_lightning as pl
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from dustcast.features.DataModule import SeviriDataModule
from dustcast.model.LightningDiffusion import LightningDiffusion as lightning_model
from dustcast.model.setup import UNet2D, get_parameters_for_UNet2DModel
from dustcast.model.utils_ddpm import ddim_sampler

print("torch version: ", torch.__version__)


def get_data_files(dir):
    if dir is not None:
        print(f"Searching for data files in {dir}")
        path = Path(dir)
        files = list(path.rglob("*.npy"))
    else:
        print("No data directory provided, returning 'None'")
        files = []
    return files


def initialize_model(config):
    model_params = get_parameters_for_UNet2DModel(config.model_name, num_frames=4)
    if not config.load_model_from_checkpoint and not config.load_model_from_artifact:
        print("No checkpoint or artefact provided. Initializing model from scratch.")
        model = UNet2D(**model_params)
    elif config.load_model_from_checkpoint:
        model = UNet2D.from_lightning_ckpoint(model_params=model_params, checkpoint_file=config.load_model_from_checkpoint, is_state_dict=config.is_state_dict)
    elif config.load_model_from_artifact:
        model = UNet2D.from_wandb_artifact(model_params=model_params, artifact_name=config.load_model_from_artifact, is_state_dict=config.is_state_dict)
    return model
    
    

def train(conf):
    pl.seed_everything(conf.seed)
    wandb_logger = WandbLogger(project='fire+omega', log_model=True, config=OmegaConf.to_container(conf), tags=['train'])

    # NCCL environment variables (needed when launching manually)
    os.environ["NCCL_DEBUG"] = "INFO"  # Useful for debugging
    os.environ["NCCL_P2P_DISABLE"] = "0"  # Enables peer-to-peer communication
    os.environ["NCCL_IB_DISABLE"] = "0"  # Enables InfiniBand (if available)

    # More settings
    accelerator = "gpu"  if torch.cuda.is_available() else "cpu"
    devices = torch.cuda.device_count() if (accelerator == "gpu") else 1
    devices = min(devices, conf.max_devices)
    torch.set_float32_matmul_precision("medium")
    
    print(f"Accelerator: {accelerator}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    sampler = ddim_sampler(steps=conf.sampler_steps, same_noise=conf.same_noise, silent=True)
    model = initialize_model(conf)
    files = get_data_files(conf.idir_data)

    dm = SeviriDataModule(
        generate_datasets=True,
        save_datasets=False,
        load_datasets=False,
        onetmstp_files=files,
        ds_train_file=os.path.join(conf.dir_ds_save, f"train_ds_{conf.img_size}.pt"),
        ds_valid_file=os.path.join(conf.dir_ds_save, f"valid_ds_{conf.img_size}.pt"),
        batch_size=conf.batch_size,
        num_workers=conf.num_workers,
        img_size=conf.img_size,
        frac_train=conf.frac_train,
        len_block=conf.len_block,
        normlims=conf.normlims,
        load_into_memory=conf.load_into_memory,
    )
    dm.prepare_data()

    ltm = lightning_model(
            model=model,
            sampler=sampler,
            timesteps=conf.noise_steps,
            loss_type=conf.loss_type,
            lr=conf.lr,
            lr_sched=conf.lr_sched,
            steps_per_epoch=ceil(len(dm.train_dataloader()) / devices),
        )

    checkpoint_best = pl.callbacks.ModelCheckpoint(
        dirpath="./checkpoints",
        filename=f"{conf.model_name}_" + "{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        every_n_epochs=1,
        save_top_k=3,
        save_weights_only=True,
    )
    checkpoint_last = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor="epoch",
        mode="max",
        dirpath="./checkpoints",
        filename=f"{conf.model_name}_" + "{epoch}-{val_loss:.4f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_best, checkpoint_last, lr_monitor]

    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        accelerator=accelerator,
        devices=devices,
        strategy="ddp_find_unused_parameters_false" if (devices > 1) else "auto",
        callbacks=callbacks,
        logger=wandb_logger,
        max_epochs=conf.epochs,
        log_every_n_steps=50,
        accumulate_grad_batches=1,
        precision="bf16-mixed", # 16-mixed 32-true bf16-mixed
        profiler="simple",
    )

    trainer.fit(ltm, datamodule=dm)
    wandb.finish()



def main(config=None, **kwargs):
    config = OmegaConf.load(config) if (config is not None) else {}
    config.update(kwargs)
    print("Configuration:")
    print(OmegaConf.to_yaml(config))
    train(config)

if __name__ == "__main__":
    fire.Fire(main)