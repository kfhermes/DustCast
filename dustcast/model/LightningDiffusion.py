import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau

from dustcast.features.utils import sample_full_images
from dustcast.model.losses.SSIM import SSIM
from dustcast.model.utils_ddpm import linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule
from dustcast.wandblog.visualisation import log_images_to_wandb


class LightningDiffusion(pl.LightningModule):
    def __init__(self,
        model,
        sampler,
        timesteps=1000,
        loss_type='MSE',
        lr=1e-4,
        lr_sched='OneCycleLR',
        log_img_freq=5, # log every n epochs to wandb
        steps_per_epoch=None,
        beta_schedule="linear",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.sampler = sampler
        self.timesteps = timesteps
        self.lr = lr
        self.lr_sched = lr_sched
        self.log_img_freq = log_img_freq
        self.steps_per_epoch = steps_per_epoch
        self.beta_schedule = beta_schedule
        self.img_batch = None
        self.loss_fn = self.get_loss_fn(loss_type)


    def get_loss_fn(self, loss_type):
        print(f"Using loss function: {loss_type}")
        if loss_type == 'L1':
            loss_fn = nn.L1Loss()
        elif loss_type == 'MSE':
            loss_fn = nn.MSELoss()
        elif loss_type == 'SSIM':
            loss_fn = 1 - SSIM()
        else:
            raise ValueError(f"Loss not implemented: {loss_type}")
        return loss_fn

    
    def get_beta_schedule(self):
        if self.beta_schedule == 'linear':
            return linear_beta_schedule(self.timesteps)
        elif self.beta_schedule == 'cosine':
            return cosine_beta_schedule(self.timesteps)
        elif self.beta_schedule == 'sigmoid':
            return sigmoid_beta_schedule(self.timesteps)
        else:
            raise NotImplementedError(f"beta_schedule {self.beta_schedule} not implemented.")
    

    def noisify_ddpm(self, x_0, t, noise=None):
        if noise is None:
            device = x_0.device
            noise = torch.randn(x_0.shape, device=device)
        beta = self.get_beta_schedule()
        alpha = 1.-beta
        alphabar = alpha.cumprod(dim=0)
        ᾱ_t = alphabar[t].reshape(-1, 1, 1, 1).to(device)
        xt = ᾱ_t.sqrt()*x_0 + (1-ᾱ_t).sqrt()*noise
        return xt, t.to(device), noise
    
    
    def noisify_and_condition(self, batch_in, num_channels=3):
        x_0 = batch_in[:, -num_channels:]
        t = torch.randint(0, self.timesteps, (batch_in.shape[0],)).long()
        x_noisy, t, noise = self.noisify_ddpm(x_0=x_0, t=t)
        cond_x_noisy = torch.cat([batch_in[:, :-num_channels], x_noisy], dim=1)
        return cond_x_noisy, t, noise


    def compute_loss(self, batch_in, noise=None, num_channels=3):
        cond_x_noisy, t, noise = self.noisify_and_condition(batch_in)
        model_out = self.model(cond_x_noisy, t)
        return self.loss_fn(model_out, noise)


    def forward(self, x, *args, **kwargs):
        return self.compute_loss(x, *args, **kwargs)
    
        
    def training_step(self, batch):
        if self.model.device.type == 'cpu':
            batch = batch.float()
        loss = self.forward(batch)        
        self.log("train_loss", loss, sync_dist=True, on_step=True, on_epoch=True, prog_bar=True)
        return loss


    def append_frame_to_img_batch(self, batch, batch_idx, num_images=8):
        if self.img_batch is None:
            print('Initializing img_batch...')
            print(f'batch[0].shape: {batch[0].shape}')
            self.img_batch = [batch[0]]
        elif type(self.img_batch) is list:
            if len(self.img_batch) < num_images:
                print(f'Appending first item in val_batch {batch_idx} to img_batch...')
                self.img_batch.append(batch[0])


    def prepare_img_batch(self):
        if type(self.img_batch) is list:
            print('Converting img_batch to tensor.')
            self.img_batch = torch.stack(self.img_batch)
            if len(self.img_batch.shape) == 3:
                print('img_batch contains only one frame, adding batch dimension...')
                self.img_batch = self.img_batch.unsqueeze(0)


    def validation_step(self, batch_in, batch_idx, num_channels=3):
        self.append_frame_to_img_batch(batch_in, batch_idx)
        cond_x_noisy, t, noise = self.noisify_and_condition(batch_in)
        model_out = self.model(cond_x_noisy, t)
        loss = self.loss_fn(model_out, noise)
        self.log("val_loss", loss, sync_dist=True, on_step=False, on_epoch=True, prog_bar=True)

        me = torch.mean(model_out - noise)
        mae = nn.functional.l1_loss(model_out, noise)
        mse = nn.functional.mse_loss(model_out, noise)
        ssim = SSIM()(model_out, noise)

        self.log("val_me", me, sync_dist=True)
        self.log("val_mae", mae, sync_dist=True)
        self.log("val_mse", mse, sync_dist=True)
        self.log("val_ssim", ssim, sync_dist=True)
        return loss


    def on_validation_epoch_end(self, num_channels=3):
        if (
            (self.current_epoch % self.log_img_freq == 0 or self.current_epoch == (self.trainer.max_epochs-1)) and
            self.trainer.is_global_zero and
            self.img_batch is not None
            ):
                print(f'Running extended validation step for epoch {self.current_epoch}.')
                self.prepare_img_batch()
                assert self.img_batch.shape[1] == 4*num_channels, f"Expected channel dimension of {4*num_channels}, received {self.img_batch.shape[1]}"

                with torch.amp.autocast(self.device.type):
                    forecast = sample_full_images(model=self.model,
                                                sampler=self.sampler,
                                                start_frames=self.img_batch[:,:-num_channels],
                                                n_future_frames=4,
                                                n_channels=num_channels).squeeze()
                    if len(forecast.shape) == 3:
                        forecast = forecast.unsqueeze(0)

                    obs_frames = self.img_batch[:,-num_channels:]
                    fcs_frames = forecast[:,:num_channels]
                    me_1 = torch.mean(fcs_frames - obs_frames)
                    mae_1 = nn.functional.l1_loss(fcs_frames, obs_frames)
                    mse_1 = nn.functional.mse_loss(fcs_frames, obs_frames)
                    ssim_1 = SSIM()(fcs_frames, obs_frames)

                log_images_to_wandb(self.logger.experiment, "sampled_1fr", self.img_batch, forecast, plot_frame=0)
                log_images_to_wandb(self.logger.experiment, "sampled_4fr", self.img_batch, forecast, plot_frame=3)
                
                self.log("me_1fr", me_1)
                self.log("mae_1fr", mae_1)
                self.log("mse_1fr", mse_1)
                self.log("ssim_1fr", ssim_1)


    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr, eps=1e-5)
        total_lr_steps = self.steps_per_epoch * self.trainer.max_epochs
        if self.lr_sched=='OneCycleLR':
            scheduler = OneCycleLR(optimizer, total_steps=total_lr_steps, max_lr=self.lr)
        elif self.lr_sched=='CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=total_lr_steps, eta_min=0, last_epoch=-1)
        elif self.lr_sched=='ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.25, verbose=True)
        else:
            raise NotImplementedError(f'Learning rate scheduler {self.lr_sched} not implemented.')
        
        if self.lr_sched in ['OneCycleLR', 'CosineAnnealingLR']:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": 'step'}
            }
        elif self.lr_sched=='ReduceLROnPlateau':
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss", "interval": 'epoch', "frequency": 1,}
            }