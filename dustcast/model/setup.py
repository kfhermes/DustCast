from pathlib import Path

import torch
import wandb
from diffusers import UNet2DModel


def get_parameters_for_UNet2DModel(model_name="unet_512_rgb", num_frames=4):
    "Return the parameters for the diffusers UNet2d model"
    if model_name == "unet_256_rgb":
        return dict(
            in_channels=num_frames*3,
            out_channels=3,
            time_embedding_type ='positional',
            down_block_types = ('DownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D'),
            up_block_types = ('AttnUpBlock2D', 'AttnUpBlock2D', 'AttnUpBlock2D', 'UpBlock2D'),
            block_out_channels=(32, 64, 128, 256),
            layers_per_block = 2,
            mid_block_scale_factor = 1,
            downsample_padding = 1,
            downsample_type = 'conv',
            upsample_type = 'conv',
            dropout = 0.0,
            act_fn = 'silu',
            attention_head_dim = 8,
            norm_num_groups=8,
            norm_eps = 1e-5,
            add_attention = True,
            )
    elif model_name == "unet_512_rgb":
        return dict(
            in_channels=num_frames*3,
            out_channels=3,
            time_embedding_type ='positional',
            down_block_types = ('DownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D'),
            up_block_types = ('AttnUpBlock2D', 'AttnUpBlock2D', 'AttnUpBlock2D', 'UpBlock2D'),
            block_out_channels=(64, 128, 256, 512),
            layers_per_block = 2,
            mid_block_scale_factor = 1,
            downsample_padding = 1,
            downsample_type = 'conv',
            upsample_type = 'conv',
            dropout = 0.0,
            act_fn = 'silu',
            attention_head_dim = 8,
            norm_num_groups=8,
            norm_eps = 1e-5,
            add_attention = True,
            )
    else:
        raise(f"Model name not found: {model_name}")
    

def init_unet(model):
    "From Jeremy's bag of tricks on fastai V2 2023"
    for o in model.down_blocks:
        for p in o.resnets:
            p.conv2.weight.data.zero_()

    for o in model.up_blocks:
        for p in o.resnets: p.conv2.weight.data.zero_()
    model.conv_out.weight.data.zero_()


class MultiGPUloggedModel:
    "Model that can be loaded from a accelerate or lightning checkpoint file saved during multi GPU training"
    def _get_model_state_dict(self, checkpoint):
        "Get the model state dict from a lightning state dict; do not use weights of loss fucntion"
        state_dict = checkpoint['state_dict']
        return {k: v for k, v in state_dict.items() if k.startswith("model")}

    def _remove_prefix(self, state_dict, prefix="model."):
        "Remove prefix from dict keys"
        new_state_dict = dict()
        for key, value in state_dict.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
            else:
                new_key = key
            new_state_dict[new_key] = value
        return new_state_dict
    
    @classmethod
    def from_lightning_ckpoint(cls, model_params, checkpoint_file, is_state_dict=False):
        model = cls(**model_params)
        print(f"Loading model from checkpoint file: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'), weights_only=True)
        state_dict = cls()._get_model_state_dict(checkpoint) if not is_state_dict else checkpoint
        state_dict = cls()._remove_prefix(state_dict, prefix="model.")
        state_dict = cls()._remove_prefix(state_dict, prefix="module.")
        model.load_state_dict(state_dict)
        return model

    @classmethod
    def from_wandb_artifact(cls, model_params, artifact_name, is_state_dict=False):
        api = wandb.Api()
        artifact = api.artifact(artifact_name, type='model')
        artifact_dir = Path(artifact.download())
        print(f"Artifact downloaded to: {artifact_dir}")
        checkpoint_file = list(artifact_dir.glob("*"))[0]
        return cls.from_lightning_ckpoint(model_params, checkpoint_file, is_state_dict)


class UNet2D(UNet2DModel, MultiGPUloggedModel):
    def __init__(self, *x, **kwargs):
        super().__init__(*x, **kwargs)
        init_unet(self)

    def forward(self, *x, **kwargs):
        return super().forward(*x, **kwargs).sample


class UNet2D_RandInit(UNet2DModel, MultiGPUloggedModel):
    def __init__(self, *x, **kwargs):
        super().__init__(*x, **kwargs)
        # init_unet(self)

    def forward(self, *x, **kwargs):
        return super().forward(*x, **kwargs).sample