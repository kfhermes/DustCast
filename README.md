# DustCast: Nowcasting of dust and convective storms via diffusion-model predictions of SEVIRI RGB imagery

[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kfhermes/DustCast_draft)
[![paper](https://img.shields.io/badge/-SSRN-154881?style=flat&logo=ssrn&logoColor=white)](http://dx.doi.org/10.2139/ssrn.5430712)
[![license](https://img.shields.io/badge/MIT-green?style=flat&logo=license&logoColor=white)](https://github.com/kfhermes/DustCast_draft/tree/main?tab=MIT-1-ov-file#MIT-1-ov-file)

This repository contains the code to train, evaluate and run our nowcast model DustCast from the paper:

[Hermes, K., Marsham, J. H., Bollasina, M. A., Brooks, M., Klose, M., & Marenco, F. "Nowcasting of dust and convective storms via diffusion-model predictions of SEVIRI RGB imagery." Submitted to Weather and Climate Extremes.](http://dx.doi.org/10.2139/ssrn.5430712)


*DustCast is a simple diffusion model for image-based nowcasting of dust storms in West Africa. The model performs image-to-image translation in pixel space to predict next frames of the SEVIRI desert dust RGB composite, a product of false-colour satellite images highlighting both dust and deep convection. DustCast was trained for a large domain over West Africa and can predict both convective storms and convectively generated dust storms which currently operational numerical weather prediction (NWP) models may not reliably reproduce. Furthermore, it can generate ensemble predictions, allowing a probabilistic forecast assessment. On average, our model achieves useful skill (Fractions Skill Score > 0.5) for predicting dust storms up to 5 hours lead time, and for convective systems for up to 4 hours. Our approach provides a valuable tool that could be used in operational forecasting to improve the prediction of dust storms, convective storms, and indeed other weather events. Due to the technical similarity of RGB composite imagery from geostationary satellites, this approach could also be adapted to nowcast other RGB composites, such as those for ash, or convective storms.*

### Example nowcast initialised on 07 June 2024, 17 UTC
![animation](./plt/supplementary_animation.gif)


# Installation

We recommended to install the code in its own environment.
Clone this repository, then run in its main directory
```
pip install -e .
```
This command installs all required packages for running the model. If you also want to evaluate the model, install additional packages by running
```
pip install -e .[eval]
```
Note that this might take several minutes to complete. For this paper we used PyTorch 2.4.1 and PyTorch Lightning 2.5.0. Compatibility with newer versions is not guaranteed.

If you want to train the model, you also need to set up your WandB account. Sign up at [wandb.ai](http://wandb.ai), then set up your WandB API key by running
```
wandb login
```



# Using DustCast

## Generating nowcasts
We provide the pretrained model and preprocessed SEVIRI observations that can be used for running the case study from our paper. Further input data can be obtained directly from [EUMETSAT](https://navigator.eumetsat.int/product/EO:EUM:DAT:MSG:HRSEVIRI). For generating nowcasts, download the model weights from [Zenodo](https://doi.org/10.5281/zenodo.17508993) and place them into `/data/models`.
In the main DustCast directory, run
```
python nowcast.py --arg_init=20240607T1700 --n_fcframes 8 --n_ens=5
```
This will initialise a nowcast with the most recent data at 07 June 2024 17:00 and generate an ensemble prediction with 5 members for the next 8 frames (+2 hours). By default, the output (predicted brightness temperatures) is saved into a netCDF file. You can use quicklook tools like [ncview](https://anaconda.org/conda-forge/ncview) or [Panoply](https://www.giss.nasa.gov/tools/panoply/) to have a look into the output fields. DustCast can also save RGB images and predicted probabilities for dust and convection. For more settings, see the config file `/dustcast/config/nowcast.yaml`. Note that command line arguments always override settings from the config file.


# Training and evaluation
All scripts for training and evaluating DustCast can be found in `/dustcast/scripts`, config files can be found in `/dustcast/config`.

## Training the model
Use the config file `train.yaml` to make your settings. Then run
```
python train.py --config=/path/to/train.yaml
```
This will launch the training process and log the progress to your WandB account. We recommend running training on a GPU.


## Evaluating the model
Model evaluation is split into three blocks:
1. Generate forecasts

   Use the config file `eval.yaml` to make your settings. Then run
   ```
    python eval_generate_nowcasts.py --config=/path/to/eval.yaml
    ```
    This will launch the generation of forecasts for evaluation purposes. The script saves all forecasts into netcdf files, and includes observations for the same timesteps where available to allow easy comparisons.

2. Compute evaluation statistics
   
   To compute metrics for the previously generated evaluation forecasts, run
    ```
    python eval_compute_stats_nowcast.py --idir_data /path/to/evaluation/forecasts --idir_btbg /path/to/cloud-screened/background/images
    ```
   To generate baseline metrics for a persistence forecast, run
    ```
    python eval_compute_stats_persistence --idir_data /path/to/evaluation/data --idir_btbg /path/to/cloud-screened/background/images
    ```
    
3. Visualise the evaluation results
    
   Use the notebook `eval_visualise_metrics.ipynb` to compute mean evaluation metrics from the previously computed statistics. For a closer look at individual forecasts, use the notebook `eval_visualise_casestudy.ipynb` to generate map plots of the RGB images and detected feature masks.

# License
This code is released under the [MIT License](https://github.com/kfhermes/DustCast_draft/blob/main/LICENSE).