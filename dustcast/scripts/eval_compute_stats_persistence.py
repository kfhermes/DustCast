import dustcast.evaluation.metrics as m
import fire
import numpy as np

from skimage.metrics import structural_similarity
from dustcast.evaluation.io import find_files_in_dir, load_background_rgb_from_file, save_metrics
from dustcast.evaluation.mask import get_ct_mask, get_du_mask
from dustcast.evaluation.utils import check_all_data_avail
from dustcast.features.datasets import SeviriDatasetEvaluation
from dustcast.features.rgb import bt_to_rgb, norm_to_bt, resize_image
from dustcast.features.utils import mkdir_if_not_exists

def compute_stats_persistence(
        img_size=256, # image size to use
        crop=8, # number of pixels from edge to crop
        n_inframes=3, # number of frames to use as input,
        n_fcframes=24, # number of future frames
        save_every=1000,# save every n computed idxs
        idir_data="./west-africa/seviri_npy/2024",
        idir_btbg='./west-africa/seviri_15dmean_cs',
        odir='./west-africa/full-eval/metrics_ps',
        normlims=[[-4,2],[0,15],[200,320]],
        scales=[2, 4, 8, 16, 32, 64],
    ):
    files = find_files_in_dir(idir_data, pattern="*.npy", recursive=False)
    eval_ds = SeviriDatasetEvaluation(files=files, num_frames=n_inframes+n_fcframes, img_size=img_size, normlims=normlims, stack_chfr=False)

    cnt = 0
    metrics_data = []
    print(f"Total number of persistence forecasts: {len(eval_ds)}")
    for idx in range(len(eval_ds)):
        data, time, data_avail = eval_ds[idx]
        time_init =  time[n_inframes-1]
        
        if check_all_data_avail(data_avail[:n_inframes]):             
            print(f"Processing index {idx}: {time_init}")
            # print(f'Data availability: {data_avail}')
            rgb_bg = load_background_rgb_from_file(idir_btbg, time_init)
            if rgb_bg is None:
                print(f'rgb_bg file not found for index {idx}, time_init {time_init}. Skipping index.')
            else:
                cnt += 1
                rgb_bg = resize_image(rgb_bg, (img_size, img_size))

                if crop > 0:
                    rgb_bg = rgb_bg[:, crop:-crop, crop:-crop]
                    data = data[:, :, crop:-crop, crop:-crop]

                obs = data[n_inframes:] # obs as prd
                ini = data[n_inframes-1] # obs at initialisation
                time_valid = time[n_inframes:] # expected time_valid
                avail = data_avail[n_inframes:] # bool indicating if obs available (True) or filled with zeros (False)        
                avail = avail.astype(bool)

                ini_bt = norm_to_bt(ini, normlims, channel_dim=0)
                obs_bt = norm_to_bt(obs,normlims, channel_dim=1)

                img_prd = bt_to_rgb(ini_bt, gamma=[1.0, 2.5, 1.0], lims = [[-4,2],[0,15],[261,289]], channel_dim=0)
                img_obs = bt_to_rgb(obs_bt, gamma=[1.0, 2.5, 1.0], lims = [[-4,2],[0,15],[261,289]], channel_dim=1)

                prd_mask_ct = get_ct_mask(ini_bt, channel_dim=0)
                obs_mask_ct = get_ct_mask(obs_bt, channel_dim=1)
                prd_mask_du = get_du_mask(ini_bt, rgb_bg, channel_dim=0)
                obs_mask_du = get_du_mask(obs_bt, rgb_bg, channel_dim=1)

                # ct_tp, ct_fp, ct_tn, ct_fn = m.compute_confusion_map(obs_mask_ct, prd_mask_ct)
                # du_tp, du_fp, du_tn, du_fn = m.compute_confusion_map(obs_mask_du, prd_mask_du)

                for fr in range(len(time_valid)):
                    if avail[fr]:
                        metrics_fr = {}
                        time_ini_str = time_init.strftime('%Y%m%d%H%M')
                        time_val_str = time_valid[fr].strftime('%Y%m%d%H%M')
                        metrics_head = {'Frame': fr,'idxs': idx,'t0': time_ini_str, 't_valid': time_val_str,'fr': fr, }

                        metrics_fr.update(metrics_head)

                        me_bt_r_mean, me_bt_g_mean, me_bt_b_mean, me_bt_mean = m.compute_me_stats(obs_bt[fr], ini_bt, channel_dim=0)
                        metrics_fr.update({'me_bt_r_mean': me_bt_r_mean, 'me_bt_g_mean': me_bt_g_mean, 'me_bt_b_mean': me_bt_b_mean, 'me_bt_mean': me_bt_mean})

                        rmse_bt_r_mean, rmse_bt_g_mean, rmse_bt_b_mean, rmse_bt_mean = m.compute_rmse_stats(obs_bt[fr], ini_bt, channel_dim=0)
                        metrics_fr.update({'rmse_bt_r_mean': rmse_bt_r_mean, 'rmse_bt_g_mean': rmse_bt_g_mean, 'rmse_bt_b_mean': rmse_bt_b_mean, 'rmse_bt_mean': rmse_bt_mean})

                        ct_fss = m.compute_fss_over_scales(obs_mask_ct[fr], prd_mask_ct, scales=scales)
                        du_fss = m.compute_fss_over_scales(obs_mask_du[fr], prd_mask_du, scales=scales)
                        metrics_fr.update({'ct_fss': ct_fss, 'du_fss': du_fss})

                        ssim = structural_similarity(img_obs[fr], img_prd, data_range=1.0, channel_axis=-1)
                        metrics_fr.update({'ssim': ssim})

                        ct_cm = m.compute_confusion_matrix(obs_mask_ct[fr], prd_mask_ct, normalize=None)
                        du_cm = m.compute_confusion_matrix(obs_mask_du[fr], prd_mask_du, normalize=None)
                        metrics_fr.update({'ct_cm': ct_cm, 'du_cm': du_cm})

                        ct_BS = m.compute_brier_score(obs_mask_ct[fr], prd_mask_ct)
                        du_BS = m.compute_brier_score(obs_mask_du[fr], prd_mask_du)
                        metrics_fr.update({'ct_BS': ct_BS, 'du_BS': du_BS})

                        du_frac = np.sum(obs_mask_du[fr])/len(obs_mask_du[fr].flatten())
                        ct_frac = np.sum(obs_mask_ct[fr])/len(obs_mask_ct[fr].flatten())
                        metrics_fr.update({'ct_frac': ct_frac, 'du_frac': du_frac})

                        metrics_data.append(metrics_fr)

        if cnt > 0 and cnt % save_every == 0 or idx == len(eval_ds)-1:
            mkdir_if_not_exists(odir)
            save_metrics(f'{odir}/metrics_persistence_num_fc-{cnt}.h5', cnt, metrics_data)



if __name__ == "__main__":
    fire.Fire(compute_stats_persistence)


