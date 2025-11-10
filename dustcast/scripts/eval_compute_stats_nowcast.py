import fire
import numpy as np

import dustcast.evaluation.metrics as m
from dustcast.evaluation.io import find_files_in_dir, sort_len_str, load_background_rgb_from_file, load_nowcast_from_file, save_metrics
from dustcast.evaluation.mask import get_ct_mask, get_du_mask
from dustcast.features.rgb import bt_to_rgb, resize_image
from dustcast.features.utils import mkdir_if_not_exists


def compute_stats_nowcast(idir_predictions="./west-africa/full-eval/FULL_12gwe56u_unet_RGB_512_100steps_5ens_24frames_256px",
                          idir_btbg="./west-africa/seviri_15dmean_cs",
                          odir="./west-africa/full-eval/metrics_nc",
                          crop = 8,
                          scales = [2, 4, 8, 16, 32, 64],
                          save_every=100,
                          ):
    eval_files = find_files_in_dir(idir_predictions, pattern='*.nc', recursive=True)
    eval_files = sort_len_str(eval_files)

    cnt = 0
    metrics_data = []
    for ii, file in enumerate(eval_files):
        print(f'Processing file {ii}: {file}')
        _, prd_bt, obs_bt, flag_obs, _, time_valid, time_init,_ ,_ = load_nowcast_from_file(file)

        # Only continue evaluation for index if cloud-screened 15 average ("background") is available
        rgb_bg = load_background_rgb_from_file(idir_btbg, time_init)
        if rgb_bg is None:
            print(f'rgb_bg file not found for index {ii}, time_init {time_init}. Skipping index.')
        else:
            cnt += 1
            img_size = obs_bt.shape[-1]
            rgb_bg = resize_image(rgb_bg, (img_size, img_size))

            if crop > 0:
                prd_bt = prd_bt[:, :, :, crop:-crop, crop:-crop]
                obs_bt = obs_bt[:, :, crop:-crop, crop:-crop]
                rgb_bg = rgb_bg[:, crop:-crop, crop:-crop]

            img_prd = bt_to_rgb(prd_bt, gamma=[1.0, 2.5, 1.0], lims = [[-4,2],[0,15],[261,289]], channel_dim=2)
            img_obs = bt_to_rgb(obs_bt, gamma=[1.0, 2.5, 1.0], lims = [[-4,2],[0,15],[261,289]], channel_dim=1)

            prd_mask_ct = get_ct_mask(prd_bt, channel_dim=2)
            obs_mask_ct = get_ct_mask(obs_bt, channel_dim=1)
            prd_mask_du = get_du_mask(prd_bt, rgb_bg, channel_dim=2)
            obs_mask_du = get_du_mask(obs_bt, rgb_bg, channel_dim=1)
            
            ct_P = np.mean(prd_mask_ct, axis=1)
            du_P = np.mean(prd_mask_du, axis=1)

            # ct_tp, ct_fp, ct_tn, ct_fn = m.compute_confusion_map(obs_mask_ct, ct_P)
            # du_tp, du_fp, du_tn, du_fn = m.compute_confusion_map(obs_mask_du, du_P)
            
            for fr, time in enumerate(time_valid):
                if flag_obs[fr]:
                    metrics_fr = {}
                    time_ini_str = time_init.strftime('%Y%m%d%H%M')
                    time_val_str = time.strftime('%Y%m%d%H%M')
                    metrics_head = {'Frame': fr,'t0': time_ini_str, 't_valid': time_val_str,'fr': fr, }

                    metrics_fr.update(metrics_head)

                    me_bt_r_mean, me_bt_g_mean, me_bt_b_mean, me_bt_mean = m.compute_me_stats(obs_bt[fr], prd_bt[fr], channel_dim=1)
                    metrics_fr.update({'me_bt_r_mean': me_bt_r_mean, 'me_bt_g_mean': me_bt_g_mean, 'me_bt_b_mean': me_bt_b_mean, 'me_bt_mean': me_bt_mean})

                    rmse_bt_r_mean, rmse_bt_g_mean, rmse_bt_b_mean, rmse_bt_mean = m.compute_rmse_stats(obs_bt[fr], prd_bt[fr], channel_dim=1)
                    metrics_fr.update({'rmse_bt_r_mean': rmse_bt_r_mean, 'rmse_bt_g_mean': rmse_bt_g_mean, 'rmse_bt_b_mean': rmse_bt_b_mean, 'rmse_bt_mean': rmse_bt_mean})

                    ct_roc_fpr, ct_roc_tpr, ct_roc_trs, ct_auc = m.compute_roc_stats(obs_mask_ct[fr], ct_P[fr], interpolate=False)
                    du_roc_fpr, du_roc_tpr, du_roc_trs, du_auc = m.compute_roc_stats(obs_mask_du[fr], du_P[fr], interpolate=False)
                    metrics_fr.update({'ct_roc_fpr': ct_roc_fpr, 'ct_roc_tpr': ct_roc_tpr, 'ct_roc_trs': ct_roc_trs, 'ct_auc': ct_auc})
                    metrics_fr.update({'du_roc_fpr': du_roc_fpr, 'du_roc_tpr': du_roc_tpr, 'du_roc_trs': du_roc_trs, 'du_auc': du_auc})

                    ct_pt, ct_pf, ct_hist = m.compute_calibration_stats(obs_mask_ct[fr], ct_P[fr], n_bins=10, interpolate=False)
                    du_pt, du_pf, du_hist = m.compute_calibration_stats(obs_mask_du[fr], du_P[fr], n_bins=10, interpolate=False)
                    metrics_fr.update({'ct_pt': ct_pt, 'ct_pf': ct_pf, 'ct_hist': ct_hist})
                    metrics_fr.update({'du_pt': du_pt, 'du_pf': du_pf, 'du_hist': du_hist})

                    ct_pfss = m.compute_pfss(obs_mask_ct[fr], prd_mask_ct[fr], window=scales)
                    du_pfss = m.compute_pfss(obs_mask_du[fr], prd_mask_du[fr], window=scales)
                    metrics_fr.update({'ct_pfss': ct_pfss, 'du_pfss': du_pfss})
                    
                    ct_fss = m.compute_fss_ens(obs_mask_ct[fr], prd_mask_ct[fr], window=scales)
                    du_fss = m.compute_fss_ens(obs_mask_du[fr], prd_mask_du[fr], window=scales)
                    metrics_fr.update({'ct_fss': ct_fss, 'du_fss': du_fss})
                    img_obs[fr].shape, img_prd[fr].shape
                    
                    ssim = m.compute_ssim_ens(img_obs[fr], img_prd[fr], data_range=1.0, mean=True)
                    metrics_fr.update({'ssim': ssim})

                    ct_cm = m.compute_confusion_matrix(obs_mask_ct[fr], ct_P[fr], normalize=None)
                    du_cm = m.compute_confusion_matrix(obs_mask_du[fr], du_P[fr], normalize=None)
                    metrics_fr.update({'ct_cm': ct_cm, 'du_cm': du_cm})
                    
                    ct_BS = m.compute_brier_score(obs_mask_ct[fr], ct_P[fr])
                    du_BS = m.compute_brier_score(obs_mask_du[fr], du_P[fr])
                    metrics_fr.update({'ct_BS': ct_BS, 'du_BS': du_BS})

                    du_frac = np.sum(obs_mask_du[fr])/len(obs_mask_du[fr].flatten())
                    ct_frac = np.sum(obs_mask_ct[fr])/len(obs_mask_ct[fr].flatten())
                    metrics_fr.update({'ct_frac': ct_frac, 'du_frac': du_frac})

                    metrics_data.append(metrics_fr)
    
        if cnt > 0 and cnt % save_every == 0 or ii == len(eval_files)-1:
            mkdir_if_not_exists(odir)
            save_metrics(f'{odir}/metrics_nowcast_num_fc-{cnt}.h5', cnt, metrics_data)



if __name__ == "__main__":
    fire.Fire(compute_stats_nowcast)