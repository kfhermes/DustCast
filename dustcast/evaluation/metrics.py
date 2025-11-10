from copy import deepcopy

import numpy as np
from skimage.metrics import mean_squared_error, structural_similarity
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score, roc_curve

from dustcast.evaluation.fss_det import fss as fss_det
from dustcast.evaluation.fss_prob import fss_prob


def rmse(obs, prd):
    return np.sqrt(mean_squared_error(obs, prd))


def roc_auc_score_check(obs, fc_P):
    if len(np.unique(obs))>1:
        roc_auc = roc_auc_score(obs, fc_P)
    else:
        roc_auc = np.nan
    return roc_auc


def compute_fss_over_scales(obs, prd, scales):
    fss_scales = np.zeros(len(scales))
    for ii, scale in enumerate(scales):
        fss_scales[ii] = fss_det(X_f=obs, X_o=prd, thr=0.5, scale=scale)
    return fss_scales


def compute_roc_stats(obs_mask, prd_P, interpolate=False, interpolate_n=100):
    obs_mask = obs_mask.flatten()
    prd_P = prd_P.flatten()
    fpr, tpr, trs = roc_curve(y_true=obs_mask, y_score=prd_P, drop_intermediate=False)
    auc = roc_auc_score_check(obs_mask, prd_P)
    if interpolate:
        fpr_stp = np.linspace(0, 1, interpolate_n)
        tpr = np.interp(fpr_stp, fpr, tpr)
        fpr = fpr_stp
    return fpr, tpr, trs, auc


def get_cal_histogram(fc_P, bins = 10):
    counts, bin_edges = np.histogram(fc_P.flatten(), bins=bins, density=True)
    # Get relative fraction of bins
    rel_frac = counts/np.sum(counts)
    return rel_frac


def compute_calibration_stats(obs_mask, prd_P, n_bins=11, interpolate=False, interpolate_n=100):
    obs_mask = obs_mask.flatten()
    prd_P = prd_P.flatten()
    pt, pf = calibration_curve(obs_mask, prd_P, n_bins=n_bins, strategy='uniform')
    hist = get_cal_histogram(prd_P, bins=n_bins)
    if interpolate:
        pt_stp = np.linspace(0, 1, interpolate_n)
        pf = np.interp(pt_stp, pt, pf)
        pt = pt_stp
    return pt, pf, hist


def compute_me_stats(obs, prd, channel_dim=0): # channel_dim=0 for pers, channel_dim=1 for ens
    if len(prd.shape) == 4: # ens input
        obs = np.repeat(obs[np.newaxis, :], prd.shape[0], axis=0) # replicate obs to match ens dimension
    me_r_mean = np.mean(np.take(prd, 0, channel_dim) - np.take(obs, 0, channel_dim))
    me_g_mean = np.mean(np.take(prd, 1, channel_dim) - np.take(obs, 1, channel_dim))
    me_b_mean = np.mean(np.take(prd, 2, channel_dim) - np.take(obs, 2, channel_dim))
    me_mean = np.mean(prd - obs)
    return me_r_mean, me_g_mean, me_b_mean, me_mean


def compute_rmse_stats(obs, prd, channel_dim=0): # channel_dim=0 for pers, channel_dim=1 for ens
    if len(prd.shape) == 4: # ens input
        obs = np.repeat(obs[np.newaxis, :], prd.shape[0], axis=0) # replicate obs to match ens dimension
    rmse_r_mean = rmse(np.take(prd, 0, channel_dim), np.take(obs, 0, channel_dim))
    rmse_g_mean = rmse(np.take(prd, 1, channel_dim), np.take(obs, 1, channel_dim))
    rmse_b_mean = rmse(np.take(prd, 2, channel_dim), np.take(obs, 2, channel_dim))
    rmse_mean = rmse(obs, prd)
    return rmse_r_mean, rmse_g_mean, rmse_b_mean, rmse_mean


def compute_std_stats(prd_ens, ens_mean=True):
    std_r = np.std(prd_ens[:,0], axis=0)
    std_g = np.std(prd_ens[:,1], axis=0)
    std_b = np.std(prd_ens[:,2], axis=0)
    std_mean = np.mean(std_ens, axis=0)
    if ens_mean:
        std_r = np.mean(std_r)
        std_g = np.mean(std_g)
        std_b = np.mean(std_b)
        std_mean = np.mean(std_mean)
    return std_r, std_g, std_b, std_mean


def compute_var_stats(prd_ens, ens_mean=True):
    var_r = np.var(prd_ens[:,0], axis=0)
    var_g = np.var(prd_ens[:,1], axis=0)
    var_b = np.var(prd_ens[:,2], axis=0)
    var_mean = np.mean(var_ens, axis=0)
    if ens_mean:    
        var_r = np.mean(var_r)
        var_g = np.mean(var_g)
        var_b = np.mean(var_b)
        var_mean = np.mean(var_mean)
    return var_r, var_g, var_b, var_mean


def compute_brier_score(obs_mask, prd_P, pos_label=1):
    obs_mask = obs_mask.flatten()
    prd_P = prd_P.flatten()
    bs = brier_score_loss(obs_mask, prd_P, pos_label=pos_label)
    return bs


def compute_pfss(obs_mask, prd_mask, thrsh=[0.5], window=[2, 4, 8, 16, 32, 64]):
    pfss = fss_prob(obs=obs_mask, fcst=prd_mask, thrsh=thrsh, window=window)
    return pfss[0]


def compute_fss_ens(obs_mask, prd_mask, window=[2, 4, 8, 16, 32, 64], mean=True):
    ens = prd_mask.shape[0]
    fss = []
    for ii in range(ens):
        fss.append(compute_fss_over_scales(obs_mask, prd_mask[ii], window))
    fss = np.array(fss)
    if mean:
        fss = np.mean(fss, axis=0)
    return fss


def compute_ssim_ens(obs, pred, data_range=1.0, mean=True, channel_axis=2):
    assert len(pred.shape) == 4 , 'Expected 4D array for pred'
    ssim = []
    for ii in range(pred.shape[0]):
        ssim.append(structural_similarity(obs, pred[ii], data_range=data_range, channel_axis=channel_axis))
    ssim = np.sort(np.array(ssim))
    if mean:
        ssim = np.mean(ssim, axis=0)
    return ssim


def compute_confusion_matrix(obs_mask, prd_P, normalize=None):
    obs_mask = obs_mask.flatten()
    prd_P = prd_P.flatten()
    prd_P = (prd_P > 0.5).astype(int)
    cm = confusion_matrix(obs_mask, prd_P, normalize=normalize)
    return cm


def compute_confusion_map(obs_mask, prd_P, keep_predicted_P=False):
    if not keep_predicted_P:
        prd_P = (prd_P > 0.5).astype(np.float16)

    tp = deepcopy(prd_P)
    fp = deepcopy(prd_P)
    tn = deepcopy(prd_P)
    fn = deepcopy(prd_P)
    # print(f'prd_P.shape {prd_P.shape}')
    # print(f'obs_mask.shape {obs_mask.shape}')

    tp[obs_mask==False] = np.nan
    fp[obs_mask==False] = np.nan
    tn[obs_mask==True] = np.nan
    fn[obs_mask==True] = np.nan
    return tp, fp, tn, fn