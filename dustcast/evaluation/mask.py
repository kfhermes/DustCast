import numpy as np


def compare_gt_lt(data, str, trs):
    if str == 'gt':
        mask = data > trs
    elif str == 'lt':
        mask = data < trs
    elif str is None:
        pass # do nothing
    else:
        raise NotImplementedError(f"Option '{str}' not implemented.")
    return mask


def mask_bt(data_bt, trs = [None, None, None], operation=[None, None, None], channel_dim=0):
    mask = []
    for ii, op in enumerate(operation):
        if op is not None:
            mask.append(compare_gt_lt(np.take(data_bt,ii,channel_dim), operation[ii], trs[ii]))
    mask = np.stack(mask, axis=channel_dim)
    mask_stack = mask.all(axis=channel_dim) 
    return mask_stack


def mask_bt_anom(data_bt, data_bt_mean, trs = [None, None, None], operation=[None, None, None], channel_dim=0):
    mask = []
    for ii, op in enumerate(operation):
        if op is not None:
            anom = np.take(data_bt,ii,channel_dim) - data_bt_mean[ii,...]
            mask.append(compare_gt_lt(anom, operation[ii], trs[ii]))
    mask = np.stack(mask, axis=channel_dim)
    mask_stack = mask.all(axis=channel_dim) 
    return mask_stack


def merge_masks(*masks):
    if len(masks) == 0:
         ValueError("At least one mask must be provided.")
    else:
        return np.all(masks, axis=0)


def get_du_mask(bt, bg, channel_dim=0):
    dev = mask_bt(bt, trs = [0, 10, 285], operation=['gt', 'lt', 'gt'], channel_dim=channel_dim)
    anom = mask_bt_anom(bt, bg, trs = [None, -2, None], operation=[None, 'lt', None], channel_dim=channel_dim)
    mask = merge_masks(dev, anom)
    return mask


def get_ct_mask(bt, channel_dim=0):
    mask = mask_bt(bt, trs = [None, None, 233.15], operation=[None, None, 'lt'], channel_dim=channel_dim)
    return mask