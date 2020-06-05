import numpy as np
from utils.data.data_process import get_bbox_3d_axes, get_bbox_center, get_bbox_size


def cal_mean_IOU_3d(pred, true, bbox_3d_list):
    ...


def cal_miss_num(pred, bbox_3d_list):
    miss_num = 0
    for bbox_3d in bbox_3d_list:
        patch_pred = pred[bbox_3d[2]:bbox_3d[5] + 1, bbox_3d[1]:bbox_3d[4], bbox_3d[0]:bbox_3d[3]]
        if 1 not in patch_pred:
            miss_num += 1
    return miss_num


def cal_axes_diff(long_axes_pred, short_axes_pred, mask_gt, bbox_3d_list, spacing):
    axes_true_list = get_bbox_3d_axes(mask_gt, bbox_3d_list, spacing)
    axes_pred_list = []
    axes_diff_list = []
    axes_true_list_out = []
    for i, bbox_3d in enumerate(bbox_3d_list):
        # if axes_true_list[i] == [0, 0]:
        #     continue
        axes_true_list_out.append(axes_true_list[i])
        bbox_center = get_bbox_center(bbox_3d)
        bbox_size = get_bbox_size(bbox_3d)
        sp = [int(bbox_center[i] - bbox_size[i] / 10) for i in range(len(bbox_size))]
        ep = [int(bbox_center[i] + bbox_size[i] / 10) + 1 for i in range(len(bbox_size))]
        long_patch = long_axes_pred[sp[0]:ep[0], sp[1]:ep[1], sp[2]:ep[2]]
        short_patch = short_axes_pred[sp[0]:ep[0], sp[1]:ep[1], sp[2]:ep[2]]
        long_axes = np.max(long_patch)
        short_axes = np.max(short_patch)
        # long_axes = long_axes_pred[int(bbox_center[0]), int(bbox_center[1]), int(bbox_center[2])]
        # short_axes = short_axes_pred[int(bbox_center[0]), int(bbox_center[1]), int(bbox_center[2])]

        axes_pred_list.append([long_axes, short_axes])
        axes_diff_list.append([abs(long_axes - axes_true_list[i][0]), abs(short_axes - axes_true_list[i][1])])
    return axes_true_list_out, axes_pred_list, axes_diff_list


def cal_axes(long_axes_pred, short_axes_pred, bbox_3d_list):
    axes_pred_list = []
    for i, bbox_3d in enumerate(bbox_3d_list):
        bbox_center = get_bbox_center(bbox_3d)
        bbox_size = get_bbox_size(bbox_3d)
        sp = [int(bbox_center[i] - bbox_size[i] / 10) for i in range(len(bbox_size))]
        ep = [int(bbox_center[i] + bbox_size[i] / 10) + 1 for i in range(len(bbox_size))]
        long_patch = long_axes_pred[sp[0]:ep[0], sp[1]:ep[1], sp[2]:ep[2]]
        short_patch = short_axes_pred[sp[0]:ep[0], sp[1]:ep[1], sp[2]:ep[2]]
        long_axes = np.max(long_patch)
        short_axes = np.max(short_patch)
        # long_axes = long_axes_pred[int(bbox_center[0]), int(bbox_center[1]), int(bbox_center[2])]
        # short_axes = short_axes_pred[int(bbox_center[0]), int(bbox_center[1]), int(bbox_center[2])]

        axes_pred_list.append([long_axes, short_axes])
    return axes_pred_list