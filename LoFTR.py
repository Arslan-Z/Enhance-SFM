import os
os.chdir("..")
from copy import deepcopy
import gc

import random
import glob
import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure

from copy import deepcopy
from src.loftr import LoFTR, default_cfg

dir_path = "/root/autodl-tmp/openMVG/build/software/SfM/imagesshiyan301"
ext = ".jpg"
# The default config uses dual-softmax.
# The outdoor and indoor models share the same config.
# You can change the default values like thr and coarse_match_type.
_default_cfg = deepcopy(default_cfg)
_default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt
matcher = LoFTR(config=_default_cfg)
matcher.load_state_dict(torch.load("/root/autodl-tmp/LoFTR-master/notebooks/indoor_ds_new.ckpt")['state_dict'])
matcher = matcher.eval().cuda()

# The default config uses dual-softmax.
# The outdoor and indoor models share the same config.
# You can change the default values like thr and coarse_match_type.
from src.loftr import LoFTR, default_cfg

# The default config uses dual-softmax.
# The outdoor and indoor models share the same config.
# You can change the default values like thr and coarse_match_type.
matcher = LoFTR(config=default_cfg)
matcher.load_state_dict(torch.load("/root/autodl-tmp/LoFTR-master/notebooks/outdoor_ds.ckpt")['state_dict'])
matcher = matcher.eval().cuda()

def get_line_count(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    return len(lines)

default_cfg['coarse']

# Load example images
# Find all image files in directory
image_files = sorted([f for f in os.listdir(dir_path) if f.endswith(ext)], key=lambda x: int(x.split("_")[1].split(".")[0]))
for i in range(len(image_files)):
    i = int(i)
    img0_pth = os.path.join(dir_path, image_files[i])
    img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)  # 换成灰度图
    img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1] // 2, img0_raw.shape[0] // 2))
    img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1] // 8 * 8, img0_raw.shape[0] // 8 * 8))  # input size shuold be divisible by 8
    img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
    for j in range(i + 1, min(i+6, len(image_files))):
    
        j = int(j)

        img1_pth = os.path.join(dir_path, image_files[j])

        print(f"Pair {i + 1}: {img0_pth}, {img1_pth}")
        if not os.path.exists(img1_pth):
            print(f"File {img1_pth} does not exist.")
            continue
        img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
        if img1_raw is None:
            print(f"Could not read file {img1_pth}.")
            continue
        img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1] // 2, img1_raw.shape[0] // 2))
        img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//8*8, img1_raw.shape[0]//8*8))# img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
        img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
        batch = {'image0': img0, 'image1': img1}

        # Inference with LoFTR and get prediction使用 LoFTR 进行推理并获得预测
        with torch.no_grad():
            matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()# Draw



        mask = mconf >= 0
        mkpts0 = mkpts0[mask]
        mkpts1 = mkpts1[mask]
        mconf = mconf[mask]

    #    if len(mkpts0) > 1500 or len(mkpts1) > 1500:
   #         del img1_raw, img1, batch, mkpts0, mkpts1, mconf  # , fig
  #          gc.collect()
 #           continue
#        else:
        

        color = cm.jet(mconf)
        text = [
            'LoFTR',
            'Matches: {}'.format(len(mkpts0)),
        ]

        #fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text)
        #filename = os.path.basename(img0_pth).split('.')[0] + '_' + os.path.basename(img1_pth).split('.')[0] + '_matching_figure.png'
        # 设置dpi参数
        #dpi = 72
        # 获取当前程序的目录
        #dir_path1 = os.path.dirname(os.path.realpath(__file__))
        dir_path1 = "/root/autodl-tmp/openMVG/build/software/SfM/loftr-detilyshiyan301"
        # 保存图片到当前程序目录下
        #file_path = os.path.join(dir_path1, filename)
        #fig.savefig(file_path, dpi=dpi, format="png")
         #Save filtered matches to a text file
        i = int(i)
        j = int(j)
        mkpts0 = np.array(mkpts0)
        mkpts1 = np.array(mkpts1)
        # Scale the feature points back to the original image size
        mkpts0[:, 0] *= 2  # Scale x coordinate
        mkpts0[:, 1] *= 2  # Scale y coordinate
        mkpts1[:, 0] *= 2  # Scale x coordinate
        mkpts1[:, 1] *= 2  # Scale y coordinate
        save_path = os.path.join(dir_path1, f"{os.path.basename(img0_pth).split('.')[0]}_{os.path.basename(img1_pth).split('.')[0]}_matches.txt")

        np.savetxt(save_path, np.column_stack((np.repeat(image_files[i], len(mkpts0)), np.repeat(i, len(mkpts0)), mkpts0[:, 0],mkpts0[:, 1], np.repeat(image_files[j], len(mkpts1)), np.repeat(j, len(mkpts1)),mkpts1[:, 0], mkpts1[:, 1])), fmt='%s', delimiter=' ')

        del img1_raw,  img1, batch, mkpts0, mkpts1, mconf#, fig
        gc.collect()

    del img0_raw, img0
    gc.collect()
