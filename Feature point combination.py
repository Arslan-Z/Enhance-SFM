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
#matcher.load_state_dict(torch.load("/root/autodl-tmp/LoFTR-master/notebooks/outdoor_ds.ckpt")['state_dict'])
matcher = matcher.eval().cuda()

def get_line_count(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    return len(lines)

default_cfg['coarse']

dir_path = "/root/autodl-tmp/openMVG/build/software/SfM/loftr-detilyshiyan301/"  # 保存.txt文件的目录
# 获取dir_path目录下的所有.txt文件
txt_files = glob.glob(os.path.join(dir_path, "*.txt"))
merged_path = "/root/autodl-tmp/openMVG/build/software/SfM/loftr-detilyshiyan301/merged.txt"  # 合并后的文件路径

filepath2 = r'/root/autodl-tmp/openMVG/build/software/SfM/matches_sequentialshiyan301/matches/matches.putative.txt'
file_path = merged_path
result = []
for txt_file in txt_files:
    
    print(txt_file)
    with open(txt_file, 'r') as infile:
        first_line = infile.readline().strip()  # 读取第一行并去除首尾空白字符
        if not first_line:
            continue
        first_value = first_line.split()[0]
        # 去掉后缀 .jpg
        filename_without_ext = first_value.replace('.jpg', '')
        filename_with_ext = filename_without_ext + '.feat'
        filepath0 = os.path.join('/root/autodl-tmp/openMVG/build/software/SfM/matches_sequentialshiyan301/matches',
                                 filename_with_ext)
        line_count_after3 = get_line_count(filepath0)#最初多少
        
        filename = first_line.split()[6]
        # 去掉后缀 .jpg
        filename_without_ext1 = filename.replace('.jpg', '')
        filename_with_ext1 = filename_without_ext1 + '.feat'
        filepath1 = os.path.join('/root/autodl-tmp/openMVG/build/software/SfM/matches_sequentialshiyan301/matches', filename_with_ext1)
        line_count_after4 = get_line_count(filepath1)  # 最初多少
        infile.seek(0)
        print(txt_file)
        # 打开文件并读取所有行
        # 打开文件并读取所有行
        line_count = 0
        with open(filepath0, 'a') as f:
            for line in infile:
                # 拆分每一行并取第一列数据
                line_split = line.split()
                data_to_write = line_split[2] + ' ' + line_split[3] + ' ' + line_split[4]+ ' ' + line_split[5] +'\n'
                f.write(data_to_write)  # 在文件末尾添加数据，以空格隔开
                # 更新行计数器
                line_count += 1
                print("新的数据1")
                print("新添加的" + line_split[0] + "数据1在第 {} 行".format(line_count))

        
        line=0
        print("新的数据21")
        infile.seek(0)
        print(txt_file)
        line_count = 0
        with open(filepath1, 'a') as f:
            for line in infile:
                # 拆分每一行并取所需数据
                line_split = line.split()
                data_to_write = line_split[8] + ' ' + line_split[9]  + ' ' + line_split[10]+ ' ' + line_split[11] +'\n'
                f.write(data_to_write)  # 在文件末尾添加数据，以空格隔开

                # 更新行计数器
                line_count += 1

                print("新的数据2")
                print("新添加的" + line_split[6] + "数据2在第 {} 行".format(line_count))

        result.append(filename_with_ext)



        with open(filepath2, 'r') as f1:
            lines1 = f1.readlines()
        infile.seek(0)
        print(txt_file)
        for num1 in range(len(lines1)):
            if lines1[num1].strip() == (first_line.split()[1] + ' ' + first_line.split()[7]) and ' ' not in lines1[num1+1].strip():
                print(f"找到，开始插入")
                with open(filepath2, 'r') as f1:
                    lines1 = f1.readlines()
                for line in infile:
                    num2 = int(lines1[num1 + 1].strip())
                    print(num2)
                    lines1.insert((num1 + num2 + 1), str(line_count_after3) + ' ' + str(line_count_after4) + '\n')
                    num_str = lines1[num1 + 1].strip()
                    num3 = int(num_str) + 1
                    lines1[num1 + 1] = str(num3) + '\n'

                    # 更新计数器
                    line_count_after3 = line_count_after3 + 1
                    line_count_after4 = line_count_after4 + 1
                    print(f"插入成功")

                # 最后，将文件写入操作移出循环
                with open(filepath2, 'w') as f1:
                    f1.writelines(lines1)
                break

