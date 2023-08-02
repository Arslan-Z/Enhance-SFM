import os
import cv2
import numpy as np
import pandas as pd
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

images_folder = '/root/autodl-tmp/openMVG/build/software/SfM/imagesshiyan301'
feat_folder = '/root/autodl-tmp/openMVG/build/software/SfM/matches_sequentialshiyan301/matches'
txt_folder = '/root/autodl-tmp/openMVG/build/software/SfM/loftr-detilyshiyan301'

image_files = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
superpixel_stat = {}
for image_file in image_files:
    print(f"Processing image file: {image_file}")
    image_path = os.path.join(images_folder, image_file)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to open image at {image_path}")

    superpixels = slic(image, n_segments=100, compactness=10, sigma=1)

    feat_path = os.path.splitext(image_file)[0] + '.feat'
    feat_path = os.path.join(feat_folder, feat_path)
    print(f"Reading features from: {feat_path}")
    features = pd.read_csv(feat_path, header=None, sep=' ', names=['x', 'y', 'val1', 'val2'])

    superpixel_feat_count = np.zeros(np.max(superpixels) + 1)
    for _, feature in features.iterrows():
        superpixel_id = superpixels[int(feature['y']), int(feature['x'])]
        superpixel_feat_count[superpixel_id] += 1

    average_feat_count = np.mean(superpixel_feat_count)
    superpixel_stat[image_file] = (superpixel_feat_count, average_feat_count)

txt_files = os.listdir(txt_folder)
for txt_file in txt_files:
    print(f"Processing txt file: {txt_file}")
    txt_path = os.path.join(txt_folder, txt_file)
    matches = pd.read_csv(txt_path, header=None, sep=" ", names=['image1', 'point1', 'x1', 'y1', 'image2', 'point2', 'x2', 'y2'])  # adjust this according to your .txt file format

    image1_name, image2_name = txt_file.replace('_matches.txt', '').split('_frame_')
    image2_name = 'frame_' + image2_name

    image1_name = image1_name + '.jpg'
    image2_name = image2_name + '.jpg'

    matches = matches[(matches['x1'] < 1920) & (matches['x2'] < 1920) & 
                      (matches['y1'] < 1080) & (matches['y2'] < 1080)]


    for index, row in matches.iterrows():
        x1, y1 = int(row['x1']), int(row['y1'])
        x2, y2 = int(row['x2']), int(row['y2'])

        try:
            superpixel_id1 = superpixels[int(y1), int(x1)]
            superpixel_id2 = superpixels[int(y2), int(x2)]

            if superpixel_stat[image1_name][0][superpixel_id1] > superpixel_stat[image1_name][1]/2 or \
               superpixel_stat[image2_name][0][superpixel_id2] > superpixel_stat[image2_name][1]/2:
                matches = matches.drop(index)
        except IndexError:
            print(f"Out of bounds error for coordinates x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            


    matches.to_csv(txt_path, header=None, index=None, sep=' ')  # Use space as separator
    print(f"Finished processing: {txt_file}")
