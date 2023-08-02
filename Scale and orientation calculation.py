
import os
import numpy as np
from joblib import Parallel, delayed

sift_dir = '/root/autodl-tmp/openMVG/build/software/SfM/matches_sequentialshiyan301/matches'
loftr_dir = '/root/autodl-tmp/openMVG/build/software/SfM/loftr-detilyshiyan301'

def process_sift_file(filename):
    print(f"Processing sift file: {filename}")
    with open(os.path.join(sift_dir, filename), 'r') as file:
        features = []
        for line in file.readlines():
            x, y, scale, orientation = map(float, line.split())
            features.append(((x, y), scale, orientation))
        return filename[:-5] + '.jpg', features 

def process_loftr_file(filename, sift_features):
    print(f"Processing loftr match file: {filename}")
    with open(os.path.join(loftr_dir, filename), 'r') as file:
        lines = file.readlines()
        # if the data is processed, it should have more than 8 values in the last line

        matches = []
        for line in lines:
            parts = line.split()
            img1, img2 = parts[0], parts[4]
            num1, num2 = parts[1], parts[5]
            x1, y1 = map(float, parts[2:4])
            x2, y2 = map(float, parts[6:8])

            features1 = sift_features.get(img1, [])
            features2 = sift_features.get(img2, [])
            closest_feature1 = min(features1, key=lambda feature: (feature[0][0] - x1) ** 2 + (
                        feature[0][1] - y1) ** 2) if features1 else ((0, 0), 0, 0)
            closest_feature2 = min(features2, key=lambda feature: (feature[0][0] - x2) ** 2 + (
                        feature[0][1] - y2) ** 2) if features2 else ((0, 0), 0, 0)
            matches.append(((x1, y1, closest_feature1[1], closest_feature1[2]),
                            (x2, y2, closest_feature2[1], closest_feature2[2])))
        with open(os.path.join(loftr_dir, filename), 'w') as file:
            for match in matches:
                file.write(
                    #f"{img1} {num1} {match[0][0]} {match[0][1]} {match[0][2]} {match[0][3]} {img2} {num2} {match[1][0]} {match[1][1]} {match[1][2]} {match[1][3]}\n")
                    f"{img1} {num1} {match[0][0]} {match[0][1]} {0} {0} {img2} {num2} {match[1][0]} {match[1][1]} {0} {0}\n")
    print(f"Finished processing loftr match file: {filename}")

sift_files = [f for f in os.listdir(sift_dir) if f.endswith('.feat')]
loftr_files = [f for f in os.listdir(loftr_dir) if f.endswith('_matches.txt')]

# Process sift files
results = Parallel(n_jobs=24)(delayed(process_sift_file)(filename) for filename in sift_files)
sift_features = dict(results)

# Process loftr files
Parallel(n_jobs=15)(delayed(process_loftr_file)(filename, sift_features) for filename in loftr_files)

