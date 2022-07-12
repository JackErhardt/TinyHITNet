import cv2
import torch
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset

if __name__ == "__main__":
    # Variables
    dataset = 'kitti2015'
    input_name = 'kitti2015_val20'
    crops_per_image = 64
    # crop_dx, crop_dy = 56, 182 # Small
    # crop_dx, crop_dy = 112, 364 # Medium
    crop_dx, crop_dy = 200, 200
    output_name = 'kitti2015_val20_square200'

    if   dataset == 'kitti2012':
        root = Path('/z/erharj/kitti/2012/training/colored_0/')
    elif dataset == 'kitti2015':
        root = Path('/z/erharj/kitti/2015/training/image_2/')
    elif dataset == 'sceneflow':
        root = Path('/z/erharj/sceneflow/frames_finalpass/')
    else:
        print("Invalid Dataset: {}".format(dataset))
        root = None

    # Read input file
    with open('./lists/{}.list'.format(input_name), 'rt') as fp:
        file_list = [Path(line.strip()) for line in fp]

    # Generate crop coordinates
    x0s = [np.random.randint(low=0, high=cv2.imread(str(root / file), cv2.IMREAD_COLOR).shape[0] - crop_dx, size=(crops_per_image)) for file in file_list]
    y0s = [np.random.randint(low=0, high=cv2.imread(str(root / file), cv2.IMREAD_COLOR).shape[1] - crop_dy, size=(crops_per_image)) for file in file_list]

    with open('./lists/{}.list'.format(output_name), 'wt') as fp:
        for file, x0, y0 in zip(file_list, x0s, y0s):
            for x, y in zip(x0, y0):
                fp.write('{} {} {} {} {}\n'.format(str(file), x, x+crop_dx, y, y+crop_dy))