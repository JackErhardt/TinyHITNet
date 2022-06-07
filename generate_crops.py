import cv2
import torch
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset

if __name__ == "__main__":
    kitti = '2012'
    input_name = 'kitti2012_val24'
    crops_per_image = 64
    crop_dx = 52
    crop_dy = 182
    output_name = 'kitti2012_val24_small'
    if kitti == '2012':
        root = Path('/z/erharj/kitti/2012/training/')
    else:
        root = Path('/z/erharj/kitti/2015/training/')

    with open('./lists/{}.list'.format(input_name), 'rt') as fp:
        file_list = [Path(line.strip()) for line in fp]


    x = []
    y = []

    if kitti == '2012':
        x_list = [np.random.randint(low=0, high=cv2.imread(str(root / "colored_0" / file), cv2.IMREAD_COLOR).shape[0] - crop_dx, size=(crops_per_image)) for file in file_list]
        y_list = [np.random.randint(low=0, high=cv2.imread(str(root / "colored_0" / file), cv2.IMREAD_COLOR).shape[1] - crop_dy, size=(crops_per_image)) for file in file_list]
    else:
        x_list = [np.random.randint(low=0, high=cv2.imread(str(root / "image_2" / file), cv2.IMREAD_COLOR).shape[0] - crop_dx, size=(crops_per_image)) for file in file_list]
        y_list = [np.random.randint(low=0, high=cv2.imread(str(root / "image_2" / file), cv2.IMREAD_COLOR).shape[1] - crop_dy, size=(crops_per_image)) for file in file_list]        

    with open('./lists/{}.list'.format(output_name), 'wt') as fp:
        for file, xs, ys in zip(file_list, x_list, y_list):
            for x, y in zip(xs, ys):
                fp.write('{} {} {} {} {}\n'.format(str(file), x, y, crop_dx, crop_dy))

    print("Complete!")