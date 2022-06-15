import cv2
import torch
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset

if __name__ == "__main__":
    # Variables
    kitti = '2012'
    input_name = 'kitti2015_val20'
    crops_per_image = 64
    crop_dx = 52
    crop_dy = 182
    output_name = 'kitti2012_val24_small_temp'
    if kitti == '2012':
        root = Path('/z/erharj/kitti/2012/training/')
    else:
        root = Path('/z/erharj/kitti/2015/training/')

    # Read input file
    with open('./lists/{}.list'.format(input_name), 'rt') as fp:
        file_list = [Path(line.strip()) for line in fp]

    # Generate crop coordinates
    if kitti == '2012':
        x0s = [np.random.randint(low=0, high=cv2.imread(str(root / "colored_0" / file), cv2.IMREAD_COLOR).shape[0] - crop_dx, size=(crops_per_image)) for file in file_list]
        y0s = [np.random.randint(low=0, high=cv2.imread(str(root / "colored_0" / file), cv2.IMREAD_COLOR).shape[1] - crop_dy, size=(crops_per_image)) for file in file_list]
    else:
        x0s = [np.random.randint(low=0, high=cv2.imread(str(root / "image_2" / file), cv2.IMREAD_COLOR).shape[0] - crop_dx, size=(crops_per_image)) for file in file_list]
        y0s = [np.random.randint(low=0, high=cv2.imread(str(root / "image_2" / file), cv2.IMREAD_COLOR).shape[1] - crop_dy, size=(crops_per_image)) for file in file_list]        

    with open('./lists/{}.list'.format(output_name), 'wt') as fp:
        for file, x0, y0 in zip(file_list, x0s, y0s):
            for x, y in zip(x0, y0):
                fp.write('{} {} {} {} {}\n'.format(str(file), x, x+crop_dx, y, y+crop_dy))

    # # Conversion from old format
    # with open('./lists/{}.list'.format(input_name), 'rt') as fp:
    #     file_list = []
    #     crops = []
    #     for line in fp:
    #         ls = line.strip().split()
    #         file_list += [Path(ls[0])]
    #         if len(ls) == 5:
    #             crops += [[int(ls[1]), int(ls[1])+int(ls[3]), int(ls[2]), int(ls[2])+int(ls[4])]]
    #         else:
    #             crops += [[0, 0, 0, 0]]

    # with open('./lists/{}.list'.format(input_name), 'wt') as fp:
    #     for file, crop in zip(file_list, crops):
    #         fp.write('{} {} {} {} {}\n'.format(str(file), crop[0], crop[1], crop[2], crop[3]))
