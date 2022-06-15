# Comment in when running as main
import sys
sys.path.append('./')

import cv2
import torch
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset

from dataset.utils import *


class KITTI2015Dataset(Dataset):
    def __init__(
        self,
        image_list,
        root,
        clip_size=None,
        training=False,
        augmentation=False,
        roi_padding=-1,
    ):
        super().__init__()
        with open(image_list, "rt") as fp:
            self.file_list = []
            self.rois = [] # top, bottom, left, right
            for line in fp:
                ls = line.strip().split()
                self.file_list += [Path(ls[0])]
                if len(ls) == 5:
                    self.rois += [[int(ls[1]), int(ls[2]), int(ls[3]), int(ls[4])]]
                else:
                    self.rois += [[0, 0, 0, 0]]
        self.root = Path(root)
        self.clip_size = clip_size
        self.training = training
        self.augmentation = augmentation
        self.roi_padding = roi_padding

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        left_path = self.root / "image_2" / self.file_list[index]
        right_path = self.root / "image_3" / self.file_list[index]
        disp_path = self.root / "disp_occ_0" / self.file_list[index]
        dxy_path = (
            self.root / "slant_window" / self.file_list[index].with_suffix(".npy")
        )
        image2roi = self.rois[index]

        data = {
            "left": np2torch(cv2.imread(str(left_path), cv2.IMREAD_COLOR), bgr=True),
            "right": np2torch(cv2.imread(str(right_path), cv2.IMREAD_COLOR), bgr=True),
            "disp": np2torch(cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256),
            "dxy": np2torch(np.load(dxy_path), t=False),
        }

        if self.clip_size is not None:
            data = clip_and_pad(data, self.clip_size, self.training)
        if self.training and self.augmentation:
            data = augmentation(data, self.training)
        if any(image2roi):
            data = crop_and_roi(data, image2roi, self.roi_padding)

        return data


if __name__ == "__main__":
    import torchvision
    from colormap import apply_colormap, dxy_colormap
    # from colormap import apply_colormap

    dataset = KITTI2015Dataset(
        "lists/kitti2015_val20.list",
        "/z/erharj/kitti/2015/training",
        training=True,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for ids, data in enumerate(loader):
        disp = data["disp"]
        disp = torch.clip(disp / 192 * 255, 0, 255).long()
        disp = apply_colormap(disp)

        dxy = data["dxy"]
        dxy = dxy_colormap(dxy)
        output = torch.cat((data["left"], data["right"], disp, dxy), dim=0)
        # output = torch.cat((data["left"], data["right"], disp), dim=0)
        torchvision.utils.save_image(output, "{:06d}.png".format(ids), nrow=1)
