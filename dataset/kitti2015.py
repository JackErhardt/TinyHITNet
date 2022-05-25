# # Comment in when running as main
# import sys
# sys.path.append('./')

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
        crop_size=None,
        training=False,
        augmentation=False,
        crops_per_image=None,
        crop_width=None,
        crop_height=None
    ):
        super().__init__()
        with open(image_list, "rt") as fp:
            self.file_list = [Path(line.strip()) for line in fp]
        self.root = Path(root)
        self.crop_size = crop_size
        self.training = training
        self.augmentation = augmentation
        self.crops_per_image = crops_per_image
        self.crop_width = crop_width
        self.crop_height = crop_height
        if self.crops_per_image is not None:
            self.crop_x = np.random.randint(0, 375-self.crop_width, size=(len(self)))
            self.crop_y = np.random.randint(0, 1242-self.crop_height, size=(len(self)))

    def __len__(self):
        if self.crops_per_image is not None:
            return len(self.file_list) * self.crops_per_image
        else:
            return len(self.file_list)

    def __getitem__(self, index):
        if self.crops_per_image is not None:
            file_index = int(index / self.crops_per_image)
        else:
            file_index = index
        left_path = self.root / "image_2" / self.file_list[file_index]
        right_path = self.root / "image_3" / self.file_list[file_index]
        disp_path = self.root / "disp_occ_0" / self.file_list[file_index]
        dxy_path = (
            self.root / "slant_window" / self.file_list[file_index].with_suffix(".npy")
        )

        shp = cv2.imread(str(left_path), cv2.IMREAD_COLOR).shape    
        if self.crops_per_image is not None:
            # crop_index = (index % self.crops_per_image)
            (dx, dy) = (self.crop_width, self.crop_height)
            (x, y) = (self.crop_x[index], self.crop_y[index])
        else:
            (dx, dy) = (shp[0], shp[1])
            (x, y) = (0, 0)

        data = {
            "left": np2torch(cv2.imread(str(left_path), cv2.IMREAD_COLOR)[x:x+dx,y:y+dy,:], bgr=True),
            "right": np2torch(cv2.imread(str(right_path), cv2.IMREAD_COLOR)[x:x+dx,y:y+dy,:], bgr=True),
            "disp": np2torch(
                cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED)[x:x+dx,y:y+dy].astype(np.float32)
                / 256
            ),
            "dxy": np2torch(np.load(dxy_path)[:,x:x+dx,y:y+dy], t=False),
        }
        if self.crop_size is not None:
            data = crop_and_pad(data, self.crop_size, self.training)
        if self.training and self.augmentation:
            data = augmentation(data, self.training)
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
