import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from models import build_model


class PredictModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(self.hparams)

    def forward(self, left, right):
        left = left * 2 - 1
        right = right * 2 - 1
        return self.model(left, right)


@torch.no_grad()
def predict(model, lp, rp, width, roi, roi_padding, highlight, op):
    left = cv2.imread(str(lp), cv2.IMREAD_COLOR)
    right = cv2.imread(str(rp), cv2.IMREAD_COLOR)

    image2roi = roi
    image = left.shape
    if not any(image2roi):
        image2roi = [0, image[0], 0, image[1]]
    if(roi_padding != -1):
        roi_pad = [ min(roi_padding, image2roi[0]), min(roi_padding, image[0] - image2roi[1]),
                    min(roi_padding, image2roi[2]), min(roi_padding, image[1] - image2roi[3])]
    else:
        roi_pad = [ image2roi[0], image[0] - image2roi[1],
                    image2roi[2], image[1] - image2roi[3]]
    image2crop = [image2roi[0]-roi_pad[0], image2roi[1]+roi_pad[1], image2roi[2]-roi_pad[2], image2roi[3]+roi_pad[3]]
    crop2roi   = [roi_pad[0], roi_pad[0]+image2roi[1]-image2roi[0], roi_pad[2], roi_pad[2]+image2roi[3]-image2roi[2]]

    left  = left[image2crop[0]:image2crop[1], image2crop[2]:image2crop[3], :]
    right = right[image2crop[0]:image2crop[1], image2crop[2]:image2crop[3], :]

    if width is not None and width != left.shape[1]:
        height = int(round(width / left.shape[1] * left.shape[0]))
        left = cv2.resize(
            left,
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )
        right = cv2.resize(
            right,
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )
    left = np2torch(left, bgr=True).cuda().unsqueeze(0)
    right = np2torch(right, bgr=True).cuda().unsqueeze(0)
    pred = model(left, right)

    disp = pred["disp"]
    disp = torch.clip(disp / 192 * 255, 0, 255).long()
    disp = apply_colormap(disp)

    if highlight:
        i = crop2roi[0]
        for j in range(crop2roi[2], crop2roi[3]):
            left[0, 0, i, j], left[0, 1, i, j], left[0, 2, i, j] = 255, 0, 0
            disp[0, 0, i, j], disp[0, 1, i, j], disp[0, 2, i, j] = 255, 0, 0
        i = crop2roi[1]-1
        for j in range(crop2roi[2], crop2roi[3]):
            left[0, 0, i, j], left[0, 1, i, j], left[0, 2, i, j] = 255, 0, 0
            disp[0, 0, i, j], disp[0, 1, i, j], disp[0, 2, i, j] = 255, 0, 0
        j = crop2roi[2]
        for i in range(crop2roi[0], crop2roi[1]):
            left[0, 0, i, j], left[0, 1, i, j], left[0, 2, i, j] = 255, 0, 0
            disp[0, 0, i, j], disp[0, 1, i, j], disp[0, 2, i, j] = 255, 0, 0
        j = crop2roi[3]-1
        for i in range(crop2roi[0], crop2roi[1]):
            left[0, 0, i, j], left[0, 1, i, j], left[0, 2, i, j] = 255, 0, 0
            disp[0, 0, i, j], disp[0, 1, i, j], disp[0, 2, i, j] = 255, 0, 0

    output = [left, disp]
    if "slant" in pred:
        dxy = dxy_colormap(pred["slant"][-1][1])
        output.append(dxy)

    if not highlight:
        output = [img[:, :, crop2roi[0]:crop2roi[1], crop2roi[2]:crop2roi[3]] for img in output]

    # output = np.concatenate(output, axis=0)
    output = torch.cat(output, dim=0)
    torchvision.utils.save_image(output, op, nrow=1)
    return


if __name__ == "__main__":
    import cv2
    import argparse
    import torchvision
    from pathlib import Path

    from dataset.utils import np2torch
    from colormap import apply_colormap, dxy_colormap

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs=2, required=True)
    parser.add_argument("--roi", nargs=4, type=int, default=[0, 0, 0, 0])
    parser.add_argument("--roi_padding", type=int, default=0)
    parser.add_argument("--highlight", type=bool, default=False)
    parser.add_argument("--model", type=str, default="HITNet")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--output", default="./")
    args = parser.parse_args()

    model = PredictModel(**vars(args)).eval()
    ckpt = torch.load(args.ckpt)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.model.load_state_dict(ckpt)
    model.cuda()

    if "*" in args.images[0]:
        lps = list(sorted(Path(".").glob(args.images[0])))
        rps = list(sorted(Path(".").glob(args.images[1])))

        for ids, (lp, rp) in enumerate(zip(lps, rps)):
            op = Path(args.output) / f"{lp.stem}_{ids}.png"
            predict(model, lp, rp, args.width, args.roi, args.roi_padding, args.highlight, op)
            print("output: {}".format(op))
    else:
        lp = Path(args.images[0])
        rp = Path(args.images[1])
        op = Path("./predict_out/{}/{}_{}_{}_{}/pad{}.png".format(lp.stem, *args.roi, args.roi_padding))
        # op = Path(args.output)
        # if op.is_dir():
        #     op = op / lp.name
        predict(model, lp, rp, args.width, args.roi, args.roi_padding, args.highlight, op)
        print("output: {}".format(op))
