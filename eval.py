import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from models import build_model
from dataset import build_dataset
from metrics.epe import EPEMetric
from metrics.rate import RateMetric
from torchmetrics import MetricCollection


class EvalModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = build_model(self.hparams)
        self.max_disp = self.hparams.max_disp
        self.metric = MetricCollection(
            {
                "epe": EPEMetric(),
                "rate_1e-1": RateMetric(0.1),
                "rate_1": RateMetric(1.0),
                "rate_3": RateMetric(3.0),
            }
        )

    def forward(self, left, right):
        left = left * 2 - 1
        right = right * 2 - 1
        return self.model(left, right)

    def test_step(self, batch, batch_idx):
        pred = self(batch["left"], batch["right"])
        if "crop2roi" in batch:
            c2r = batch["crop2roi"]
            pred_disp  = pred[ "disp"][:, :, c2r[0]:c2r[1], c2r[2]:c2r[3]]
            batch_disp = batch["disp"][:, :, c2r[0]:c2r[1], c2r[2]:c2r[3]]
        else:
            pred_disp = pred["disp"]
            batch_disp = batch["disp"]
        mask = (batch_disp < self.max_disp) & (batch_disp > 1e-3)
        self.metric(pred_disp, batch_disp, mask)
        return

    def test_epoch_end(self, outputs):
        self.log_dict(self.metric.compute())
        return

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--max_disp", type=int, default=192)
    parser.add_argument("--data_type_val", type=str, nargs="+")
    parser.add_argument("--data_root_val", type=str, nargs="+")
    parser.add_argument("--data_list_val", type=str, nargs="+")
    parser.add_argument("--data_size_val", type=int, nargs=2, default=None)
    parser.add_argument("--data_augmentation", type=int, default=0)
    parser.add_argument("--roi_w_pad", type=int, default=0)
    parser.add_argument("--roi_h_pad", type=int, default=0)

    args = parser.parse_args()

    model = EvalModel(**vars(args)).eval()
    ckpt = torch.load(args.ckpt, map_location='cuda:0')
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.model.load_state_dict(ckpt)

    dataset = build_dataset(args, training=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=2,
    )

    trainer = pl.Trainer(
        gpus=-1,
        strategy="ddp",
        logger=False,
        checkpoint_callback=False,
    )
    trainer.test(model, loader)
