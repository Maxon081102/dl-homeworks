import torch
import pytorch_lightning as L

from torch import nn, Tensor
from torch.nn import functional as F
from typing import Any, Callable, Optional, Union
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics.detection import MeanAveragePrecision
from torch.optim.optimizer import Optimizer

class PeopleArtModule(L.LightningModule):
    def __init__(self, model: nn.Module, learning_rate: float) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.val_metrics = MeanAveragePrecision(iou_type="bbox")
        
        self.save_hyperparameters()

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        x, y = batch
        res = self.model(x, y)
        loss1 = res["bbox_regression"]
        loss2 = res["classification"]

        a = 0.9

        loss = a * loss1 + (1 - a) * loss2
        
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> STEP_OUTPUT | None:
        x, y = batch
        res = self.model(x, y)
        loss = [elem["scores"].mean().item() for elem in res]
        loss = torch.Tensor(loss).mean()
        self.val_metrics.update(res, y)
        self.log("val_loss", loss)
        self.log("val_map", self.val_metrics.compute()["map"])

        return {
            "loss": loss
            
        }

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[5, 10, 15]
            )
        }