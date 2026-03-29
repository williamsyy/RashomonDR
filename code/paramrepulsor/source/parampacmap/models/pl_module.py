import torch
import pytorch_lightning as pl
from torch import optim



class PaCMAPTraining(pl.LightningModule):
    """PyTorch Lightning Training Module for PaCMAP.
    """
    def __init__(self, 
                 model, 
                 loss, 
                 weight=[1, 0.3, 0.7], 
                 pacmap_scale=None,
                 dataset=None,
                 optim_type="Adam",
                 lr=1e-3,
                 lr_schedule=None) -> None:
        super().__init__()
        self.model = model
        self.loss = loss
        self.pacmap_scale = pacmap_scale
        self.weight = weight
        self.dataset = dataset
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.optim_type = optim_type
        self.save_hyperparameters("weight",
                                  "pacmap_scale",
                                  "lr",
                                  "optim_type",
                                  self.model.model_dict)


    def tune_weight(self, epoch) -> None:
        '''Tune the weight between different losses.
        The format is NN:FP:MN.
        '''
        if self.pacmap_scale is None or self.pacmap_scale is False:
            weight = self.weight
        elif self.pacmap_scale == 0:
            # Original PaCMAP Scale
            if epoch < 100:
                w_mn = 10 * (100 - epoch) + 0.03 * epoch
                w_nn = 2
                w_fp = 1
            elif epoch < 200:
                w_mn = 3
                w_nn = 3
                w_fp = 1
            else:
                w_mn = 0
                w_nn = 1
                w_fp = 1
            weight = [w_nn, w_fp, w_mn]
        elif self.pacmap_scale == 1:
            if epoch < 10:
                w_mn = 100 * (10 - epoch)
                w_nn = 2
                w_fp = 1
            else:
                w_mn = 0
                w_nn = 2
                w_fp = 1 + 0.1 * (epoch - 10)
            weight = [w_nn, w_fp, w_mn]
        elif self.pacmap_scale == 2:
            if epoch < 10:
                w_mn = 100 * (10 - epoch)
                w_nn = 2
                w_fp = 1
            else:
                w_mn = 0
                w_nn = 2
                w_fp = 2
            weight = [w_nn, w_fp, w_mn]
        elif self.pacmap_scale == 3:
            if epoch < 10:
                w_mn = 100 * (10 - epoch)
                w_nn = 10
                w_fp = 5
            else:
                w_mn = 0
                w_nn = 10
                w_fp = 10
            weight = [w_nn, w_fp, w_mn]
        elif self.pacmap_scale == 4:
            if epoch < 10:
                w_mn = 100 * (10 - epoch) + 0.3 * epoch
                w_nn = 2
                w_fp = 1
            elif epoch < 20:
                w_mn = 3
                w_nn = 3
                w_fp = 1
            else:
                w_mn = 0
                w_nn = 1
                w_fp = 1
            weight = [w_nn, w_fp, w_mn]
        else:
            raise ValueError("Unsupported style")

        self.loss.update_weight(weight)

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        basis, nn_pairs, fp_pairs, mn_pairs = batch

        # The pairs are under the format (i, num_pairs, ...)
        num_items = basis.size(0)
        num_NN = nn_pairs.size(1)
        num_FP = fp_pairs.size(1)
        num_mn = mn_pairs.size(1)
        merged_shape = (-1,) + nn_pairs.shape[2:]
        nn_pairs = nn_pairs.view(*merged_shape)
        fp_pairs = fp_pairs.view(*merged_shape)
        mn_pairs = mn_pairs.view(*merged_shape)

        # Use the model to perform forward
        basis = self.model(basis)
        nn_pairs = self.model(nn_pairs)
        fp_pairs = self.model(fp_pairs)
        mn_pairs = self.model(mn_pairs)
        basis = torch.unsqueeze(basis, 1)
        nn_pairs = nn_pairs.view(num_items, num_NN, nn_pairs.shape[1])
        fp_pairs = fp_pairs.view(num_items, num_FP, fp_pairs.shape[1])
        mn_pairs = mn_pairs.view(num_items, num_mn, mn_pairs.shape[1])

        # Calculate the loss
        self.tune_weight(epoch=self.current_epoch)
        loss = self.loss(basis, nn_pairs, fp_pairs, mn_pairs)
        self.log("loss", loss)
        return loss


    def validation_step(self, batch, batch_idx):
        basis, nn_pairs, fp_pairs, mn_pairs = batch
        # The pairs are under the format (i, num_pairs, -1)
        # basis = basis.view(basis.size(0), -1)
        # nn_pairs = nn_pairs.view(nn_pairs.size(0), nn_pairs.size(1), -1)
        # fp_pairs = fp_pairs.view(fp_pairs.size(0), fp_pairs.size(1), -1)
        # mn_pairs = mn_pairs.view(mn_pairs.size(0), mn_pairs.size(1), -1)

        # Use the model to perform forward
        basis = self.model(basis)
        nn_pairs = self.model(nn_pairs)
        fp_pairs = self.model(fp_pairs)
        mn_pairs = self.model(mn_pairs)
        basis = torch.unsqueeze(basis, 1)

        # Calculate the loss
        loss = self.loss(basis, nn_pairs, fp_pairs, mn_pairs)
        self.log_dict({'val_loss': loss})


    def configure_optimizers(self):
        if self.optim_type == "Adam":
            optimizer = optim.Adam(self.model.parameters(),
                                   lr=self.lr)
        elif self.optim_type == "SGD":
            optimizer = optim.SGD(self.model.parameters(), 
                                  lr=self.lr, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optim_type}")
        if self.lr_schedule is not None:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                            mode="min",
                                                            factor=0.25,
                                                            patience=2,
                                                            threshold=1e-2,
                                                            )
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "loss",
            }
        else:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
