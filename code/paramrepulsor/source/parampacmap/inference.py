import argparse
import os
import re

import numpy as np
import torch
import pytorch_lightning as pl
import pytorch_lightning.trainer as pltrain
import yaml
from sklearn import preprocessing
import tqdm

from .models import module, dataset, pl_module
from .utils import data, utils

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expdir", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def ckpt_key(name):
    """Find out the epoch number of a ckpt name. Will be used as key for sort."""
    pattern = r"epoch=([0-9]*)-step=([0-9]*).ckpt"
    match = re.search(pattern, name)
    return int(match[1])


def get_ckpts(expdir: str):
    ckpt_dir = os.path.join(expdir, "checkpoints")
    ckpt_names = os.listdir(ckpt_dir)
    ckpt_names.sort(key=ckpt_key)
    return ckpt_dir, ckpt_names


def get_config(expdir: str):
    config_path = os.path.join(expdir, "config.yaml")
    return yaml.load(open(config_path, "r"), Loader=yaml.Loader)


def main(args):
    # Obtain config
    expdir = args.expdir
    if args.config is None:
        config = get_config(expdir)
    else:
        config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    input_dims = config["datadim"]

    # Construct dataset
    X, y = data.data_prep(
        dataset=config["dataset"],
        size=config["datasize"],
        dim=input_dims,
        pca=config["datapca"],
    )
    if config['datascale']:
        scaler = preprocessing.StandardScaler()
        X = scaler.fit_transform(X)
    val_set = dataset.TensorDataset(data=X)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=4096,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=2
    )

    # Construct base model
    model_dict = config["model_dict"]
    input_dims = X.shape[1]
    if input_dims > 100 and config["datapca"]:
        input_dims = 100
    base_model = module.ParamPaCMAP(input_dims=input_dims, model_dict=model_dict)
    loss = module.PaCMAPLoss([1, 0.3, 0.7])  # Placeholder only

    # Load the checkpoint
    ckpt_base, ckpt_names = get_ckpts(expdir)
    for ckpt_name in tqdm.tqdm(ckpt_names):
        ckpt = os.path.join(ckpt_base, ckpt_name)
        model = pl_module.PaCMAPTraining(base_model, loss)
        model.load_from_checkpoint(ckpt,
                                   model=base_model,
                                   loss=loss)

        # Perform Inference
        trainer = pltrain.Trainer(max_epochs=1)
        results = trainer.predict(model=model, dataloaders=val_loader)
        results = torch.concatenate(results)
        results = results.cpu().numpy()
        outputdir = os.path.join(expdir, "outputs")
        os.makedirs(outputdir, exist_ok=True)
        np.save(os.path.join(outputdir, f"epoch_{ckpt_key(ckpt_name)}.npy"),
                results)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    args = get_args()
    main(args)
