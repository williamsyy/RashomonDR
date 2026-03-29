'''Main Training script of parametric pacmap.
'''
import argparse
import os
import time
import pickle as pkl

import torch
import torch.utils.data
import numpy as np
import pytorch_lightning.callbacks as plcall
import pytorch_lightning.trainer as pltrain
import pytorch_lightning.loggers as pllog
from sklearn import preprocessing
import yaml

from parampacmap.models import module, dataset, pl_module
from parampacmap.utils import data, utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    return args


def get_config(args):
    config = args.config
    config = utils.read_yaml(config)
    config = utils.impute_default(config, utils.DEFAULT_CONFIG)
    return config


def convert_pairs(pair_neighbors, pair_FP, pair_MN, N):
    pair_neighbors = pair_neighbors[:, 1].reshape((N, -1))
    pair_FP = pair_FP[:, 1].reshape((N, -1))
    pair_MN = pair_MN[:, 1].reshape((N, -1))
    return pair_neighbors, pair_FP, pair_MN


def get_loaders(config):
    '''
    Prepare the dataloader for training based on the dataset name and desired shape.
    '''
    # Load the data
    X, y = data.data_prep(dataset=config['dataset'],
                          size=config['datasize'],
                          dim=config['datadim'],
                          pca=config['datapca'],)
    input_dims = X.shape[1]
    # Construct the pairs
    n_neighbors, n_FP, n_MN = config['n_neighbors'], config['n_FP'], config['n_MN']
    pair_neighbors, pair_MN, pair_FP, _ = data.generate_pair(
        X, n_neighbors=n_neighbors, n_MN=n_MN, n_FP=n_FP,
        distance=config['distance'], verbose=False
    )
    if config['datascale'] == 1:
        scaler = preprocessing.StandardScaler()
        X = scaler.fit_transform(X)
    elif config['datascale'] == 2:
        scaler = preprocessing.MinMaxScaler()
        X = scaler.fit_transform(X)
    nn_pairs, fp_pairs, mn_pairs = convert_pairs(pair_neighbors, pair_FP, pair_MN, X.shape[0])
    assert isinstance(config['use_negative_sampling'], bool)
    if config['use_negative_sampling']:
        train_set = dataset.NegativeSamplingDataset(
            data=X,
            nn_pairs=nn_pairs,
            fp_pairs=fp_pairs,
            mn_pairs=mn_pairs,
            reshape=config['datareshape']
        )
    else:
        train_set = dataset.PaCMAPDataset(
            data=X,
            nn_pairs=nn_pairs,
            fp_pairs=fp_pairs,
            mn_pairs=mn_pairs,
            reshape=config['datareshape']
        )
    train_loader = torch.utils.data.DataLoader(dataset=train_set, 
                                               batch_size=config['batch_size'],
                                               shuffle=True,
                                               drop_last=False,
                                               pin_memory=True,
                                               num_workers=config['dlworker'],
                                               persistent_workers=True)
    val_set = dataset.PaCMAPDataset(data=X,
                                    nn_pairs=nn_pairs,
                                    fp_pairs=fp_pairs,
                                    mn_pairs=mn_pairs,
                                    reshape=config['datareshape'])
    val_loader = torch.utils.data.DataLoader(dataset=val_set, 
                                             batch_size=config['batch_size_val'],
                                             shuffle=True,
                                             drop_last=False,
                                             pin_memory=True,
                                             num_workers=config['dlworker'],
                                             persistent_workers=True)
    test_set = dataset.TensorDataset(data=X, reshape=config['datareshape'])
    test_loader = torch.utils.data.DataLoader(dataset=test_set, 
                                             batch_size=config['batch_size_inference'],
                                             shuffle=False,
                                             drop_last=False,
                                             pin_memory=True,
                                             num_workers=config['dlworker'],
                                             persistent_workers=True)
    return train_loader, val_loader, test_loader, input_dims


def get_model(config, input_dims):
    '''Prepare the lightning module for model training.
    '''
    model = module.ParamPaCMAP(input_dims=input_dims,
                               output_dims=config['output_dims'],
                               model_dict=config['model_dict'])
    loss = module.PaCMAPLoss(weight=config['loss_weight'])
    model = pl_module.PaCMAPTraining(model=model, loss=loss,
                                  weight=config['loss_weight'], 
                                  pacmap_scale=config['pacmap_scale'],
                                  dataset=config['dataset'],
                                  optim_type=config['optim_type'],
                                  lr=config['lr'],
                                  lr_schedule=config['lr_schedule'])
    return model


def get_logger(config):
    logger = pllog.CSVLogger(
        save_dir="/home/users/hh219/FastPaCMAP/ParamPaCMAPdev/logs",
        name=config['name'],
        version=config['version']
    )
    return logger


def get_logger_dir(logger: pllog.Logger):
    save_dir = str(logger.save_dir)
    name = str(logger.name)
    version = "version_" + str(logger.version)
    return os.path.join(save_dir, name, version)


def get_logger_basedir(logger: pllog.Logger):
    save_dir = str(logger.save_dir)
    name = str(logger.name)
    return os.path.join(save_dir, name)


def get_callbacks(config):
    '''
    Configure the callbacks
    '''
    checkpoint_callback = plcall.ModelCheckpoint(
        save_top_k=-1,
        monitor="loss",
        every_n_epochs=20,
    )
    return checkpoint_callback


def main(config):
    train_loader, val_loader, test_loader, input_dims = get_loaders(config)
    callbacks = get_callbacks(config)
    logger = get_logger(config)
    torch.set_float32_matmul_precision("medium")

    trainer = pltrain.Trainer(max_epochs=config['epoch'], 
                              callbacks=callbacks,
                              logger=logger,
                              enable_progress_bar=False,
                              profiler=config["profiler"],
                              devices=1)  # Ensure single gpu is used.
    model = get_model(config, input_dims)
    logger_dir = get_logger_dir(trainer.logger)

    # Immediately generate an output and save
    results = trainer.predict(model=model, dataloaders=test_loader)
    results = torch.concatenate(results)
    results = results.cpu().numpy()

    np.save(os.path.join(logger_dir, "output_initial.npy"), results)

    start = time.perf_counter()
    # Train the model
    trainer.fit(model=model, train_dataloaders=train_loader)

    end = time.perf_counter()
    time_used = end - start  # In seconds
    time_name = os.path.join(get_logger_basedir(trainer.logger), "time.pkl")
    if os.path.exists(time_name):
        with open(time_name, "rb") as fp:
            time_dict = pkl.load(fp)
    else:
        time_dict = {}
    time_dict[config['dataset']] = time_used
    print(f"Finished in {time_used:.3f} seconds.")
    with open(time_name, "wb") as fp:
        pkl.dump(time_dict, fp)

    # Create final embedding and save
    results = trainer.predict(model=model, dataloaders=test_loader)
    results = torch.concatenate(results)
    results = results.cpu().numpy()

    if config['output_path'] is not None:
        np.save(config['output_path'], results)

    np.save(os.path.join(logger_dir, "output.npy"), results)
    # Save the config file into the log directory
    yaml.dump(config, open(
        os.path.join(logger_dir, "config.yaml"), "w"
    ))


if __name__ == "__main__":
    args = get_args()
    config = get_config(args)
    main(config)
