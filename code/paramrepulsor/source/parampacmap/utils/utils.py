import yaml
import os

DEFAULT_CONFIG = {
    "datapca": True,
    "datascale": False,
    "datareshape": None,
    "output_dims": 2,
    "n_neighbors": 10,
    "n_FP": 20,
    "n_MN": 5,
    "pacmap_scale": False,
    "distance": "euclidean",
    "optim_type": "Adam",
    "lr": 1e-3,
    "lr_schedule": None,
    "name": "lightning_logs",
    "output_path": None,
    "version": None,
    "profiler": None,
    "use_negative_sampling": False
}

DEFAULT_MODEL_DICT = {
    "backbone": "ANN",
    "layer_size": [100, 100, 100]
}


def read_yaml(config:str):
    with open(config, 'r') as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise exc
    return config_dict


def makedir(dirname:str):
    if not os.path.exists(dirname):
        try:
            os.mkdir(dirname)
        except FileExistsError:
            pass


def impute_default(config: dict, default_config: dict):
    '''
    Impute the training config with a set of default values.
    '''
    for key in default_config.keys():
        if key not in config:
            config[key] = default_config[key]
    
    return config
