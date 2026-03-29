import argparse
from parampacmap import util_scripts
from parampacmap.utils import utils
from parampacmap.training import main as train
from parampacmap.initial_embedding import main as init_embed

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--initonly", action="store_true")
    return parser.parse_args()


def get_configs(batch_config: dict):
    is_batch_keys = batch_config['is_batch']
    configs = []

    # Assert the length of the keys are the same
    key_length = -1
    for key in is_batch_keys:
        assert(type(batch_config[key]) == list)
        assert(key_length == -1 or key_length == len(batch_config[key]))
        key_length = len(batch_config[key])
    config_base = batch_config.copy()
    for key in is_batch_keys:
        config_base.pop(key)
    for i in range(key_length):
        config = config_base.copy()
        for key in is_batch_keys:
            config[key] = batch_config[key][i]
        config = utils.impute_default(config, utils.DEFAULT_CONFIG)
        configs.append(config)
    return configs


def main(args):
    batch_config = util_scripts.read_yaml(args.config)
    # We separate the config space into two different sets,
    # the base config and the variable config.
    configs = get_configs(batch_config)
    for config in configs:
        if args.initonly:
            init_embed(config)
        else:
            train(config)


if __name__ == "__main__":
    args = get_args()
    main(args)
