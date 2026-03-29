import yaml

def read_yaml(config:str):
    with open(config, 'r') as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise exc
    return config_dict
