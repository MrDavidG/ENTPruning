# encoding: utf-8
import configparser
import json


def get_cfg(dataset_name, file_cfg_model, file_cfg_global='global.cfg'):
    cfg_global = load_cfg('../config/' + file_cfg_global)

    cfg_path = load_cfg('/path.cfg')

    for section in cfg_path.sections():
        for option in cfg_path[section]:
            if section not in cfg_global.sections():
                cfg_global.add_section(section)
            cfg_global.set(section, option, cfg_path[section][option])

    # Get model cfg
    cfg_model = load_cfg('../config/' + file_cfg_model)

    for section in cfg_model.sections():
        for option in cfg_model.options(section):
            if section not in cfg_global.sections():
                cfg_global.add_section(section)
            cfg_global.set(section, option, cfg_model[section][option])

    # Get data cfg
    cfg_data = load_cfg('../config/data.cfg')

    cfg_global.add_section('data')

    cfg_global.set('data', 'dataset_name', dataset_name)
    for option in cfg_data.options(dataset_name):
        cfg_global.set('data', option, cfg_data[dataset_name][option])

    # reconstruct the model dimensions
    cfg_global['structure']['layers'] = str(
        [cfg_data[dataset_name].getint('dimension')] + json.loads(cfg_global['structure']['layers'])[1:-1] + [
            cfg_data[dataset_name].getint('n_classes')])

    return cfg_global


def load_cfg(path_cfg, localize=False, task_name=None):
    """
    :param path_cfg:
    :param localize:
    :return:
    """
    cfg = configparser.ConfigParser()
    cfg.read(path_cfg)
    if localize:
        cfg_path = configparser.ConfigParser()
        cfg_path.read('/path.cfg')

        cfg.set('path', 'path_dataset', cfg_path['path']['path_dataset'] + task_name + '/')
        cfg.set('path', 'path_save', cfg_path['path']['path_save'])
    return cfg
