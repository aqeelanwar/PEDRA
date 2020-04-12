# Author: Aqeel Anwar(ICSRL)
# Created: 9/20/2019, 12:43 PM
# Email: aqeel.anwar@gatech.edu

from configparser import ConfigParser
from dotmap import DotMap


def ConvertIfStringIsInt(input_string):
    try:
        float(input_string)

        try:
            if int(input_string) == float(input_string):
                return int(input_string)
            else:
                return float(input_string)
        except ValueError:
            return float(input_string)

    except ValueError:
        true_array = ['True', 'TRUE', 'true', 'Yes', 'YES', 'yes']
        false_array = ['False', 'FALSE', 'false', 'No', 'NO', 'no']
        if input_string in true_array:
            input_string = True
        elif input_string in false_array:
            input_string = False

        return input_string


def read_cfg(config_filename='configs/main.cfg', verbose=False):
    parser = ConfigParser()
    parser.optionxform = str
    parser.read(config_filename)
    cfg = DotMap()

    if verbose:
        hyphens = '-' * int((80 - len(config_filename))/2)
        print(hyphens + ' ' + config_filename + ' ' + hyphens)

    for section_name in parser.sections():
        if verbose:
            print('[' + section_name + ']')
        for name, value in parser.items(section_name):
            value = ConvertIfStringIsInt(value)
            cfg[name] = value
            spaces = ' ' * (30 - len(name))
            if verbose:
                print(name + ':' + spaces + str(cfg[name]))

    return cfg

def update_algorithm_cfg(algorithm_cfg, cfg):
    if algorithm_cfg.distributed_algo=='GlobalLearningGlobalUpdate-MA':
        algorithm_cfg.wait_before_train = algorithm_cfg.wait_before_train*cfg.num_agents
        algorithm_cfg.max_iters = algorithm_cfg.max_iters * cfg.num_agents
        algorithm_cfg.buffer_len = algorithm_cfg.buffer_len * cfg.num_agents
        algorithm_cfg.epsilon_saturation = algorithm_cfg.epsilon_saturation*cfg.num_agents
        algorithm_cfg.update_target_interval = algorithm_cfg.update_target_interval*cfg.num_agents

    return algorithm_cfg