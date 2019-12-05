# Author: Aqeel Anwar(ICSRL)
# Created: 9/20/2019, 12:43 PM
# Email: aqeel.anwar@gatech.edu

import configparser as cp
from dotmap import DotMap

def read_cfg(config_filename = 'configs/main.cfg', verbose = False):
    # Load from config file
    cfg = DotMap()

    config = cp.ConfigParser()
    config.read(config_filename)

    cfg.run_name = config.get('general_params', 'run_name')
    if str(config.get('general_params', 'custom_load')) =='True':
        cfg.custom_load = True
    else:
        cfg.custom_load = False
    cfg.custom_load_path = str(config.get('general_params', 'custom_load_path'))
    cfg.env_type = config.get('general_params', 'env_type')
    cfg.env_name = config.get('general_params', 'env_name')

    # [Simulation Parameters]
    if str(config.get('simulation_params', 'load_data')) =='True':
        cfg.load_data = True
    else:
        cfg.load_data = False
    cfg.load_data_path = str(config.get('simulation_params', 'load_data_path'))
    cfg.ip_address = str(config.get('simulation_params', 'ip_address'))

    # [RL Parameters]
    cfg.input_size = int(config.get('RL_params', 'input_size').split(',')[0])
    cfg.num_actions = int(config.get('RL_params', 'num_actions').split(',')[0])
    cfg.train_type = config.get('RL_params', 'train_type')
    cfg.wait_before_train = int(config.get('RL_params', 'wait_before_train').split(',')[0])
    cfg.max_iters = int(config.get('RL_params', 'max_iters').split(',')[0])
    cfg.buffer_len = int(config.get('RL_params', 'buffer_len').split(',')[0])
    cfg.batch_size = int(config.get('RL_params', 'batch_size').split(',')[0])
    cfg.epsilon_saturation = int(config.get('RL_params', 'epsilon_saturation').split(',')[0])
    cfg.crash_thresh = float(config.get('RL_params', 'crash_thresh').split(',')[0])
    cfg.gamma = float(config.get('RL_params', 'gamma').split(',')[0])
    cfg.dropout_rate = float(config.get('RL_params', 'dropout_rate').split(',')[0])
    cfg.lr = float(config.get('RL_params', 'learning_rate').split(',')[0])
    cfg.switch_env_steps = int(config.get('RL_params', 'switch_env_steps').split(',')[0])
    cfg.epsilon_model = config.get('RL_params', 'epsilon_model')
    cfg.Q_clip = bool(config.get('RL_params', 'Q_clip'))
    cfg.train_interval = int(config.get('RL_params', 'train_interval').split(',')[0])
    cfg.update_target_interval = int(config.get('RL_params', 'update_target_interval').split(',')[0])


    if verbose:
        print('------------------------------ Config File ------------------------------')
        for param in cfg:
            spaces = ' '*(30-len(param))
            print(param+':'+spaces + str(cfg[param]))

    # print('-------------------------------------------------------------------------')
    print()
    return cfg

