from aux_functions import *
from configs.read_cfg import read_cfg
import importlib

# TF Debug message suppressed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    # Read the config file
    cfg = read_cfg(config_filename='configs/config.cfg', verbose=True)
    algorithm = importlib.import_module('algorithms.'+cfg.algorithm)
    name = 'algorithm.' + cfg.algorithm + '(cfg)'
    eval(name)





