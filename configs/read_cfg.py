# Author: Aqeel Anwar(ICSRL)
# Created: 9/20/2019, 12:43 PM
# Email: aqeel.anwar@gatech.edu

import configparser as cp
from dotmap import DotMap
import os, json

def read_env_cfg(config_filename = 'configs/main.cfg'):
    # Load from config file
    cfg = DotMap()

    config = cp.ConfigParser()
    config.read(config_filename)

    cfg.run_name = config.get('general_params', 'env_name')
    cfg.floorplan = str(config.get('general_params', 'floorplan'))
    cfg.o_x = float(config.get('general_params', 'o_x').split(',')[0])
    cfg.o_y = float(config.get('general_params', 'o_y').split(',')[0])
    cfg.alpha = float(config.get('general_params', 'alpha').split(',')[0])
    cfg.ceiling_z = float(config.get('general_params', 'ceiling_z').split(',')[0])
    cfg.floor_z = float(config.get('general_params', 'floor_z').split(',')[0])
    cfg.player_start_z = float(config.get('general_params', 'player_start_z').split(',')[0])

    return cfg

# def generate_json(cfg):
#     path = os.path.expanduser('~\Documents\Airsim')
#     if not os.path.exists(path):
#         os.makedirs(path)
#
#     filename = path + '\settings.json'
#
#     data = {}
#     data['SettingsVersion'] = 1.2
#     data['LocalHostIp'] = cfg.ip_address
#     data['SimMode'] = cfg.SimMode
#     data['ClockSpeed'] = cfg.ClockSpeed
#
#     PawnPaths = {}
#     PawnPaths["DefaultQuadrotor"] = {}
#     PawnPaths["DefaultQuadrotor"]['PawnBP'] = ''' Class'/AirSim/Blueprints/BP_''' + cfg.drone + '''.BP_''' + cfg.drone + '''_C' '''
#     data['PawnPaths']=PawnPaths
#
#     CameraDefaults = {}
#     CameraDefaults['CaptureSettings']=[]
#     # CaptureSettings=[]
#
#     camera = {}
#     camera['ImageType'] = 0
#     camera['Width'] = cfg.width
#     camera['Height'] = cfg.height
#     camera['FOV_Degrees'] = cfg.fov_degrees
#
#     CameraDefaults['CaptureSettings'].append(camera)
#
#     camera = {}
#     camera['ImageType'] = 3
#     camera['Width'] = cfg.width
#     camera['Height'] = cfg.height
#     camera['FOV_Degrees'] = cfg.fov_degrees
#
#     CameraDefaults['CaptureSettings'].append(camera)
#
#     data['CameraDefaults']=CameraDefaults
#     with open(filename, 'w') as outfile:
#         json.dump(data, outfile)




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
    cfg.phase = config.get('general_params', 'phase')
    cfg.SimMode = str(config.get('general_params', 'SimMode'))
    cfg.drone = str(config.get('general_params', 'drone'))
    cfg.ClockSpeed = int(config.get('general_params', 'ClockSpeed').split(',')[0])
    cfg.algorithm = str(config.get('general_params', 'algorithm'))
    cfg.ip_address = str(config.get('general_params', 'ip_address'))

    # [Camera_params]
    cfg.width = int(config.get('camera_params', 'width').split(',')[0])
    cfg.height = int(config.get('camera_params', 'height').split(',')[0])
    cfg.fov_degrees = int(config.get('camera_params', 'fov_degrees').split(',')[0])

    if verbose and cfg.phase=='train':
        print('------------------------------ Config File ------------------------------')
        for param in cfg:
            spaces = ' '*(30-len(param))
            print(param+':'+spaces + str(cfg[param]))

    # print('-------------------------------------------------------------------------')
    print()
    return cfg

