from os import close
from aux_functions import *
from configs.read_cfg import read_cfg
import importlib, json
from unreal_envs.initial_positions import *
# from aux_functions import *
# TF Debug message suppressed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



def generate_json(cfg):
    flag  = True
    path = os.path.expanduser('~\Documents\Airsim')
    if not os.path.exists(path):
        os.makedirs(path)

    filename = path + '\settings.json'

    data = {}


    data['SettingsVersion'] = 1.2
    data['LocalHostIp'] = cfg.ip_address
    data['ApiServerPort'] = cfg.api_port

    data['SimMode'] = cfg.SimMode
    data['ClockSpeed'] = cfg.ClockSpeed
    data["ViewMode"]= "NoDisplay"
    PawnPaths = {}
    PawnPaths["DefaultQuadrotor"] = {}
    PawnPaths["DefaultQuadrotor"]['PawnBP'] = ''' Class'/AirSim/Blueprints/BP_''' + cfg.drone + '''.BP_''' + cfg.drone + '''_C' '''
    data['PawnPaths']=PawnPaths

    # Define agents:
    _, reset_array_raw, _, _ = initial_positions(cfg.env_name, num_agents=cfg.num_agents)
    Vehicles = {}
    if len(reset_array_raw) < cfg.num_agents:
        print("Error: Either reduce the number of agents or add more initial positions")
        flag = False
    else:
        for agents in range(cfg.num_agents):
            name_agent = "drone" + str(agents)
            agent_position = reset_array_raw[name_agent].pop(0)
            Vehicles[name_agent] = {}
            Vehicles[name_agent]["VehicleType"] = "SimpleFlight"
            Vehicles[name_agent]["X"] = agent_position[0]
            Vehicles[name_agent]["Y"] = agent_position[1]
            # Vehicles[name_agent]["Z"] = agent_position[2]
            Vehicles[name_agent]["Z"] = 0
            Vehicles[name_agent]["Yaw"] = agent_position[3]
        data["Vehicles"] = Vehicles

    CameraDefaults = {}
    CameraDefaults['CaptureSettings']=[]
    # CaptureSettings=[]

    camera = {}
    camera['ImageType'] = 0
    camera['Width'] = cfg.width
    camera['Height'] = cfg.height
    camera['FOV_Degrees'] = cfg.fov_degrees

    CameraDefaults['CaptureSettings'].append(camera)

    camera = {}
    camera['ImageType'] = 3
    camera['Width'] = cfg.width
    camera['Height'] = cfg.height
    camera['FOV_Degrees'] = cfg.fov_degrees

    CameraDefaults['CaptureSettings'].append(camera)

    data['CameraDefaults'] = CameraDefaults
    with open(filename, 'w') as outfile:
        json.dump(data, outfile, indent=4)

    return flag

def getObservation(client : airsim.MultirotorClient):
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("1", airsim.ImageType.Segmentation, False, False),
            airsim.ImageRequest("2", airsim.ImageType.DepthPlanner, True, False)
        ])
        
        rgbImg = responses[0]
        segImg = responses[1]
        depthImg = responses[2]
        depth = np.array(depthImg.image_data_float).reshape(depthImg.height, depthImg.width)

        depth = np.clip(depth, 0, 50)/50
        
        depth = 1 - depth

        seg1d = np.fromstring(segImg.image_data_uint8, dtype=np.uint8)
        seg = seg1d.reshape(segImg.height, segImg.width, 3)

        rgb1d = np.fromstring(rgbImg.image_data_uint8, dtype=np.uint8)
        rgb = rgb1d.reshape(rgbImg.height, rgbImg.width, 3)

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        observation = {}
        observation['droneState'] = client.getMultirotorState()
        observation["collision"] = client.simGetCollisionInfo().has_collided

        

        observation['depth'] = depth
        observation['fpv'] = cv2.resize(rgb, (256, 256))

        position = observation['droneState'].kinematics_estimated.position


        observation['quadLocation'] =  [
            position.x_val,
            position.y_val,
            position.z_val
        ]

        
        
        return observation
if __name__ == '__main__':
    # Read the config file
    cfg = read_cfg(config_filename='configs/config.cfg', verbose=True)
    cfg.num_agents=1
    can_proceed = generate_json(cfg)
    name = 'drone0'
    # Check if NVIDIA GPU is available
    try:
        nvidia_smi.nvmlInit()
        cfg.NVIDIA_GPU = True
    except:
        cfg.NVIDIA_GPU = False

    if can_proceed:
        # Start the environment
        try:
            env_process, env_folder = start_environment(env_name=cfg.env_name)
            client, old_posit, initZ = connect_drone()
            val = 1
            while True:
                # client.moveByVelocityZAsync(0.01, 0, 0, 0.01)
                # obs = getObservation(client)
                image = get_MonocularImageRGB(client, name)
                cv2.imshow('a', image)
                k  = cv2.waitKey(1)
                # time.sleep(0.5)
                if k == ord('o'):
                    close_env(env_process)
                    exit()
                elif k == ord('i'):
                    client.moveByVelocityAsync(0.0, 0, val, 1)
                elif k == ord('j'):
                    client.moveByVelocityAsync(0.0, 0, -val, 1)
                elif k == ord('w'):
                    client.moveByVelocityAsync(val, 0, 0.0, 1)
                elif k == ord('s'):
                    client.moveByVelocityAsync(-val, 0, 0.0,  1)
                elif k == ord('a'):
                    client.moveByVelocityAsync(0.0, val, 0.0, 1)
                elif k == ord('d'):
                    client.moveByVelocityAsync(0.0, -val, 0.0,  1)
                elif k == ord('e'):
                    # client.rotateToYawAsync(90.,3e+38,1)
                    # client.rotateToYawAsync(45, timeout_sec=3e+38, margin=5)
                    # client.rotateByYawRateAsync(20, 1)#, vehicle_name='')
                    from airsim.types import YawMode
                    ya = YawMode()
                    ya.is_rate = False
                    ya.yaw_or_rate=90
                    client.moveByVelocityAsync(0.0, 0.0, 0.0,  3, drivetrain=1,yaw_mode=ya)

        except Exception as e:

            print(e)
            close_env(env_process)
            print("Closed environment")

