import numpy as np
import airsim

# Define the initial positions of teh environments being used.
# The module name is the name of the environment to be used in the config file
# and should be same as the .exe file (and folder) within the unreal_envs folder

def indoor_meta():

    orig_ip = [     #x, y, theta in DEGREES

                    # One - Pyramid
                    [-21593, -1563, -45],  # Player Start
                    [-22059, -2617, -45],
                    [-22800, -3489, 90],

                    # Two - FrogEyes
                    [-15744, -1679, 0],
                    [-15539, -3043, 180],
                    [-13792, -3371, 90],

                    # Three - UpDown
                    [-11221, -3171, 180],
                    [-9962, -3193, 0],
                    [-7464, -4558, 90],

                    # Four - Long
                    [-649, -4287, 180],  # Player Start
                    [-4224, -2601, 180],
                    [1180, -2153, -90],

                    # Five - VanLeer
                    [6400, -4731, 90],  # Player Start
                    [5992, -2736, 180],
                    [8143, -2835, -90],

                    # Six - Complex_Indoor
                    [11320, -2948, 0],
                    [12546, -3415, -180],
                    [10809, -2106, 0],

                    # Seven - Techno
                    [19081, -8867, 0],
                    [17348, -3864, -120],
                    [20895, -4757, 30],

                    # Eight - GT
                    [26042, -4336, 180],
                    [26668, -3070, 0],
                    [27873, -2792, -135]



                ]# x, y, theta
    level_name = [
                    'Pyramid1', 'Pyramid2', 'Pyramid3',
                    'FrogEyes1', 'FrogEyes2', 'FrogEyes3',
                    'UpDown1', 'UpDown2', 'UpDown3',
                    'Long1', 'Long2', 'Long3',
                    'VanLeer1', 'VanLeer2', 'VanLeer3',
                    'ComplexIndoor1', 'ComplexIndoor2', 'ComplexIndoor3',
                    'Techno1', 'Techno2', 'Techno3',
                    'GT1', 'GT2', 'GT3',
                ]
    crash_threshold = 0.07
    initZ = 0
    return orig_ip, level_name, crash_threshold, initZ


# Train complex indoor initial positions
def indoor_complex():
    # The environment can be downloaded from
    # https://drive.google.com/drive/u/2/folders/1u5teth6l4JW2IXAkZAg1CbDGR6zE-v6Z
    orig_ip = [
        [-195, 812, 0],  # Player start
        [-1018, 216, -50],
        [-77, -118, 180],
        [800, -553, 190]
    ]
    level_name = ['Complex1', 'Complex2', 'Complex3', 'Complex4']
    crash_threshold = 0.07
    initZ = 0
    return orig_ip, level_name, crash_threshold, initZ

# Test condo indoor initial positions
def indoor_cloud():
    # The environment can be downloaded from
    # https://drive.google.com/drive/u/2/folders/1u5teth6l4JW2IXAkZAg1CbDGR6zE-v6Z
    orig_ip = [
        [3308, 610, 0],  # Player start
        [2228, 380, 270],
        [1618, -990, 30],
        [618, 610, 0]
    ]
    level_name = ['Condo1', 'Condo2', 'Condo3', 'Condo4']
    crash_threshold = 0.07
    initZ = 0
    return orig_ip, level_name, crash_threshold, initZ

def indoor_gt():
    # The environment can be downloaded from
    # https://drive.google.com/drive/u/2/folders/1u5teth6l4JW2IXAkZAg1CbDGR6zE-v6Z
    orig_ip = [
        [-30, 460, 0],  # Player start
        [640, 900, 270],
        [-130, -1600, 200],
        [-1000, 350, 200]
    ]
    level_name = ['GT1', 'GT2', 'GT3', 'GT4']
    crash_threshold = 0.07
    initZ = 0
    return orig_ip, level_name, crash_threshold, initZ

def indoor_techno():
    # The environment can be downloaded from
    # https://drive.google.com/drive/u/2/folders/1u5teth6l4JW2IXAkZAg1CbDGR6zE-v6Z
    orig_ip = [
        [4289, -7685, 0],  #Player Start
        [3750, -1750, -120],
        [3580, -4770, -70],
        [6220, -2150, -40]
    ]
    level_name = ['Techno1', 'Techno2', 'Techno3', 'Techno4']
    crash_threshold = 0.07
    initZ = 0
    return orig_ip, level_name, crash_threshold, initZ

def indoor_vanleer():
    # The environment can be downloaded from
    # https://drive.google.com/drive/u/2/folders/1u5teth6l4JW2IXAkZAg1CbDGR6zE-v6Z
    orig_ip = [
        [-3100, -4530, 90],  #Player Start
        [-1340, -2240, -90],
        [-3790, -5450, 180],
        [-3980, -1760, -90]
    ]
    level_name = ['VanLeer1', 'VanLeer2', 'VanLeer3', 'VanLeer4']
    crash_threshold = 0.07
    initZ = 0
    return orig_ip, level_name, crash_threshold, initZ

def indoor_long():
    # The environment can be downloaded from
    # https://drive.google.com/drive/u/2/folders/1u5teth6l4JW2IXAkZAg1CbDGR6zE-v6Z
    orig_ip = [
        [-649, -4287, 180],  # Player Start
        [-4224, -2601, 180],
        [1180, -2153, -90],
        [2058, -3184, 50],
        [1644,-1464, 15],
        [-3754, -4302, 0]
    ]
    level_name = ['Long1', 'Long2', 'Long3', 'Long4', 'Long5', 'Long6']
    crash_threshold = 0.07
    initZ = 0
    return orig_ip, level_name, crash_threshold, initZ

def indoor_pyramid():
    # The environment can be downloaded from
    #https://drive.google.com/drive/u/2/folders/1u5teth6l4JW2IXAkZAg1CbDGR6zE-v6Z
    orig_ip = [
        [-1450, -520, 90],  # Player Start
        [-940, 240, 130],
        [860, -880, 160],
        [-750, -860, 0]
    ]
    level_name = ['Pyramid1', 'Pyramid2', 'Pyramid3', 'Pyramid4']
    crash_threshold = 0.07
    initZ = 0
    return orig_ip, level_name, crash_threshold, initZ


def indoor_frogeyes():
    # The environment can be downloaded from
    # https://drive.google.com/drive/u/2/folders/1u5teth6l4JW2IXAkZAg1CbDGR6zE-v6Z
    orig_ip = [
        [120, -350, 0],  # Player Start
        [1030, 850, 150],
        [-1480, -850, 0],
        [2000, -400, -110]
    ]
    level_name = ['Pyramid1', 'Pyramid2', 'Pyramid3', 'Pyramid4']
    crash_threshold = 0.07
    initZ = 0
    return orig_ip, level_name, crash_threshold, initZ


def indoor_twist():
    # The environment can be downloaded from
    # https://drive.google.com/drive/u/2/folders/1u5teth6l4JW2IXAkZAg1CbDGR6zE-v6Z
    orig_ip = [
        [-1990, 1070, 90],  # Player Start
        [-1440, 320, 0],
        [1460, -1440, 0],
        [1970, 360, 180]
    ]
    level_name = ['Twist1', 'Twist2', 'Twist3', 'Twist4']
    crash_threshold = 0.07
    initZ = 0
    return orig_ip, level_name, crash_threshold, initZ


def indoor_pretzel():
    orig_ip =   [
                  [3308, 650, 180], # Player start
                   [3330, -200, -160],
                   [1480, -1040, 25]
                ]
    level_name = ['Cloud1', 'Cloud2', 'Cloud3']
    crash_threshold = 0.07
    initZ = 0
    return orig_ip, level_name, crash_threshold, initZ

def initial_positions(name):
    name = name+'()'
    orig_ip, level_name, crash_threshold, initZ = eval(name)
    player_start_unreal=orig_ip[0]
    reset_array = []

    for i in range(0, len(orig_ip)):
        x1 = (orig_ip[i][0]-player_start_unreal[0])/100
        y1 = (orig_ip[i][1]-player_start_unreal[1])/100
        z1 = 0
        # z1 = initZ # in case of computervision mode
        pitch = 0
        roll = 0
        yaw = orig_ip[i][2]*np.pi/180
        pp = airsim.Pose(airsim.Vector3r(x1, y1, z1), airsim.to_quaternion(pitch, roll, yaw))
        reset_array.append(pp)

    return reset_array, level_name, crash_threshold, initZ
