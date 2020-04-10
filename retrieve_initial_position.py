# Author: Aqeel Anwar(ICSRL)
# Created: 3/10/2020, 7:48 PM
# Email: aqeel.anwar@gatech.edu

import matplotlib.pyplot as plt
from configs.read_cfg import read_cfg
from unreal_envs.initial_positions import *

from tkinter import filedialog
from tkinter import *
import cv2, os

root = Tk()
filename =filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("PNG files","*.PNG"),("png files","*.png"),("All files","*.*"))   )
root.destroy()

filename_split = os.path.split(filename)
folder = filename_split[0]
floor_cfg_filepath = folder+'/config.cfg'
floor_cfg = read_cfg(floor_cfg_filepath)
name = floor_cfg.env_name+'()'
orig_ip, levels, crash_threshold = eval(name)
player_x_env = orig_ip[0][0]
player_y_env = orig_ip[0][1]

env_cfg_filepath = 'environments'

env_cfg = read_cfg(env_cfg_filepath)

coords =[]
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
floor_image = cv2.imread(filename)
floor_image = cv2.cvtColor(floor_image, cv2.COLOR_BGR2RGB)
plt.imshow(floor_image)

def onclick(event):
    ix, iy = event.xdata, event.ydata
    global coords
    x_unreal = (ix - floor_cfg.o_x)/floor_cfg.alpha
    y_unreal = (iy - floor_cfg.o_y) / floor_cfg.alpha

    x_env = 100 * x_unreal+player_x_env
    y_env = 100 * y_unreal+player_y_env

    coords.append((ix, iy))
    global text

    if len(coords) > 1:
        text.remove()
    text_str = 'Image coordinates \nx: ' + str(np.round(ix,2)) + '\ny:' + str(np.round(iy,2))
    text_str += '\n\nUnreal coordinates \nx: ' + str(np.round(x_unreal, 2)) + '\ny:' + str(np.round(y_unreal, 2))
    text_str += '\n\nEnvironment coordinates \nx: ' + str(np.round(x_env, 2)) + '\ny:' + str(np.round(y_env, 2))
    print(text_str)
    print('-'*50)
    text = ax.text(25, 275, text_str, style='italic',
                           bbox={'facecolor': 'white', 'alpha': 0.5})
    plt.axis('off')
    plt.show()

    return coords
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()