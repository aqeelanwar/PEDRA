# # # # Author: Aqeel Anwar(ICSRL)
# # # # Created: 11/8/2019, 4:02 PM
# # # # Email: aqeel.anwar@gatech.edu
# # #
# # #
# # # # import airsim
# # import os
# # import time
# # from aux_functions import *
# # from util.transformations import euler_from_quaternion
# # import airsim
# #
# # # Connect to Unreal Engine and get the drone handle: client
# # client, old_posit = connect_drone(ip_address='127.0.0.1', phase='infer')
# #
# #
# # # Async methods returns Future. Call join() to wait for task to complete.
# # # client.takeoffAsync().join()
# # pos = client.simGetVehiclePose()
# # # print(pos)
# # num_actions = 25
# # action = [20]
# # fov_v = 45 * np.pi / 180
# # fov_h = 80 * np.pi / 180
# # r = 0.4
# #
# # sqrt_num_actions = np.sqrt(num_actions)
# #
# #
# # for i in range(5):
# #
# #     client.moveByVelocityAsync(vx=0, vy=0, vz=-1, duration=1).join()
# #     posit = client.simGetVehiclePose()
# #     pos = posit.position
# #     orientation = posit.orientation
# #     print("Z", pos.z_val)
# #     cc=1
#
# #
#
# # quat = (orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val)
# # eulers = euler_from_quaternion(quat)
# # alpha = eulers[2]
# #
# # theta_ind = int(action[0] / sqrt_num_actions)
# # psi_ind = action[0] % sqrt_num_actions
# #
# # theta = fov_v/sqrt_num_actions * (theta_ind - (sqrt_num_actions - 1) / 2)
# # psi = fov_h / sqrt_num_actions * (psi_ind - (sqrt_num_actions - 1) / 2)
# #
# # vx = r * np.cos(alpha + psi)
# # vy = r * np.sin(alpha + psi)
# # vz = r * np.sin(theta)
# # print("ang:", 180*(alpha+psi)/np.pi)
# # client.moveByVelocityAsync(vx=vx, vy=vy, vz=vz, duration=3, drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=180*(alpha+psi)/np.pi)).join()
# # client.moveByVelocityAsync(vx=0,vy=0,vz=0, duration=0.1).join()
# #
# # posit = client.simGetVehiclePose()
# # pos = posit.position
# # orientation = posit.orientation
# #
# # quat = (orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val)
# # eulers = euler_from_quaternion(quat)
# # alpha = eulers[2]
# # print(eulers)
# #
# #
# # # time.sleep(5)
# # # client.moveToPositionAsync(-10, 10, -10, 5).join()
# # print("Done")
# # #
# #
# #
# # def action_take(action):
# #     action = [22]
# #
# #     fov_v = 45 * np.pi / 180
# #     fov_h = 80 * np.pi / 180
# #
# #     sqrt_num_actions = 5
# #     theta_ind = int(action[0] / sqrt_num_actions)
# #     psi_ind = action[0] % sqrt_num_actions
# #
# #     theta = fov_v / sqrt_num_actions * (theta_ind - (sqrt_num_actions - 1) / 2)
# #     psi = fov_h / sqrt_num_actions * (psi_ind - (sqrt_num_actions - 1) / 2)
# #
# #     yaw = psi
# #     throttle = 0.596 + np.sin(theta)
# #
# #     print("")
# #     print("Throttle:", throttle)
# #     print('Yaw:', yaw)
# #
# #     return throttle, yaw
#
#
# import numpy as np
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# # import time, random
# # # x, y, z = np.random.rand(3, 100)
# # plt.style.use('seaborn-pastel')
# # cmap = sns.cubehelix_palette(as_cmap=True)
# # plt.ion()
# # fig = plt.figure()
# # ax = fig.add_subplot(111)
# # x=[]
# # y=[]
# # points, = ax.plot(x, y)
# # # ax.set_ylim([0, 5])
# # # ax.set_xlim([0, 10])
# # img = plt.imread("unreal_envs/indoor_condo/floor.png")
# # ax.imshow(img)
# # plt.axis('off')
# # plt.title("Navigational map")
# # o_x = 1318.49
# # o_y = 769.635
# # # a = 10
# # # b = [3,5]
# # # plt.axis('off')
# # for i in range(10):
# #     x.append(i)
# #     y.append(random.randint(1,5))
# #     points.set_data(x,y)
# #     fig.canvas.draw()
# #     fig.canvas.flush_events()
# #     time.sleep(1)
#
#
import matplotlib.pyplot as plt
import time, random

altitude=[]
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(0, 0)
plt.title("Altitude variation")

nav_x=[]
nav_y=[]
o_x = 519
o_y = 112
alpha = 32.72
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
img = plt.imread("unreal_envs/indoor_vanleer/techno_floor.png")
ax2.imshow(img)
plt.axis('off')
plt.title("Navigational map")
plt.plot(o_x, o_y, 'r*', linewidth=5)
nav = ax2.plot(o_x, o_y, animated=True)
import numpy as np
from matplotlib import animation, rc
from matplotlib.animation import PillowWriter
navs=[]
for i in range(10):
    nav_text = ax2.text(25, 55, 'Distance: ' + str(np.round(1.2333, 2)), style='italic',
                           bbox={'facecolor': 'white', 'alpha': 0.5})
    nav_x.append(alpha * random.randint(0,5) + o_x)
    nav_y.append(alpha * random.randint(0,5) + o_y)
    nav[0].set_data(nav_x, nav_y)
    # line1.set_ydata(altitude)
    # print("alt: ", altitude)
    navs.append([nav])
    altitude.append(random.randint(0,5))
    line1.set_data(altitude, altitude)
    fig.canvas.draw()
    fig.canvas.flush_events()
#
#     time.sleep(0.1)
# print('Getting animation')
# anim = animation.ArtistAnimation(fig, navs, interval=50, blit=True,
#                                 repeat_delay=1000)
# writer = PillowWriter(fps=20)
# anim.save("demo1.mp4")
#
# import numpy as npice
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from matplotlib.animation import PillowWriter
#
# fig = plt.figure()
#
# def f(x, y):
#     return np.sin(x) + np.cos(y)
#
# x = np.linspace(0, 2 * np.pi, 120)
# y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
# # ims is a list of lists, each row is a list of artists to draw in the
# # current frame; here we are just animating one artist, the image, in
# # each frame
# ims = []
# for i in range(20):
#     x += np.pi / 15.
#     y += np.pi / 20.
#     im = plt.imshow(f(x, y))
#     im1 = plt.plot(f(x,y))
#     ims.append([im])
#     cc=1
#
# ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True,
#                                 repeat_delay=500)
#
# writer = PillowWriter(fps=20)
# ani.save("demo2.gif", writer=writer)
#
# plt.show()
