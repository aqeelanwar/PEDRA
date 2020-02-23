# Author: Aqeel Anwar(ICSRL)
# Created: 10/14/2019, 12:50 PM
# Email: aqeel.anwar@gatech.edu
import numpy as np
import os, subprocess, psutil
import math
import random
import time
import airsim
import pygame
from configs.read_cfg import read_cfg
import matplotlib.pyplot as plt


def close_env(env_process):
    process = psutil.Process(env_process.pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


def save_network_path(cfg, algorithm_cfg):
    # Save the network to the directory network_path
    weights_type = 'Imagenet'
    if algorithm_cfg.custom_load == True:
        algorithm_cfg.network_path = 'models/trained/' + cfg.env_type + '/' + cfg.env_name + '/' + 'CustomLoad/' + algorithm_cfg.train_type + '/' + algorithm_cfg.train_type
    else:
        algorithm_cfg.network_path = 'models/trained/' + '/' + cfg.env_type + '/' + cfg.env_name + '/' + weights_type + '/' + algorithm_cfg.train_type + '/' + algorithm_cfg.train_type

    if not os.path.exists(algorithm_cfg.network_path):
        os.makedirs(algorithm_cfg.network_path)

    return cfg, algorithm_cfg



def start_environment(env_name):
    env_folder = os.path.dirname(os.path.abspath(__file__)) + "/unreal_envs/" + env_name + "/"
    path = env_folder + env_name + ".exe"
    env_process = subprocess.Popen(path)
    time.sleep(5)
    print("Successfully loaded environment: " + env_name)

    return env_process, env_folder


def initialize_infer(env_cfg, client, env_folder):

    if not os.path.exists(env_folder+'results'):
        os.makedirs(env_folder+'results')

    # Mapping floor to 0 height
    c_z = (env_cfg.ceiling_z-env_cfg.floor_z)/100
    p_z = (env_cfg.player_start_z-env_cfg.floor_z)/100

    plt.ion()
    fig_z = plt.figure()
    ax_z = fig_z.add_subplot(111)
    line_z, = ax_z.plot(0, 0)
    ax_z.set_ylim(0, c_z)
    plt.title("Altitude variation")

    # start_posit = client.simGetVehiclePose()

    fig_nav = plt.figure()
    ax_nav = fig_nav.add_subplot(111)
    img = plt.imread(env_folder+ env_cfg.floorplan)
    ax_nav.imshow(img)
    plt.axis('off')
    plt.title("Navigational map")
    plt.plot(env_cfg.o_x, env_cfg.o_y, 'b*', linewidth=20)
    nav, = ax_nav.plot(env_cfg.o_x, env_cfg.o_y)


    return p_z, fig_z, ax_z, line_z, fig_nav, ax_nav, nav

def translate_action(action, num_actions):
    # action_word = ['Forward', 'Right', 'Left', 'Sharp Right', 'Sharp Left']
    sqrt_num_actions = np.sqrt(num_actions)
    # ind = np.arange(sqrt_num_actions)
    if sqrt_num_actions % 2 == 0:
        v_string = list('U' * int((sqrt_num_actions - 1) / 2) + 'D' * int((sqrt_num_actions - 1) / 2))
        h_string = list('L' * int((sqrt_num_actions - 1) / 2) + 'R' * int((sqrt_num_actions - 1) / 2))
    else:
        v_string = list('U' * int(sqrt_num_actions / 2) + 'F' + 'D' * int(sqrt_num_actions / 2))
        h_string = list('L' * int(sqrt_num_actions / 2) + 'F' + 'R' * int(sqrt_num_actions / 2))

    v_ind = int(action[0]/sqrt_num_actions)
    h_ind = int(action[0]%sqrt_num_actions)
    action_word = v_string[v_ind] + str(int(np.ceil(abs((sqrt_num_actions-1)/2-v_ind)))) + '-' + h_string[h_ind]+str(int(np.ceil(abs((sqrt_num_actions-1)/2-h_ind))))

    return action_word

def get_errors(data_tuple, choose, ReplayMemory, input_size, agent, target_agent, gamma, Q_clip):

    _, Q_target, _, err, _ = minibatch_double(data_tuple, len(data_tuple), choose,  ReplayMemory, input_size, agent, target_agent, gamma, Q_clip)

    return err


def minibatch_double(data_tuple, batch_size, choose, ReplayMemory, input_size, agent, target_agent, gamma, Q_clip):
    # Needs NOT to be in DeepAgent
    # NO TD error term, and using huber loss instead
    # Bellman Optimality equation update, with less computation, updated

    if batch_size==1:
        train_batch = data_tuple
        idx=None
    else:
        batch = ReplayMemory.sample(batch_size)
        train_batch = np.array([b[1][0] for b in batch])
        idx = [b[0] for b in batch]


    actions = np.zeros(shape=(batch_size), dtype=int)
    crashes = np.zeros(shape=(batch_size))
    rewards = np.zeros(shape=batch_size)
    curr_states = np.zeros(shape=(batch_size, input_size, input_size, 3))
    new_states = np.zeros(shape=(batch_size, input_size, input_size, 3))
    for ii, m in enumerate(train_batch):
        curr_state_m, action_m, new_state_m, reward_m, crash_m = m
        curr_states[ii, :, :, :] = curr_state_m[...]
        actions[ii] = action_m
        new_states[ii,:,:,:] = new_state_m
        rewards[ii] = reward_m
        crashes[ii] = crash_m

    #
    # oldQval = np.zeros(shape = [batch_size, num_actions])
    if choose:
        oldQval_A = target_agent.Q_val(curr_states)
        newQval_A = target_agent.Q_val(new_states)
        newQval_B = agent.Q_val(new_states)
    else:
        oldQval_A = agent.Q_val(curr_states)
        newQval_A = agent.Q_val(new_states)
        newQval_B = target_agent.Q_val(new_states)


    TD = np.zeros(shape=[batch_size])
    err = np.zeros(shape=[batch_size])
    Q_target = np.zeros(shape=[batch_size])

    term_ind = np.where(rewards==-1)[0]
    nonterm_ind = np.where(rewards!=-1)[0]

    TD[nonterm_ind] = rewards[nonterm_ind] + gamma* newQval_B[nonterm_ind, np.argmax(newQval_A[nonterm_ind], axis=1)] - oldQval_A[nonterm_ind, actions[nonterm_ind].astype(int)]
    TD[term_ind] = rewards[term_ind]

    if Q_clip:
        TD_clip = np.clip(TD, -1, 1)
    else:
        TD_clip = TD

    Q_target[nonterm_ind] = oldQval_A[nonterm_ind, actions[nonterm_ind].astype(int)] + TD_clip[nonterm_ind]
    Q_target[term_ind] = TD_clip[term_ind]

    err=abs(TD) # or abs(TD_clip)
    return curr_states, Q_target, actions, err, idx


def policy(epsilon,curr_state, iter, b, epsilon_model, wait_before_train, num_actions, agent):
    qvals=[]

    epsilon_ceil=0.95
    if epsilon_model=='linear':
        epsilon = epsilon_ceil* (iter-wait_before_train) / (b-wait_before_train)
        if epsilon > epsilon_ceil:
            epsilon = epsilon_ceil

    elif epsilon_model=='exponential':
        epsilon = 1- math.exp(-2/(b-wait_before_train) * (iter-wait_before_train) )
        if epsilon > epsilon_ceil:
            epsilon = epsilon_ceil

    if random.random() > epsilon:
        sss =curr_state.shape
        action = np.random.randint(0, num_actions, size = sss[0], dtype=np.int32)
        action_type = 'Rand'
    else:
        # Use NN to predict action
        action = agent.action_selection(curr_state)
        action_type = 'Pred'
        # print(action_array/(np.mean(action_array)))
    return action, action_type, epsilon, qvals

def reset_to_initial(level, reset_array, client):
    reset_pos = reset_array[level]

    client.simSetVehiclePose(reset_pos, ignore_collison=True)
    time.sleep(0.1)


def connect_drone(ip_address='127.0.0.0', phase='infer'):
    print('------------------------------ Drone ------------------------------')
    client = airsim.MultirotorClient(ip=ip_address, timeout_value=10)
    client.confirmConnection()
    old_posit = client.simGetVehiclePose()
    # if phase == 'train':
    #     client.simSetVehiclePose(
    #         airsim.Pose(airsim.Vector3r(0, 0, 0), old_posit.orientation),
    #         ignore_collison=True)
    # elif phase == 'infer':
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()

    return client, old_posit

def blit_text(surface, text, pos, font, color=pygame.Color('black')):
    words = [word.split(' ') for word in text.splitlines()]  # 2D array where each row is a list of words.
    space = font.size(' ')[0]  # The width of a space.
    max_width, max_height = surface.get_size()
    x, y = pos
    for line in words:
        for word in line:
            word_surface = font.render(word, 0, color)
            word_width, word_height = word_surface.get_size()
            if x + word_width >= max_width:
                x = pos[0]  # Reset the x.
                y += word_height  # Start on new row.
            surface.blit(word_surface, (x, y))
            x += word_width + space
        x = pos[0]  # Reset the x.
        y += word_height  # Start on new row.

def pygame_connect(phase):
    pygame.init()

    if phase == 'train':
        img_path = 'images/train_keys.png'
    elif phase == 'infer':
        img_path = 'images/infer_keys.png'
    img = pygame.image.load(img_path)

    screen = pygame.display.set_mode(img.get_rect().size)

    screen.blit(img, (0, 0))
    pygame.display.set_caption('DLwithTL')
    # screen.fill((21, 116, 163))
    # text = 'Supported Keys:\n'
    # font = pygame.font.SysFont('arial', 32)
    # blit_text(screen, text, (20, 20), font, color = (214, 169, 19))
    # pygame.display.update()
    #
    # font = pygame.font.SysFont('arial', 24)
    # text = 'R - Reconnect unreal\nbackspace - Pause/play\nL - Update configurations\nEnter - Save Network'
    # blit_text(screen, text, (20, 70), font, color=(214, 169, 19))
    pygame.display.update()

    return screen

def check_user_input(active, automate, lr, epsilon, agent, network_path, client, old_posit, initZ, phase, fig_z, fig_nav, env_folder):
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            active = False
            pygame.quit()

        # Training keys control
        if event.type == pygame.KEYDOWN and phase =='train':
            if event.key == pygame.K_l:
                # Load the parameters - epsilon
                cfg = read_cfg(config_filename='configs/config.cfg', verbose=False)
                lr = cfg.lr
                print('Updated Parameters')
                print('Learning Rate: ', cfg.lr)

            if event.key == pygame.K_RETURN:
                # take_action(-1)
                automate = False
                print('Saving Model')
                # agent.save_network(iter, save_path, ' ')
                agent.save_network(network_path)
                # agent.save_data(iter, data_tuple, tuple_path)
                print('Model Saved: ', network_path)


            if event.key == pygame.K_BACKSPACE:
                automate = automate ^ True

            if event.key == pygame.K_r:
                # reconnect
                client = []
                client = airsim.MultirotorClient()
                client.confirmConnection()
                # posit1_old = client.simGetVehiclePose()
                client.simSetVehiclePose(old_posit,
                                         ignore_collison=True)
                agent.client = client

            if event.key == pygame.K_m:
                agent.get_state()
                print('got_state')
                # automate = automate ^ True

            # Set the routine for manual control if not automate

            if not automate:
                # print('manual')
                # action=[-1]
                if event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_RIGHT:
                    action = 1
                elif event.key == pygame.K_LEFT:
                    action = 2
                elif event.key == pygame.K_d:
                    action = 3
                elif event.key == pygame.K_a:
                    action = 4
                elif event.key == pygame.K_DOWN:
                    action = -2
                elif event.key == pygame.K_y:
                    pos = client.getPosition()

                    client.moveToPosition(pos.x_val, pos.y_val, 3 * initZ, 1)
                    time.sleep(0.5)
                elif event.key == pygame.K_h:
                    client.reset()
                # agent.take_action(action)

        elif event.type == pygame.KEYDOWN and phase == 'infer':
            if event.key == pygame.K_s:
                # Save the figures
                file_path = env_folder + 'results/'
                fig_z.savefig(file_path+'altitude_variation.png', dpi=1000)
                fig_nav.savefig(file_path+'navigation.png', dpi=1000)
                print('Figures saved')

            if event.key == pygame.K_BACKSPACE:
                client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=0.1)
                automate = automate ^ True

    return active, automate, lr, client



