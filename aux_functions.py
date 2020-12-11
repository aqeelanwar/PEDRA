# Author: Aqeel Anwar(ICSRL)
# Created: 10/14/2019, 12:50 PM
# Email: aqeel.anwar@gatech.edu
import numpy as np
import nvidia_smi
import os, subprocess, psutil
import math
import random
import time
import airsim
import pygame
from configs.read_cfg import read_cfg
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from skimage.util import random_noise


def close_env(env_process):
    process = psutil.Process(env_process.pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


def save_network_path(cfg, algorithm_cfg):
    # Save the network to the directory network_path
    weights_type = 'Imagenet'
    if algorithm_cfg.custom_load == True:
        algorithm_cfg.network_path = 'models/trained/' + cfg.env_type + '/' + cfg.env_name + '/' + 'CustomLoad/' + algorithm_cfg.train_type + '/'
    else:
        algorithm_cfg.network_path = 'models/trained/' + cfg.env_type + '/' + cfg.env_name + '/' + weights_type + '/' + algorithm_cfg.train_type + '/'

    if not os.path.exists(algorithm_cfg.network_path):
        os.makedirs(algorithm_cfg.network_path)

    return cfg, algorithm_cfg


def communicate_across_agents(agent, name_agent_list, algorithm_cfg):
    name_agent = name_agent_list[0]
    update_done = False
    if 'GlobalLearningGlobalUpdate' in algorithm_cfg.distributed_algo:
        # No need to do anything
        update_done = True

    elif algorithm_cfg.distributed_algo == 'LocalLearningGlobalUpdate':
        agent_on_same_network = name_agent_list
        agent[name_agent].network_model.initialize_graphs_with_average(agent, agent_on_same_network)

    elif algorithm_cfg.distributed_algo == 'LocalLearningLocalUpdate':
        agent_connectivity_graph = []
        for j in range(int(np.floor(len(name_agent_list) / algorithm_cfg.average_connectivity))):
            div1 = random.sample(name_agent_list, algorithm_cfg.average_connectivity)
            # print(div1)
            agent_connectivity_graph.append(div1)
            name_agent_list = list(set(name_agent_list) - set(div1))

        if name_agent_list:
            agent_connectivity_graph.append(name_agent_list)

        for agent_network in agent_connectivity_graph:
            agent_on_same_network = agent_network
            agent[name_agent].network_model.initialize_graphs_with_average(agent, agent_on_same_network)

    return update_done


def start_environment(env_name):
    print_orderly('Environment', 80)
    env_folder = os.path.dirname(os.path.abspath(__file__)) + "/unreal_envs/" + env_name + "/"
    path = env_folder + env_name + ".exe"
    # env_process = []
    env_process = subprocess.Popen(path)
    time.sleep(5)
    print("Successfully loaded environment: " + env_name)

    return env_process, env_folder


def initialize_infer(env_cfg, client, env_folder):
    if not os.path.exists(env_folder + 'results'):
        os.makedirs(env_folder + 'results')

    # Mapping floor to 0 height
    f_z = env_cfg.floor_z / 100
    c_z = (env_cfg.ceiling_z - env_cfg.floor_z) / 100
    p_z = (env_cfg.player_start_z - env_cfg.floor_z) / 100

    plt.ion()
    fig_z = plt.figure()
    ax_z = fig_z.add_subplot(111)
    line_z, = ax_z.plot(0, 0)
    ax_z.set_ylim(0, c_z)
    plt.title("Altitude variation")

    # start_posit = client.simGetVehiclePose()

    fig_nav = plt.figure()
    ax_nav = fig_nav.add_subplot(111)
    img = plt.imread(env_folder + env_cfg.floorplan)
    ax_nav.imshow(img)
    plt.axis('off')
    plt.title("Navigational map")
    plt.plot(env_cfg.o_x, env_cfg.o_y, 'b*', linewidth=20)
    nav, = ax_nav.plot(env_cfg.o_x, env_cfg.o_y)

    return p_z, f_z, fig_z, ax_z, line_z, fig_nav, ax_nav, nav


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

    v_ind = int(action[0] / sqrt_num_actions)
    h_ind = int(action[0] % sqrt_num_actions)
    action_word = v_string[v_ind] + str(int(np.ceil(abs((sqrt_num_actions - 1) / 2 - v_ind)))) + '-' + h_string[
        h_ind] + str(int(np.ceil(abs((sqrt_num_actions - 1) / 2 - h_ind))))

    return action_word


def get_errors(data_tuple, choose, ReplayMemory, input_size, agent, target_agent, gamma, Q_clip):
    _, Q_target, _, err, _ = minibatch_double(data_tuple, len(data_tuple), choose, ReplayMemory, input_size, agent,
                                              target_agent, gamma, Q_clip)

    return err


def train_REINFORCE(data_tuple, batch_size, agent, lr, input_size, gamma, epi_num):
    episode_len = len(data_tuple)

    curr_states = np.zeros(shape=(episode_len, input_size, input_size, 3))
    actions = np.zeros(shape=(episode_len), dtype=int)
    crashes = np.zeros(shape=(episode_len))
    rewards = np.zeros(shape=episode_len)

    for ii, m in enumerate(data_tuple):
        curr_state_m, action_m, reward_m, crash_m = m
        curr_states[ii, :, :, :] = curr_state_m[...]
        actions[ii] = action_m
        rewards[ii] = reward_m
        crashes[ii] = crash_m

    Gs = np.zeros(episode_len)
    r = 0
    for episode_step in range(episode_len - 1, -1, -1):
        r = rewards[episode_step] + r * gamma
        Gs[episode_step] = r

    # Normalize the reward to reduce variance in training
    Gs -= np.mean(Gs)
    Gs /= (np.std(Gs) + 1e-8)

    num_batches = int(np.ceil(episode_len / batch_size))
    for i in range(num_batches):
        if i != num_batches - 1:
            x = curr_states[i * batch_size:(i + 1) * batch_size, :, :, :]
            G = Gs[i * batch_size:(i + 1) * batch_size]
            action = actions[i * batch_size:(i + 1) * batch_size]
        else:
            x = curr_states[i * batch_size:, :, :, :]
            G = Gs[i * batch_size:]
            action = actions[i * batch_size:]

        G = np.array([G])
        G = G.T

        # Restructure array
        action = np.array([action])
        action = action.T

        # Get the baseline value
        B = agent.network_model.get_baseline(x)
        # Train the baseline network
        B_ = agent.network_model.train_baseline(x, G, action, lr, epi_num)
        # Train policy network
        agent.network_model.train_policy(x, action, B, G, lr, epi_num)


def train_PPO(data_tuple_total, algorithm_cfg, agent, lr, input_size, gamma, epi_num):
    batch_size = algorithm_cfg.batch_size
    train_epoch_per_batch = algorithm_cfg.train_epoch_per_batch
    lmbda = algorithm_cfg.lmbda
    # # Divide the data tuple in PPO_steps
    # ppo_steps = 3
    # for i in range(int(np.ceil(len(data_tuple) / float(ppo_steps)))):
    #     print(i)
    #     start_ind = i * ppo_steps
    #     end_ind = np.min((len(data_tuple), (i + 1) * ppo_steps))
    #     data_sub = data_tuple[start_ind: end_ind]
    #
    #
    episode_len_total = len(data_tuple_total)
    num_batches = int(np.ceil(episode_len_total / float(batch_size)))
    for i in range(num_batches):
        start_ind = i * batch_size
        end_ind = np.min((len(data_tuple_total), (i + 1) * batch_size))
        data_tuple = data_tuple_total[start_ind: end_ind]
        episode_len = len(data_tuple)

        curr_states = np.zeros(shape=(episode_len, input_size, input_size, 3))
        next_states = np.zeros(shape=(episode_len, input_size, input_size, 3))
        actions = np.zeros(shape=(episode_len, 1), dtype=int)
        crashes = np.zeros(shape=(episode_len, 1))
        rewards = np.zeros(shape=(episode_len, 1))
        p_a = np.zeros(shape=(episode_len,1))

        for ii, m in enumerate(data_tuple):
            curr_state_m, action_m, next_state_m, reward_m, p_a_m, crash_m = m
            curr_states[ii, :, :, :] = curr_state_m[...]
            next_states[ii, :, :, :] = next_state_m[...]
            actions[ii] = action_m
            rewards[ii] = reward_m
            p_a[ii] = p_a_m
            crashes[ii] = ~crash_m

        for i in range(train_epoch_per_batch):
            V_s = agent.network_model.get_state_value(curr_states)
            V_s_ = agent.network_model.get_state_value(next_states)
            TD_target = rewards + gamma*V_s_* crashes
            delta = TD_target - V_s

            GAE_array = []
            GAE=0
            for delta_t in delta[::-1]:
                GAE = gamma*lmbda* GAE + delta_t
                GAE_array.append(GAE)

            GAE_array.reverse()
            GAE = np.array(GAE_array)
            # Normalize the reward to reduce variance in training
            GAE -= np.mean(GAE)
            GAE /= (np.std(GAE) + 1e-8)
            # TODO: zero mean unit std GAE
            agent.network_model.train_policy(curr_states, actions, TD_target, p_a, GAE, lr, epi_num)


def minibatch_double(data_tuple, batch_size, choose, ReplayMemory, input_size, agent, target_agent, gamma, Q_clip):
    # Needs NOT to be in DeepAgent
    # NO TD error term, and using huber loss instead
    # Bellman Optimality equation update, with less computation, updated

    if batch_size == 1:
        train_batch = data_tuple
        idx = None
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
        new_states[ii, :, :, :] = new_state_m
        rewards[ii] = reward_m
        crashes[ii] = crash_m

    #
    # oldQval = np.zeros(shape = [batch_size, num_actions])
    if choose:
        oldQval_A = target_agent.network_model.Q_val(curr_states)
        newQval_A = target_agent.network_model.Q_val(new_states)
        newQval_B = agent.network_model.Q_val(new_states)
    else:
        oldQval_A = agent.network_model.Q_val(curr_states)
        newQval_A = agent.network_model.Q_val(new_states)
        newQval_B = target_agent.network_model.Q_val(new_states)

    TD = np.zeros(shape=[batch_size])
    err = np.zeros(shape=[batch_size])
    Q_target = np.zeros(shape=[batch_size])

    term_ind = np.where(rewards == -1)[0]
    nonterm_ind = np.where(rewards != -1)[0]

    TD[nonterm_ind] = rewards[nonterm_ind] + gamma * newQval_B[nonterm_ind, np.argmax(newQval_A[nonterm_ind], axis=1)] - \
                      oldQval_A[nonterm_ind, actions[nonterm_ind].astype(int)]
    TD[term_ind] = rewards[term_ind]

    if Q_clip:
        TD_clip = np.clip(TD, -1, 1)
    else:
        TD_clip = TD

    Q_target[nonterm_ind] = oldQval_A[nonterm_ind, actions[nonterm_ind].astype(int)] + TD_clip[nonterm_ind]
    Q_target[term_ind] = TD_clip[term_ind]

    err = abs(TD)  # or abs(TD_clip)
    return curr_states, Q_target, actions, err, idx


def policy_REINFORCE(curr_state, agent):
    action = agent.network_model.action_selection(curr_state)
    action_type = 'Prob'
    return action[0], action_type

def policy_PPO(curr_state, agent):
    action, p_a = agent.network_model.action_selection_with_prob(curr_state)
    action_type = 'Prob'
    return action[0], p_a, action_type

def policy(epsilon, curr_state, iter, b, epsilon_model, wait_before_train, num_actions, agent):
    qvals = []

    epsilon_ceil = 0.95
    if epsilon_model == 'linear':
        epsilon = epsilon_ceil * (iter - wait_before_train) / (b - wait_before_train)
        if epsilon > epsilon_ceil:
            epsilon = epsilon_ceil

    elif epsilon_model == 'exponential':
        epsilon = 1 - math.exp(-2 / (b - wait_before_train) * (iter - wait_before_train))
        if epsilon > epsilon_ceil:
            epsilon = epsilon_ceil

    if random.random() > epsilon:
        sss = curr_state.shape
        action = np.random.randint(0, num_actions, size=sss[0], dtype=np.int32)
        action_type = 'Rand'
    else:
        # Use NN to predict action
        action = agent.network_model.action_selection(curr_state)
        action_type = 'Pred'
        # print(action_array/(np.mean(action_array)))
    return action, action_type, epsilon, qvals


def reset_to_initial(level, reset_array, client, vehicle_name):
    reset_pos = reset_array[vehicle_name][level]
    client.simSetVehiclePose(reset_pos, ignore_collison=True, vehicle_name=vehicle_name)
    time.sleep(0.1)


def print_orderly(str, n):
    print('')
    hyphens = '-' * int((n - len(str)) / 2)
    print(hyphens + ' ' + str + ' ' + hyphens)


def connect_drone(ip_address='127.0.0.0', phase='infer', num_agents=1, client=[]):
    if client != []:
        client.reset()
    print_orderly('Drone', 80)
    client = airsim.MultirotorClient(ip=ip_address, timeout_value=10)
    client.confirmConnection()
    time.sleep(1)

    old_posit = {}
    for agents in range(num_agents):
        name_agent = "drone" + str(agents)
        client.enableApiControl(True, name_agent)
        client.armDisarm(True, name_agent)
        # time.sleep(1)
        client.takeoffAsync(vehicle_name=name_agent)
        time.sleep(1)
        old_posit[name_agent] = client.simGetVehiclePose(vehicle_name=name_agent)

    initZ = old_posit[name_agent].position.z_val

    # client.enableApiControl(True)
    # client.armDisarm(True)
    # client.takeoffAsync().join()

    return client, old_posit, initZ


def get_SystemStats(process, NVIDIA_GPU):
    if NVIDIA_GPU:
        deviceCount = nvidia_smi.nvmlDeviceGetCount()
        gpu_memory = []
        gpu_utilization = []
        for i in range(0, deviceCount):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            gpu_stat = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            gpu_memory.append(gpu_stat.memory)
            gpu_utilization.append(gpu_stat.gpu)
    else:
        gpu_memory = []
        gpu_utilization = []

    sys_memory = process.memory_info()[0] / 2. ** 30

    return gpu_memory, gpu_utilization, sys_memory


def get_MonocularImageRGB(client, vehicle_name):
    responses1 = client.simGetImages([
        airsim.ImageRequest('front_center', airsim.ImageType.Scene, False,
                            False)], vehicle_name=vehicle_name)  # scene vision image in uncompressed RGBA array

    response = responses1[0]
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)  # get numpy array
    img_rgba = img1d.reshape(response.height, response.width, 3)
    img = Image.fromarray(img_rgba)
    img_rgb = img.convert('RGB')
    camera_image_rgb = np.asarray(img_rgb)
    camera_image = camera_image_rgb

    return camera_image


def get_StereoImageRGB(client, vehicle_name):
    camera_image = []
    responses = client.simGetImages(
        [
            airsim.ImageRequest('front_left', airsim.ImageType.Scene, False, False),
            airsim.ImageRequest('front_right', airsim.ImageType.Scene, False, False)
        ], vehicle_name=vehicle_name)

    for i in range(2):
        response = responses[i]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)  # get numpy array
        img_rgba = img1d.reshape(response.height, response.width, 3)
        img = Image.fromarray(img_rgba)
        img_rgb = img.convert('RGB')
        camera_image_rgb = np.asarray(img_rgb)
        camera_image.append(camera_image_rgb)

    return camera_image


def get_CustomImage(client, vehicle_name, camera_name):
    responses1 = client.simGetImages([
        airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False,
                            False)], vehicle_name=vehicle_name)  # scene vision image in uncompressed RGBA array

    response = responses1[0]
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)  # get numpy array
    img_rgba = img1d.reshape(response.height, response.width, 3)
    img = Image.fromarray(img_rgba)
    img_rgb = img.convert('RGB')
    camera_image_rgb = np.asarray(img_rgb)
    camera_image = camera_image_rgb

    return camera_image


# def get_image(client, vehicle_name, camera_type, first_frame, last_frame):
#     responses1 = client.simGetImages([  # depth visualization image
#         airsim.ImageRequest("1", airsim.ImageType.Scene, False,
#                             False)], vehicle_name=vehicle_name)  # scene vision image in uncompressed RGBA array
#
#     response = responses1[0]
#     img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)  # get numpy array
#     img_rgba = img1d.reshape(response.height, response.width, 3)
#     img = Image.fromarray(img_rgba)
#     img_rgb = img.convert('RGB')
#     camera_image_rgb = np.asarray(img_rgb)
#
#     if camera_type == 'optical':
#         camera_image = camera_image_rgb
#
#     if camera_type == 'DVS':
#         # camera_image = cv2.normalize(camera_image_rgb, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#         frame1 = cv2.cvtColor(camera_image_rgb, cv2.COLOR_BGR2GRAY)
#         # frame23 = cv2.normalize(frame1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#         frame = np.uint8(np.log1p(frame1))
#         frame = cv2.normalize(frame, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#
#         if first_frame:
#             camera_image = frame
#             first_frame = False
#         else:
#             camera_image = frame - last_frame
#         # ret, thresh1 = cv2.threshold(display_frame, 0.2, 0.8, cv2.THRESH_BINARY)
#         # display_frame1 = cv2.bitwise_and(display_frame, thresh1)
#         last_frame = frame
#
#         camera_image = random_noise(camera_image, mode='s&p', amount=0.005)
#         camera_image = cv2.cvtColor(camera_image, cv2.COLOR_GRAY2BGR)
#
#         cv2.imshow('rgb', camera_image_rgb)
#         cv2.imshow('dvs', camera_image)
#         cc=1

# return camera_image, first_frame, last_frame


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


def check_user_input(active, automate, agent, client, old_posit, initZ, fig_z, fig_nav, env_folder, cfg, algorithm_cfg):
    # algorithm_cfg.learning_rate, algorithm_cfg.epsilon,algorithm_cfg.network_path,cfg.mode,
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            active = False
            pygame.quit()

        # Training keys control
        if event.type == pygame.KEYDOWN and cfg.mode == 'train':
            if event.key == pygame.K_l:
                # Load the parameters - epsilon
                path = 'configs/' + cfg.algorithm + '.cfg'
                algorithm_cfg = read_cfg(config_filename=path, verbose=False)
                cfg, algorithm_cfg = save_network_path(cfg=cfg, algorithm_cfg=algorithm_cfg)
                print('Updated Parameters')

            if event.key == pygame.K_RETURN:
                # take_action(-1)
                automate = False
                print('Saving Model')
                # agent.save_network(iter, save_path, ' ')
                agent.network_model.save_network(algorithm_cfg.network_path, episode='user')
                # agent.save_data(iter, data_tuple, tuple_path)

            if event.key == pygame.K_BACKSPACE:
                automate = automate ^ True

            if event.key == pygame.K_r:
                client, old_posit, initZ = connect_drone(ip_address=cfg.ip_address, phase=cfg.mode,
                                                         num_agents=cfg.num_agents)

                agent.client = client

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

        elif event.type == pygame.KEYDOWN and cfg.mode == 'infer':
            if event.key == pygame.K_s:
                # Save the figures
                file_path = env_folder + 'results/'
                fig_z.savefig(file_path + 'altitude_variation.png', dpi=1000)
                fig_nav.savefig(file_path + 'navigation.png', dpi=1000)
                print('Figures saved')

            if event.key == pygame.K_BACKSPACE:
                client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=0.1)
                automate = automate ^ True

    return active, automate, algorithm_cfg, client
