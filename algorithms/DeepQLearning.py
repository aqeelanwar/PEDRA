# Author: Aqeel Anwar(ICSRL)
# Created: 2/19/2020, 8:39 AM
# Email: aqeel.anwar@gatech.edu

import sys, cv2
import nvidia_smi
from network.agent import PedraAgent
from unreal_envs.initial_positions import *
from os import getpid
from network.Memory import Memory
from aux_functions import *
import os
from util.transformations import euler_from_quaternion
from configs.read_cfg import read_cfg, update_algorithm_cfg


def DeepQLearning(cfg, env_process, env_folder):
    algorithm_cfg = read_cfg(config_filename='configs/DeepQLearning.cfg', verbose=True)
    algorithm_cfg.algorithm = cfg.algorithm

    if 'GlobalLearningGlobalUpdate-SA' in algorithm_cfg.distributed_algo:
        # algorithm_cfg = update_algorithm_cfg(algorithm_cfg, cfg)
        cfg.num_agents = 1

    # # Start the environment
    # env_process, env_folder = start_environment(env_name=cfg.env_name)
    # Connect to Unreal Engine and get the drone handle: client
    client, old_posit, initZ = connect_drone(ip_address=cfg.ip_address, phase=cfg.mode, num_agents=cfg.num_agents)
    initial_pos = old_posit.copy()
    # Load the initial positions for the environment
    reset_array, reset_array_raw, level_name, crash_threshold = initial_positions(cfg.env_name, initZ, cfg.num_agents)

    # Initialize System Handlers
    process = psutil.Process(getpid())
    # nvidia_smi.nvmlInit()

    # Load PyGame Screen
    screen = pygame_connect(phase=cfg.mode)

    fig_z = []
    fig_nav = []
    debug = False
    # Generate path where the weights will be saved
    cfg, algorithm_cfg = save_network_path(cfg=cfg, algorithm_cfg=algorithm_cfg)
    current_state = {}
    new_state = {}
    posit = {}
    name_agent_list = []
    agent = {}
    # Replay Memory for RL
    if cfg.mode == 'train':
        ReplayMemory = {}
        target_agent = {}

        if algorithm_cfg.distributed_algo == 'GlobalLearningGlobalUpdate-MA':
            print_orderly('global', 40)
            # Multiple agent acts as data collecter and one global learner
            global_agent = PedraAgent(algorithm_cfg, client, name='DQN', vehicle_name='global')
            ReplayMemory = Memory(algorithm_cfg.buffer_len)
            target_agent = PedraAgent(algorithm_cfg, client, name='Target', vehicle_name='global')

        for drone in range(cfg.num_agents):
            name_agent = "drone" + str(drone)
            name_agent_list.append(name_agent)
            print_orderly(name_agent, 40)
            # TODO: turn the neural network off if global agent is present
            agent[name_agent] = PedraAgent(algorithm_cfg, client, name='DQN', vehicle_name=name_agent)

            if algorithm_cfg.distributed_algo != 'GlobalLearningGlobalUpdate-MA':
                ReplayMemory[name_agent] = Memory(algorithm_cfg.buffer_len)
                target_agent[name_agent] = PedraAgent(algorithm_cfg, client, name='Target', vehicle_name=name_agent)
            current_state[name_agent] = agent[name_agent].get_state()

    elif cfg.mode == 'infer':
        name_agent = 'drone0'
        name_agent_list.append(name_agent)
        agent[name_agent] = PedraAgent(algorithm_cfg, client, name=name_agent + 'DQN', vehicle_name=name_agent)

        env_cfg = read_cfg(config_filename=env_folder + 'config.cfg')
        nav_x = []
        nav_y = []
        altitude = {}
        altitude[name_agent] = []
        p_z, f_z, fig_z, ax_z, line_z, fig_nav, ax_nav, nav = initialize_infer(env_cfg=env_cfg, client=client,
                                                                               env_folder=env_folder)
        nav_text = ax_nav.text(0, 0, '')

        # Select initial position
        reset_to_initial(0, reset_array, client, vehicle_name=name_agent)
        old_posit[name_agent] = client.simGetVehiclePose(vehicle_name=name_agent)

    # Initialize variables
    iter = 1
    # num_collisions = 0
    episode = {}
    active = True

    print_interval = 1
    automate = True
    choose = False
    print_qval = False
    last_crash = {}
    ret = {}
    distance = {}
    num_collisions = {}
    level = {}
    level_state = {}
    level_posit = {}
    times_switch = {}
    last_crash_array = {}
    ret_array = {}
    distance_array = {}
    epi_env_array = {}
    log_files = {}

    # If the phase is inference force the num_agents to 1
    hyphens = '-' * int((80 - len('Log files')) / 2)
    print(hyphens + ' ' + 'Log files' + ' ' + hyphens)
    for name_agent in name_agent_list:
        ret[name_agent] = 0
        num_collisions[name_agent] = 0
        last_crash[name_agent] = 0
        level[name_agent] = 0
        episode[name_agent] = 0
        level_state[name_agent] = [None] * len(reset_array[name_agent])
        level_posit[name_agent] = [None] * len(reset_array[name_agent])
        times_switch[name_agent] = 0
        last_crash_array[name_agent] = np.zeros(shape=len(reset_array[name_agent]), dtype=np.int32)
        ret_array[name_agent] = np.zeros(shape=len(reset_array[name_agent]))
        distance_array[name_agent] = np.zeros(shape=len(reset_array[name_agent]))
        epi_env_array[name_agent] = np.zeros(shape=len(reset_array[name_agent]), dtype=np.int32)
        distance[name_agent] = 0
        # Log file
        log_path = algorithm_cfg.network_path + '/' + name_agent + '/' + cfg.mode + 'log.txt'
        print("Log path: ", log_path)
        log_files[name_agent] = open(log_path, 'w')

    print_orderly('Simulation begins', 80)

    while active:
        try:
            active, automate, algorithm_cfg, client = check_user_input(active, automate, agent[name_agent], client,
                                                                       old_posit[name_agent], initZ, fig_z, fig_nav,
                                                                       env_folder, cfg, algorithm_cfg)

            if automate:

                if cfg.mode == 'train':

                    if iter % algorithm_cfg.switch_env_steps == 0:
                        switch_env = True
                    else:
                        switch_env = False

                    for name_agent in name_agent_list:

                        start_time = time.time()
                        if switch_env:
                            posit1_old = client.simGetVehiclePose(vehicle_name=name_agent)
                            times_switch[name_agent] = times_switch[name_agent] + 1
                            level_state[name_agent][level[name_agent]] = current_state[name_agent]
                            level_posit[name_agent][level[name_agent]] = posit1_old
                            last_crash_array[name_agent][level[name_agent]] = last_crash[name_agent]
                            ret_array[name_agent][level[name_agent]] = ret[name_agent]
                            distance_array[name_agent][level[name_agent]] = distance[name_agent]
                            epi_env_array[name_agent][level[name_agent]] = episode[name_agent]

                            level[name_agent] = (level[name_agent] + 1) % len(reset_array[name_agent])

                            print(name_agent + ' :Transferring to level: ', level[name_agent], ' - ',
                                  level_name[name_agent][level[name_agent]])

                            if times_switch[name_agent] < len(reset_array[name_agent]):
                                reset_to_initial(level[name_agent], reset_array, client, vehicle_name=name_agent)
                            else:
                                current_state[name_agent] = level_state[name_agent][level[name_agent]]
                                posit1_old = level_posit[name_agent][level[name_agent]]
                                reset_to_initial(level[name_agent], reset_array, client, vehicle_name=name_agent)
                                client.simSetVehiclePose(posit1_old, ignore_collison=True, vehicle_name=name_agent)
                                time.sleep(0.1)

                            last_crash[name_agent] = last_crash_array[name_agent][level[name_agent]]
                            ret[name_agent] = ret_array[name_agent][level[name_agent]]
                            distance[name_agent] = distance_array[name_agent][level[name_agent]]
                            episode[name_agent] = epi_env_array[name_agent][int(level[name_agent] / 3)]
                            # environ = environ^True
                        else:
                            # TODO: policy from one global agent: DONE
                            if algorithm_cfg.distributed_algo == 'GlobalLearningGlobalUpdate-MA':
                                agent_this_drone = global_agent
                                ReplayMemory_this_drone = ReplayMemory
                                target_agent_this_drone = target_agent
                            else:
                                agent_this_drone = agent[name_agent]
                                ReplayMemory_this_drone = ReplayMemory[name_agent]
                                target_agent_this_drone = target_agent[name_agent]

                            action, action_type, algorithm_cfg.epsilon, qvals = policy(algorithm_cfg.epsilon,
                                                                                       current_state[name_agent], iter,
                                                                                       algorithm_cfg.epsilon_saturation,
                                                                                       algorithm_cfg.epsilon_model,
                                                                                       algorithm_cfg.wait_before_train,
                                                                                       algorithm_cfg.num_actions,
                                                                                       agent_this_drone)

                            action_word = translate_action(action, algorithm_cfg.num_actions)
                            # Take the action
                            agent[name_agent].take_action(action, algorithm_cfg.num_actions, Mode='static')
                            # time.sleep(0.05)
                            new_state[name_agent] = agent[name_agent].get_state()
                            new_depth1, thresh = agent[name_agent].get_CustomDepth(cfg)

                            # Get GPS information
                            posit[name_agent] = client.simGetVehiclePose(vehicle_name=name_agent)
                            position = posit[name_agent].position
                            old_p = np.array(
                                [old_posit[name_agent].position.x_val, old_posit[name_agent].position.y_val])
                            new_p = np.array([position.x_val, position.y_val])

                            # calculate distance
                            distance[name_agent] = distance[name_agent] + np.linalg.norm(new_p - old_p)
                            old_posit[name_agent] = posit[name_agent]

                            reward, crash = agent[name_agent].reward_gen(new_depth1, action, crash_threshold, thresh,
                                                                         debug, cfg)

                            ret[name_agent] = ret[name_agent] + reward
                            agent_state = agent[name_agent].GetAgentState()

                            if agent_state.has_collided or distance[name_agent] < 0.1:
                                num_collisions[name_agent] = num_collisions[name_agent] + 1
                                print('crash')
                                crash = True
                                reward = -1
                            data_tuple = []
                            data_tuple.append([current_state[name_agent], action, new_state[name_agent], reward, crash])
                            # TODO: one replay memory global, target_agent, agent: DONE
                            err = get_errors(data_tuple, choose, ReplayMemory_this_drone, algorithm_cfg.input_size,
                                             agent_this_drone, target_agent_this_drone, algorithm_cfg.gamma,
                                             algorithm_cfg.Q_clip)
                            ReplayMemory_this_drone.add(err, data_tuple)

                            # Train if sufficient frames have been stored
                            if iter > algorithm_cfg.wait_before_train:
                                if iter % algorithm_cfg.train_interval == 0:
                                    # Train the RL network
                                    old_states, Qvals, actions, err, idx = minibatch_double(data_tuple,
                                                                                            algorithm_cfg.batch_size,
                                                                                            choose,
                                                                                            ReplayMemory_this_drone,
                                                                                            algorithm_cfg.input_size,
                                                                                            agent_this_drone,
                                                                                            target_agent_this_drone,
                                                                                            algorithm_cfg.gamma,
                                                                                            algorithm_cfg.Q_clip)
                                    # TODO global replay memory: DONE
                                    for i in range(algorithm_cfg.batch_size):
                                        ReplayMemory_this_drone.update(idx[i], err[i])

                                    if print_qval:
                                        print(Qvals)
                                    # TODO global agent, target_agent: DONE
                                    if choose:
                                        # Double-DQN
                                        target_agent_this_drone.network_model.train_n(old_states, Qvals, actions,
                                                                                      algorithm_cfg.batch_size,
                                                                                      algorithm_cfg.dropout_rate,
                                                                                      algorithm_cfg.learning_rate,
                                                                                      algorithm_cfg.epsilon, iter)
                                    else:
                                        agent_this_drone.network_model.train_n(old_states, Qvals, actions,
                                                                               algorithm_cfg.batch_size,
                                                                               algorithm_cfg.dropout_rate,
                                                                               algorithm_cfg.learning_rate,
                                                                               algorithm_cfg.epsilon, iter)

                            time_exec = time.time() - start_time
                            gpu_memory, gpu_utilization, sys_memory = get_SystemStats(process, cfg.NVIDIA_GPU)

                            for i in range(0, len(gpu_memory)):
                                tag_mem = 'GPU' + str(i) + '-Memory-GB'
                                tag_util = 'GPU' + str(i) + 'Utilization-%'
                                agent[name_agent].network_model.log_to_tensorboard(tag=tag_mem, group='SystemStats',
                                                                                   value=gpu_memory[i],
                                                                                   index=iter)
                                agent[name_agent].network_model.log_to_tensorboard(tag=tag_util, group='SystemStats',
                                                                                   value=gpu_utilization[i],
                                                                                   index=iter)
                            agent[name_agent].network_model.log_to_tensorboard(tag='Memory-GB', group='SystemStats',
                                                                               value=sys_memory,
                                                                               index=iter)

                            s_log = '{:<6s} - Level {:>2d} - Iter: {:>6d}/{:<5d} {:<8s}-{:>5s} Eps: {:<1.4f} lr: {:>1.8f} Ret = {:<+6.4f} Last Crash = {:<5d} t={:<1.3f} SF = {:<5.4f}  Reward: {:<+1.4f}  '.format(
                                name_agent,
                                int(level[name_agent]),
                                iter,
                                episode[name_agent],
                                action_word,
                                action_type,
                                algorithm_cfg.epsilon,
                                algorithm_cfg.learning_rate,
                                ret[name_agent],
                                last_crash[name_agent],
                                time_exec,
                                distance[name_agent],
                                reward)

                            if iter % print_interval == 0:
                                print(s_log)
                            log_files[name_agent].write(s_log + '\n')

                            last_crash[name_agent] = last_crash[name_agent] + 1
                            if debug:
                                cv2.imshow(name_agent, np.hstack((np.squeeze(current_state[name_agent], axis=0),
                                                                  np.squeeze(new_state[name_agent], axis=0))))
                                cv2.waitKey(1)

                            if crash:
                                if distance[name_agent] < 0.01:
                                    # Drone won't move, reconnect
                                    print('Recovering from drone mobility issue')

                                    agent[name_agent].client, old_posit, initZ = connect_drone(
                                        ip_address=cfg.ip_address, phase=cfg.mode,
                                        num_agents=cfg.num_agents, client=client)
                                    time.sleep(2)
                                else:

                                    agent[name_agent].network_model.log_to_tensorboard(tag='Return', group=name_agent,
                                                                                       value=ret[name_agent],
                                                                                       index=episode[name_agent])
                                    agent[name_agent].network_model.log_to_tensorboard(tag='Safe Flight',
                                                                                       group=name_agent,
                                                                                       value=distance[name_agent],
                                                                                       index=episode[name_agent])

                                    ret[name_agent] = 0
                                    distance[name_agent] = 0
                                    episode[name_agent] = episode[name_agent] + 1
                                    last_crash[name_agent] = 0

                                    reset_to_initial(level[name_agent], reset_array, client, vehicle_name=name_agent)
                                    # time.sleep(0.2)
                                    current_state[name_agent] = agent[name_agent].get_state()
                                    old_posit[name_agent] = client.simGetVehiclePose(vehicle_name=name_agent)
                            else:
                                current_state[name_agent] = new_state[name_agent]

                            if iter % algorithm_cfg.max_iters == 0:
                                automate = False

                    # TODO define and state agents
                    if iter % algorithm_cfg.update_target_interval == 0 and iter > algorithm_cfg.wait_before_train:

                        if algorithm_cfg.distributed_algo == 'GlobalLearningGlobalUpdate-MA':
                            print('global' + ' - Switching Target Network')
                            global_agent.network_model.save_network(algorithm_cfg.network_path, episode[name_agent])
                        else:
                            for name_agent in name_agent_list:
                                agent[name_agent].take_action([-1], algorithm_cfg.num_actions, Mode='static')
                                print(name_agent + ' - Switching Target Network')
                                agent[name_agent].network_model.save_network(algorithm_cfg.network_path,
                                                                             episode[name_agent])

                        choose = not choose

                    # if iter % algorithm_cfg.communication_interval == 0 and iter > algorithm_cfg.wait_before_train:
                    #     print('Communicating the weights and averaging them')
                    #     communicate_across_agents(agent, name_agent_list, algorithm_cfg)
                    #     communicate_across_agents(target_agent, name_agent_list, algorithm_cfg)

                    iter += 1

                elif cfg.mode == 'infer':
                    # Inference phase
                    agent_state = agent[name_agent].GetAgentState()
                    if agent_state.has_collided:
                        print('Drone collided')
                        print("Total distance traveled: ", np.round(distance[name_agent], 2))
                        active = False
                        client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=1, vehicle_name=name_agent).join()

                        if nav_x:  # Nav_x is empty if the drone collides in first iteration
                            ax_nav.plot(nav_x.pop(), nav_y.pop(), 'r*', linewidth=20)
                        file_path = env_folder + 'results/'
                        fig_z.savefig(file_path + 'altitude_variation.png', dpi=500)
                        fig_nav.savefig(file_path + 'navigation.png', dpi=500)
                        close_env(env_process)
                        print('Figures saved')
                    else:
                        posit[name_agent] = client.simGetVehiclePose(vehicle_name=name_agent)
                        distance[name_agent] = distance[name_agent] + np.linalg.norm(np.array(
                            [old_posit[name_agent].position.x_val - posit[name_agent].position.x_val,
                             old_posit[name_agent].position.y_val - posit[name_agent].position.y_val]))
                        # altitude[name_agent].append(-posit[name_agent].position.z_val+p_z)
                        altitude[name_agent].append(-posit[name_agent].position.z_val - f_z)

                        quat = (posit[name_agent].orientation.w_val, posit[name_agent].orientation.x_val,
                                posit[name_agent].orientation.y_val, posit[name_agent].orientation.z_val)
                        yaw = euler_from_quaternion(quat)[2]

                        x_val = posit[name_agent].position.x_val
                        y_val = posit[name_agent].position.y_val
                        z_val = posit[name_agent].position.z_val

                        nav_x.append(env_cfg.alpha * x_val + env_cfg.o_x)
                        nav_y.append(env_cfg.alpha * y_val + env_cfg.o_y)
                        nav.set_data(nav_x, nav_y)
                        nav_text.remove()
                        nav_text = ax_nav.text(25, 55, 'Distance: ' + str(np.round(distance[name_agent], 2)),
                                               style='italic',
                                               bbox={'facecolor': 'white', 'alpha': 0.5})

                        line_z.set_data(np.arange(len(altitude[name_agent])), altitude[name_agent])
                        ax_z.set_xlim(0, len(altitude[name_agent]))
                        fig_z.canvas.draw()
                        fig_z.canvas.flush_events()

                        current_state[name_agent] = agent[name_agent].get_state()
                        action, action_type, algorithm_cfg.epsilon, qvals = policy(1, current_state[name_agent], iter,
                                                                                   algorithm_cfg.epsilon_saturation,
                                                                                   'inference',
                                                                                   algorithm_cfg.wait_before_train,
                                                                                   algorithm_cfg.num_actions,
                                                                                   agent[name_agent])
                        action_word = translate_action(action, algorithm_cfg.num_actions)
                        # Take continuous action
                        agent[name_agent].take_action(action, algorithm_cfg.num_actions, Mode='static')
                        old_posit[name_agent] = posit[name_agent]

                        # Verbose and log making
                        s_log = 'Position = ({:<3.2f},{:<3.2f}, {:<3.2f}) Orientation={:<1.3f} Predicted Action: {:<8s}  '.format(
                            x_val, y_val, z_val, yaw, action_word
                        )

                        print(s_log)
                        log_files[name_agent].write(s_log + '\n')



        except Exception as e:
            if str(e) == 'cannot reshape array of size 1 into shape (0,0,3)':
                print('Recovering from AirSim error')
                client, old_posit, initZ = connect_drone(ip_address=cfg.ip_address, phase=cfg.mode,
                                                         num_agents=cfg.num_agents)
                time.sleep(2)
                agent[name_agent].client = client
            else:
                print('------------- Error -------------')
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print(exc_obj)
                automate = False
                print('Hit r and then backspace to start from this point')
