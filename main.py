import sys, cv2
from network.agent import DeepAgent
from environments.initial_positions import *
import psutil
from os import getpid
from network.Memory import Memory
from aux_functions import *
from configs.read_cfg import read_cfg, read_env_cfg, generate_json

# TF Debug message suppressed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Get memory handler
process = psutil.Process(getpid())

# Read the config file
cfg = read_cfg(config_filename='configs/config.cfg', verbose=True)
generate_json(cfg)


# Start the environment
env_process, env_folder = start_environment(env_name=cfg.env_name)

# Load PyGame Screen
screen = pygame_connect(phase = cfg.phase)

# Load the initial positions for the environment
reset_array, level_name, crash_threshold, initZ = initial_positions(cfg.env_name)

# Generate path where the weighst will be saved
cfg = save_network_path(cfg=cfg)

# Replay Memory for RL
ReplayMemory = Memory(cfg.buffer_len)

# Connect to Unreal Engine and get the drone handle: client
client, old_posit = connect_drone(ip_address=cfg.ip_address, phase=cfg.phase)

# Define DQN agents
fig_z=[]
fig_nav=[]
agent = DeepAgent(cfg, client, name='DQN')
if cfg.phase == 'train':
    target_agent = DeepAgent(cfg, client, name='Target')

elif cfg.phase == 'infer':
    env_cfg = read_env_cfg(config_filename=env_folder+'config.cfg')
    nav_x=[]
    nav_y=[]
    altitude=[]
    p_z, fig_z, ax_z, line_z, fig_nav, ax_nav, nav = initialize_infer(env_cfg=env_cfg, client=client, env_folder=env_folder)
    nav_text = ax_nav.text(0, 0, '')

# Initialize variables
iter = 0
num_collisions = 0
episode = 0
active = True

automate = True
choose = False
print_qval = False
last_crash = 0
ret = 0
distance = 0
switch_env = False
level = 0
times_switch = 0
save_posit = old_posit
level_state = [None]*len(level_name)
level_posit = [None]*len(level_name)
last_crash_array = np.zeros(shape=len(level_name), dtype=np.int32)
ret_array = np.zeros(shape=len(level_name))
distance_array = np.zeros(shape=len(level_name))
epi_env_array = np.zeros(shape=len(level_name), dtype=np.int32)


current_state = agent.get_state()

# Log file
log_path = cfg.network_path+'log.txt'
print("Log path: ", log_path)
f = open(log_path, 'w')


while active:
    try:
        active, automate, cfg.lr, client = check_user_input(active, automate, cfg.lr, cfg.epsilon, agent, cfg.network_path, client, old_posit, initZ, cfg.phase, fig_z, fig_nav, env_folder)

        if automate:

            if cfg.phase == 'train':

                start_time = time.time()
                if switch_env:
                    posit1_old = client.simGetVehiclePose()
                    times_switch=times_switch+1
                    level_state[level] = current_state
                    level_posit[level] = posit1_old
                    last_crash_array[level] = last_crash
                    ret_array[level] = ret
                    distance_array[level] = distance
                    epi_env_array[int(level/3)] = episode

                    level = (level + 1) % len(reset_array)

                    print('Transferring to level: ', level ,' - ', level_name[level])

                    if times_switch < len(reset_array):
                        reset_to_initial(level, reset_array, client)
                    else:
                        current_state = level_state[level]
                        posit1_old = level_posit[level]

                        reset_to_initial(level, reset_array, client)
                        client.simSetVehiclePose(posit1_old, ignore_collison=True)
                        time.sleep(0.1)

                    last_crash = last_crash_array[level]
                    ret = ret_array[level]
                    distance = distance_array[level]
                    episode = epi_env_array[int(level/3)]
                    xxx = client.simGetVehiclePose()
                    # environ = environ^True

                action, action_type, cfg.epsilon, qvals = policy(cfg.epsilon, current_state, iter, cfg.epsilon_saturation, cfg.epsilon_model,  cfg.wait_before_train, cfg.num_actions, agent)

                action_word = translate_action(action, cfg.num_actions)
                # Take the action
                agent.take_action(action, cfg.num_actions, SimMode=cfg.SimMode)
                # time.sleep(0.05)

                posit = client.simGetVehiclePose()

                new_state = agent.get_state()
                new_depth1, thresh = agent.get_depth()

                # Get GPS information
                posit = client.simGetVehiclePose()
                orientation = posit.orientation
                position = posit.position
                old_p = np.array([old_posit.position.x_val, old_posit.position.y_val])
                new_p = np.array([position.x_val, position.y_val])
                # calculate distance
                distance = distance + np.linalg.norm(new_p - old_p)
                old_posit = posit

                reward, crash = agent.reward_gen(new_depth1, action, crash_threshold, thresh)

                ret = ret+reward
                agent_state1 = agent.GetAgentState()

                if agent_state1.has_collided:
                    num_collisions = num_collisions + 1
                    print('crash')
                    crash = True
                    reward = -1
                data_tuple=[]
                data_tuple.append([current_state, action, new_state, reward, crash])
                err = get_errors(data_tuple, choose, ReplayMemory, cfg.input_size, agent, target_agent, cfg.gamma, cfg.Q_clip)
                ReplayMemory.add(err, data_tuple)

                # Train if sufficient frames have been stored
                if iter > cfg.wait_before_train:
                    if iter%cfg.train_interval==0:
                    # Train the RL network
                        old_states, Qvals, actions, err, idx = minibatch_double(data_tuple, cfg.batch_size, choose, ReplayMemory, cfg.input_size, agent, target_agent, cfg.gamma, cfg.Q_clip)

                        for i in range(cfg.batch_size):
                            ReplayMemory.update(idx[i], err[i])

                        if print_qval:
                            print(Qvals)

                        if choose:
                            # Double-DQN
                            target_agent.train_n(old_states, Qvals, actions, cfg.batch_size, cfg.dropout_rate, cfg.lr, cfg.epsilon, iter)
                        else:
                            agent.train_n(old_states, Qvals,actions,  cfg.batch_size, cfg.dropout_rate, cfg.lr, cfg.epsilon, iter)

                    if iter % cfg.update_target_interval == 0:
                        agent.take_action([-1], cfg.num_actions, SimMode=cfg.SimMode)
                        print('Switching Target Network')
                        choose = not choose
                        agent.save_network(cfg.network_path)

                iter += 1

                time_exec = time.time()-start_time

                mem_percent = process.memory_info()[0]/2.**30

                s_log = 'Level :{:>2d}: Iter: {:>6d}/{:<5d} {:<8s}-{:>5s} Eps: {:<1.4f} lr: {:>1.8f} Ret = {:<+6.4f} Last Crash = {:<5d} t={:<1.3f} Mem = {:<5.4f}  Reward: {:<+1.4f}  '.format(
                        int(level),iter, episode,
                        action_word,
                        action_type,
                        cfg.epsilon,
                        cfg.lr,
                        ret,
                        last_crash,
                        time_exec,
                        mem_percent,
                        reward)

                print(s_log)
                f.write(s_log+'\n')

                last_crash=last_crash+1
                # cv2.imshow('state', np.hstack((np.squeeze(current_state, axis=0), np.squeeze(new_state, axis=0))))
                # cv2.waitKey(1)


                if crash:
                    agent.return_plot(ret, episode, int(level/3), mem_percent, iter, distance)
                    ret = 0
                    distance = 0
                    episode = episode + 1
                    last_crash = 0

                    reset_to_initial(level, reset_array, client)
                    time.sleep(0.2)
                    current_state = agent.get_state()
                else:
                    current_state = new_state

                if iter%cfg.switch_env_steps == 0:
                    switch_env = True
                else:
                    switch_env = False

                if iter % cfg.max_iters == 0:
                    automate=False

                # if iter >140:
                #     active=False
            elif cfg.phase == 'infer':
                # Inference phase
                agent_state = agent.GetAgentState()
                if agent_state.has_collided:
                    print('Drone collided')
                    print("Total distance traveled: ", np.round(distance, 2))
                    active = False
                    client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=1).join()
                    ax_nav.plot(nav_x.pop(), nav_y.pop(), 'r*', linewidth=20)
                    file_path = env_folder + 'results/'
                    fig_z.savefig(file_path + 'altitude_variation.png', dpi=500)
                    fig_nav.savefig(file_path + 'navigation.png', dpi=500)
                    close_env(env_process)
                    print('Figures saved')
                else:
                    posit = agent.client.simGetVehiclePose()
                    distance = distance + np.linalg.norm(np.array([old_posit.position.x_val-posit.position.x_val,old_posit.position.y_val-posit.position.y_val]))
                    altitude.append(-posit.position.z_val+p_z)

                    x_val = posit.position.x_val
                    y_val = posit.position.y_val

                    nav_x.append(env_cfg.alpha*x_val+env_cfg.o_x)
                    nav_y.append(env_cfg.alpha*y_val+env_cfg.o_y)
                    nav.set_data(nav_x, nav_y)
                    nav_text.remove()
                    nav_text = ax_nav.text(25, 55, 'Distance: '+str(np.round(distance, 2)), style='italic',
                                           bbox={'facecolor': 'white', 'alpha': 0.5})

                    line_z.set_data(np.arange(len(altitude)), altitude)
                    ax_z.set_xlim(0, len(altitude))
                    fig_z.canvas.draw()
                    fig_z.canvas.flush_events()

                    current_state = agent.get_state()
                    action, action_type, cfg.epsilon, qvals = policy(1, current_state, iter,
                                                                      cfg.epsilon_saturation, 'inference',
                                                                      cfg.wait_before_train, cfg.num_actions, agent)
                    # Take continuous action
                    agent.take_action(action, cfg.num_actions, SimMode=cfg.SimMode)
                    old_posit=posit



    except Exception as e:
        print('------------- Error -------------')
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(exc_obj)
        automate = False
        print('Hit r and then backspace to start from this point')





