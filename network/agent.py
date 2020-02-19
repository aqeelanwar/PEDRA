import numpy as np
import os
import tensorflow as tf
import cv2
from network.network import *
import airsim
import random
import matplotlib.pyplot as plt
from util.transformations import euler_from_quaternion
from PIL import Image
from network.loss_functions import *
from numpy import linalg as LA

class DeepAgent():
    def __init__(self, cfg, client, name):
        print('------------------------------ ' +str(name)+ ' ------------------------------')
        self.g = tf.Graph()
        self.iter=0
        with self.g.as_default():

            self.stat_writer = tf.summary.FileWriter(cfg.network_path+'return_plot')
            # name_array = 'D:/train/loss'+'/'+name
            self.loss_writer = tf.summary.FileWriter(cfg.network_path+'loss/'+name)
            self.env_type=cfg.env_type
            self.client=client
            self.input_size = cfg.input_size
            self.num_actions = cfg.num_actions

            #Placeholders
            self.batch_size = tf.placeholder(tf.int32, shape=())
            self.learning_rate = tf.placeholder(tf.float32, shape=())
            self.X1 = tf.placeholder(tf.float32, [None, cfg.input_size, cfg.input_size, 3], name='States')

            #self.X = tf.image.resize_images(self.X1, (227, 227))


            self.X = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), self.X1)
            self.target = tf.placeholder(tf.float32,    shape = [None], name='Qvals')
            self.actions= tf.placeholder(tf.int32,      shape = [None], name='Actions')

            self.model = AlexNetDuel(self.X, cfg.num_actions, cfg.train_fc)

            self.predict = self.model.output
            ind = tf.one_hot(self.actions, cfg.num_actions)
            pred_Q = tf.reduce_sum(tf.multiply(self.model.output, ind), axis=1)
            self.loss = huber_loss(pred_Q, self.target)
            self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.99).minimize(self.loss, name="train")

            self.sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            self.saver = tf.train.Saver()

            self.sess.graph.finalize()

        # Load custom weights from custom_load_path if required
        if cfg.custom_load:
            print('Loading weights from: ', cfg.custom_load_path)
            self.load_network(cfg.custom_load_path)


        # print()

    def Q_val(self, xs):
        target = np.zeros(shape=[xs.shape[0]], dtype=np.float32)
        actions = np.zeros(dtype=int, shape=[xs.shape[0]])
        return self.sess.run(self.predict,
                      feed_dict={self.batch_size: xs.shape[0], self.learning_rate: 0, self.X1: xs,
                                 self.target: target, self.actions:actions})


    def train_n(self, xs, ys,actions, batch_size, dropout_rate, lr, epsilon, iter):
        # loss = self.sess.run(self.loss,
        #                      feed_dict={self.batch_size: batch_size, self.dropout_rate: dropout_rate, self.learning_rate: lr, self.X: xs,
        #                                        self.Y: ys, self.actions:actions})
        _, loss, Q = self.sess.run([self.train,self.loss, self.predict],
                      feed_dict={self.batch_size: batch_size, self.learning_rate: lr, self.X1: xs,
                                               self.target: ys, self.actions: actions})
        meanQ = np.mean(Q)
        maxQ = np.max(Q)


        summary = tf.Summary()
        # summary.value.add(tag='Loss', simple_value=LA.norm(loss[ind, actions.astype(int)]))
        summary.value.add(tag='Loss', simple_value=LA.norm(loss)/batch_size)
        self.loss_writer.add_summary(summary, iter)

        summary = tf.Summary()
        summary.value.add(tag='Epsilon', simple_value=epsilon)
        self.loss_writer.add_summary(summary, iter)

        summary = tf.Summary()
        summary.value.add(tag='Learning Rate', simple_value=lr)
        self.loss_writer.add_summary(summary, iter)

        summary = tf.Summary()
        summary.value.add(tag='MeanQ', simple_value=meanQ)
        self.loss_writer.add_summary(summary, iter)

        summary = tf.Summary()
        summary.value.add(tag='MaxQ', simple_value=maxQ)
        self.loss_writer.add_summary(summary, iter)

        # return _correct

    def action_selection(self, state):
        target = np.zeros(shape=[state.shape[0]], dtype=np.float32)
        actions = np.zeros(dtype=int, shape=[state.shape[0]])
        qvals= self.sess.run(self.predict,
                             feed_dict={self.batch_size: state.shape[0], self.learning_rate: 0.0001,
                                        self.X1: state,
                                        self.target: target, self.actions:actions})

        if qvals.shape[0]>1:
            # Evaluating batch
            action = np.argmax(qvals, axis=1)
        else:
            # Evaluating one sample
            action = np.zeros(1)
            action[0]=np.argmax(qvals)



            # self.action_array[action[0].astype(int)]+=1
        return action.astype(int)

    def take_action(self, action, num_actions, mode):
        # Set Paramaters
        fov_v = 22.5 * np.pi / 180
        fov_h = 40 * np.pi / 180
        r = 0.4

        ignore_collision = False
        sqrt_num_actions = np.sqrt(num_actions)

        posit = self.client.simGetVehiclePose()
        pos = posit.position
        orientation = posit.orientation

        quat = (orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val)
        eulers = euler_from_quaternion(quat)
        alpha = eulers[2]

        theta_ind = int(action[0] / sqrt_num_actions)
        psi_ind = action[0] % sqrt_num_actions

        theta = fov_v/sqrt_num_actions * (theta_ind - (sqrt_num_actions - 1) / 2)
        psi = fov_h / sqrt_num_actions * (psi_ind - (sqrt_num_actions - 1) / 2)


        if mode == 'ComputerVision':
            noise_theta = (fov_v / sqrt_num_actions) / 6
            noise_psi = (fov_h / sqrt_num_actions) / 6

            psi = psi + random.uniform(-1, 1) * noise_psi
            theta = theta + random.uniform(-1, 1) * noise_theta

            x = pos.x_val + r * np.cos(alpha + psi)
            y = pos.y_val + r * np.sin(alpha + psi)
            z = pos.z_val + r * np.sin(theta)  # -ve because Unreal has -ve z direction going upwards

            self.client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0, 0, alpha + psi)),
                                     ignore_collison=ignore_collision)
        elif mode == 'Multirotor':
            r_infer = 1
            vx = r_infer * np.cos(alpha + psi)
            vy = r_infer * np.sin(alpha + psi)
            vz = r_infer * np.sin(theta)
            # TODO
            # Take average of previous velocities and current to smoothen out drone movement.
            self.client.moveByVelocityAsync(vx=vx, vy=vy, vz=vz, duration=1,
                                       drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                       yaw_mode=airsim.YawMode(is_rate=False,
                                                               yaw_or_rate=180 * (alpha + psi) / np.pi))
            # self.client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=0.01).join()
            # print("")
            # print("Throttle:", throttle)
            # print('Yaw:', yaw)

            # self.client.moveByAngleThrottleAsync(pitch=-0.015, roll=0, throttle=throttle, yaw_rate=yaw, duration=0.2).join()
            # self.client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=0.005)
    def get_depth(self):
        responses = self.client.simGetImages([airsim.ImageRequest(2, airsim.ImageType.DepthVis, False, False)])
        depth = []
        img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
        depth = img1d.reshape(responses[0].height, responses[0].width, 3)[:, :, 0]

        # To make sure the wall leaks in the unreal environment doesn't mess up with the reward function
        thresh = 50
        super_threshold_indices = depth > thresh
        depth[super_threshold_indices] = thresh
        depth = depth / thresh
        # plt.imshow(depth)
        # # plt.gray()
        # plt.show()
        return depth, thresh

    def get_state(self):
        responses1 = self.client.simGetImages([  # depth visualization image
            airsim.ImageRequest("1", airsim.ImageType.Scene, False,
                                False)])  # scene vision image in uncompressed RGBA array

        response = responses1[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)  # get numpy array
        img_rgba = img1d.reshape(response.height, response.width, 3)
        img = Image.fromarray(img_rgba)
        img_rgb = img.convert('RGB')
        self.iter = self.iter+1
        state = np.asarray(img_rgb)

        state = cv2.resize(state, (self.input_size, self.input_size), cv2.INTER_LINEAR)
        state = cv2.normalize(state, state, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        state_rgb = []
        state_rgb.append(state[:, :, 0:3])
        state_rgb = np.array(state_rgb)
        state_rgb = state_rgb.astype('float32')

        return state_rgb

    def avg_depth(self, depth_map1, thresh):
        # Version 0.3 - NAN issue resolved
        # Thresholded depth map to ignore objects too far and give them a constant value
        # Globally (not locally as in the version 0.1) Normalise the thresholded map between 0 and 1
        # Threshold depends on the environment nature (indoor/ outdoor)
        depth_map = depth_map1
        # L1=0
        # R1=0
        # C1=0
        # print(global_depth)
        # dynamic_window = False
        plot_depth = True
        global_depth = np.mean(depth_map)
        n = max(global_depth * thresh / 3, 1)
        # print("n=", n)
        # n = 3
        H = np.size(depth_map, 0)
        W = np.size(depth_map, 1)
        grid_size = (np.array([H, W]) / n)

        # scale by 0.9 to select the window towards top from the mid line
        h = max(int(0.9 * H * (n - 1) / (2 * n)), 0)
        w = max(int(W * (n - 1) / (2 * n)), 0)
        grid_location = [h, w]

        x_start = int(round(grid_location[0]))
        y_start_center = int(round(grid_location[1]))
        x_end = int(round(grid_location[0] + grid_size[0]))
        y_start_right = min(int(round(grid_location[1] + grid_size[1])), W)
        y_start_left = max(int(round(grid_location[1] - grid_size[1])), 0)
        y_end_right = min(int(round(grid_location[1] + 2 * grid_size[1])), W)

        fract_min = 0.05

        L_map = depth_map[x_start:x_end, y_start_left:y_start_center]
        C_map = depth_map[x_start:x_end, y_start_center:y_start_right]
        R_map = depth_map[x_start:x_end, y_start_right:y_end_right]

        if not L_map.any():
            L1 = 0
        else:
            L_sort = np.sort(L_map.flatten())
            end_ind = int(np.round(fract_min * len(L_sort)))
            L1 = np.mean(L_sort[0:end_ind])

        if not R_map.any():
            R1 = 0
        else:
            R_sort = np.sort(R_map.flatten())
            end_ind = int(np.round(fract_min * len(R_sort)))
            R1 = np.mean(R_sort[0:end_ind])

        if not C_map.any():
            C1 = 0
        else:
            C_sort = np.sort(C_map.flatten())
            end_ind = int(np.round(fract_min * len(C_sort)))
            C1 = np.mean(C_sort[0:end_ind])

        if plot_depth:
            cv2.rectangle(depth_map1, (y_start_center, x_start), (y_start_right, x_end), (0, 0, 0), 3)
            cv2.rectangle(depth_map1, (y_start_left, x_start), (y_start_center, x_end), (0, 0, 0), 3)
            cv2.rectangle(depth_map1, (y_start_right, x_start), (y_end_right, x_end), (0, 0, 0), 3)

            dispL = str(np.round(L1, 3))
            dispC = str(np.round(C1, 3))
            dispR = str(np.round(R1, 3))
            cv2.putText(depth_map1, dispL, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)
            cv2.putText(depth_map1, dispC, (int(W / 2 - 40), 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)
            cv2.putText(depth_map1, dispR, (int(W - 80), 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), thickness=2)
            cmap = plt.get_cmap('jet')
            #
            depth_map_heat = cmap(depth_map1)
            cv2.imshow('Depth Map', depth_map_heat)
            cv2.waitKey(1)

        # print(L1, C1, R1)
        return L1, C1, R1




    def avg_depth_old(self, depth_map1, thresh):
        # Version 0.2
        # Thresholded depth map to ignore objects too far and give them a constant value
        # Globally (not locally as in the version 0.1) Normalise the thresholded map between 0 and 1
        # Threshold depends on the environment nature (indoor/ outdoor)
        depth_map = depth_map1
        # L1=0
        # R1=0
        # C1=0
        # print(global_depth)
        # dynamic_window = False
        plot_depth = False
        global_depth = np.mean(depth_map)
        n = global_depth*thresh/3
        # print("n=", n)
        # n = 3
        H = np.size(depth_map, 0)
        W = np.size(depth_map, 1)
        grid_size = (np.array([H, W]) / n)

        # scale by 0.9 to select the window towards top from the mid line
        h = int(0.9 * H * (n - 1) / (2 * n))
        w = int(W * (n - 1) / (2 * n))
        grid_location = [h, w]

        x1 = int(round(grid_location[0]))
        y = int(round(grid_location[1]))

        a4 = int(round(grid_location[0] + grid_size[0]))

        a5 = int(round(grid_location[0] + grid_size[0]))
        b5 = int(round(grid_location[1] + grid_size[1]))

        a2 = int(round(grid_location[0] - grid_size[0]))
        b2 = int(round(grid_location[1] + grid_size[1]))

        a8 = int(round(grid_location[0] + 2 * grid_size[0]))
        b8 = int(round(grid_location[1] + grid_size[1]))

        b4 = int(round(grid_location[1] - grid_size[1]))
        if b4 < 0:
            b4 = 0

        a6 = int(round(grid_location[0] + grid_size[0]))
        b6 = int(round(grid_location[1] + 2 * grid_size[1]))
        if b6 > 640:
            b6 = 640

        # L = 1 / np.min(depth_map[x1:a4, b4:y])
        # C = 1 / np.min(depth_map[x1:a5, y:b5])
        # R = 1 / np.min(depth_map[x1:a6, b5:b6])

        fract_min = 0.05

        L_map = depth_map[x1:a4, b4:y]
        C_map = depth_map[x1:a5, y:b5]
        R_map = depth_map[x1:a6, b5:b6]

        L_sort = np.sort(L_map.flatten())
        end_ind = int(np.round(fract_min * len(L_sort)))
        L1 = np.mean(L_sort[0:end_ind])

        R_sort = np.sort(R_map.flatten())
        end_ind = int(np.round(fract_min * len(R_sort)))
        R1 = np.mean(R_sort[0:end_ind])

        C_sort = np.sort(C_map.flatten())
        end_ind = int(np.round(fract_min * len(C_sort)))
        C1 = np.mean(C_sort[0:end_ind])
        if plot_depth:
            cv2.rectangle(depth_map1, (y, x1), (b5, a5), (0, 0, 0), 3)
            cv2.rectangle(depth_map1, (y, x1), (b4, a4), (0, 0, 0), 3)
            cv2.rectangle(depth_map1, (b5, x1), (b6, a6), (0, 0, 0), 3)

            dispL = str(np.round(L1, 3))
            dispC = str(np.round(C1, 3))
            dispR = str(np.round(R1, 3))
            cv2.putText(depth_map1, dispL, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
            cv2.putText(depth_map1, dispC, (110, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
            cv2.putText(depth_map1, dispR, (200, 75), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))

            plt.imshow(depth_map1)
            plt.show()
            xxx = 1
            # time.sleep(0.7)
        #
        xxxxx = 1
        # print(L1, C1, R1)
        return L1, C1, R1



    def reward_gen(self, d_new, action, crash_threshold, thresh):
        L_new, C_new, R_new = self.avg_depth(d_new, thresh)
        # print('Rew_C', C_new)
        # print(L_new, C_new, R_new)
        # For now, lets keep the reward a simple one
        if C_new < crash_threshold:
            done = True
            reward = -1
        else:
            done = False
            if action == 0:
                reward = C_new
            else:
                # reward = C_new/3
                reward = C_new

            # if action != 0:
            #     reward = 0

        return reward, done

    def GetAgentState(self):
        return self.client.simGetCollisionInfo()

    def return_plot(self, ret, epi, env_type, mem_percent, iter, dist):
        # ret, episode, int(level/4), mem_percent, iter
        summary = tf.Summary()
        tag = 'Return'
        summary.value.add(tag=tag, simple_value=ret)
        self.stat_writer.add_summary(summary, epi)

        summary = tf.Summary()
        summary.value.add(tag='Memory-GB', simple_value=mem_percent)
        self.stat_writer.add_summary(summary, iter)

        summary = tf.Summary()
        summary.value.add(tag='Safe Flight', simple_value=dist)
        self.stat_writer.add_summary(summary, epi)

    def save_network(self, save_path):
        self.saver.save(self.sess, save_path)

    def save_weights(self, save_path):
        name = ['conv1W', 'conv1b', 'conv2W', 'conv2b', 'conv3W', 'conv3b', 'conv4W', 'conv4b', 'conv5W', 'conv5b',
                'fc6aW', 'fc6ab', 'fc7aW', 'fc7ab', 'fc8aW', 'fc8ab', 'fc9aW', 'fc9ab', 'fc10aW', 'fc10ab',
                'fc6vW', 'fc6vb', 'fc7vW', 'fc7vb', 'fc8vW', 'fc8vb', 'fc9vW', 'fc9vb', 'fc10vW', 'fc10vb'
                ]
        weights = {}
        print('Saving weights in .npy format')
        for i in range(0, 30):
            # weights[name[i]] = self.sess.run(self.sess.graph._collections['variables'][i])
            if i == 0:
                str1 = 'Variable:0'
            else:
                str1 = 'Variable_'+str(i)+':0'
            weights[name[i]] = self.sess.run(str1)
        save_path = save_path+'weights.npy'
        np.save(save_path, weights)

    def load_network(self, load_path):
        self.saver.restore(self.sess, load_path)



    def get_weights(self):
        xs=np.zeros(shape=(32, 227,227,3))
        actions = np.zeros(dtype=int, shape=[xs.shape[0]])
        ys = np.zeros(shape=[xs.shape[0]], dtype=np.float32)
        return self.sess.run(self.weights,
                             feed_dict={self.batch_size: xs.shape[0],  self.learning_rate: 0,
                                        self.X1: xs,
                                        self.target: ys, self.actions:actions})








