# Author: Aqeel Anwar(ICSRL)
# Created: 4/14/2020, 7:15 AM
# Email: aqeel.anwar@gatech.edu

import tensorflow as tf
import numpy as np
from network.loss_functions import huber_loss, mse_loss
from network.network import C3F2
from numpy import linalg as LA

class initialize_network_DeepQLearning():
    def __init__(self, cfg, name, vehicle_name):
        self.g = tf.Graph()
        self.vehicle_name = vehicle_name

        self.first_frame = True
        self.last_frame = []
        with self.g.as_default():
            stat_writer_path = cfg.network_path + self.vehicle_name + '/return_plot/'
            loss_writer_path = cfg.network_path + self.vehicle_name + '/loss' + name + '/'
            self.stat_writer = tf.summary.FileWriter(stat_writer_path)
            # name_array = 'D:/train/loss'+'/'+name
            self.loss_writer = tf.summary.FileWriter(loss_writer_path)
            self.env_type = cfg.env_type
            self.input_size = cfg.input_size
            self.num_actions = cfg.num_actions

            # Placeholders
            self.batch_size = tf.placeholder(tf.int32, shape=())
            self.learning_rate = tf.placeholder(tf.float32, shape=())
            self.X1 = tf.placeholder(tf.float32, [None, cfg.input_size, cfg.input_size, 3], name='States')

            # self.X = tf.image.resize_images(self.X1, (227, 227))

            self.X = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), self.X1)
            self.target = tf.placeholder(tf.float32, shape=[None], name='Qvals')
            self.actions = tf.placeholder(tf.int32, shape=[None], name='Actions')

            # self.model = AlexNetDuel(self.X, cfg.num_actions, cfg.train_fc)
            self.model = C3F2(self.X, cfg.num_actions, cfg.train_fc)

            self.predict = self.model.output
            ind = tf.one_hot(self.actions, cfg.num_actions)
            pred_Q = tf.reduce_sum(tf.multiply(self.model.output, ind), axis=1)
            self.loss = huber_loss(pred_Q, self.target)
            self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.99).minimize(
                self.loss, name="train")

            self.sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            self.saver = tf.train.Saver()
            self.all_vars = tf.trainable_variables()

            self.sess.graph.finalize()

        # Load custom weights from custom_load_path if required
        if cfg.custom_load:
            print('Loading weights from: ', cfg.custom_load_path)
            self.load_network(cfg.custom_load_path)


    def get_vars(self):
        return self.sess.run(self.all_vars)


    def initialize_graphs_with_average(self, agent, agent_on_same_network):
        values = {}
        var = {}
        all_assign = {}
        for name_agent in agent_on_same_network:
            values[name_agent] = agent[name_agent].network_model.get_vars()
            var[name_agent] = agent[name_agent].network_model.all_vars
            all_assign[name_agent] = []

        for i in range(len(values[name_agent])):
            val = []
            for name_agent in agent_on_same_network:
                val.append(values[name_agent][i])
            # Take mean here
            mean_val = np.average(val, axis=0)
            for name_agent in agent_on_same_network:
                # all_assign[name_agent].append(tf.assign(var[name_agent][i], mean_val))
                var[name_agent][i].load(mean_val, agent[name_agent].network_model.sess)


    def Q_val(self, xs):
        target = np.zeros(shape=[xs.shape[0]], dtype=np.float32)
        actions = np.zeros(dtype=int, shape=[xs.shape[0]])
        return self.sess.run(self.predict,
                             feed_dict={self.batch_size: xs.shape[0], self.learning_rate: 0, self.X1: xs,
                                        self.target: target, self.actions: actions})


    def train_n(self, xs, ys, actions, batch_size, dropout_rate, lr, epsilon, iter):
        _, loss, Q = self.sess.run([self.train, self.loss, self.predict],
                                   feed_dict={self.batch_size: batch_size, self.learning_rate: lr, self.X1: xs,
                                              self.target: ys, self.actions: actions})
        meanQ = np.mean(Q)
        maxQ = np.max(Q)
        # Log to tensorboard
        self.log_to_tensorboard(tag='Loss', group=self.vehicle_name, value=LA.norm(loss) / batch_size, index=iter)
        self.log_to_tensorboard(tag='Epsilon', group=self.vehicle_name, value=epsilon, index=iter)
        self.log_to_tensorboard(tag='Learning Rate', group=self.vehicle_name, value=lr, index=iter)
        self.log_to_tensorboard(tag='MeanQ', group=self.vehicle_name, value=meanQ, index=iter)
        self.log_to_tensorboard(tag='MaxQ', group=self.vehicle_name, value=maxQ, index=iter)


    def action_selection(self, state):
        target = np.zeros(shape=[state.shape[0]], dtype=np.float32)
        actions = np.zeros(dtype=int, shape=[state.shape[0]])
        qvals = self.sess.run(self.predict,
                              feed_dict={self.batch_size: state.shape[0], self.learning_rate: 0.0001,
                                         self.X1: state,
                                         self.target: target, self.actions: actions})

        if qvals.shape[0] > 1:
            # Evaluating batch
            action = np.argmax(qvals, axis=1)
        else:
            # Evaluating one sample
            action = np.zeros(1)
            action[0] = np.argmax(qvals)

        return action.astype(int)


    def log_to_tensorboard(self, tag, group, value, index):
        summary = tf.Summary()
        tag = group + '/' + tag
        summary.value.add(tag=tag, simple_value=value)
        self.stat_writer.add_summary(summary, index)


    def save_network(self, save_path, episode=''):
        save_path = save_path + self.vehicle_name + '/' + self.vehicle_name + '_' + str(episode)
        self.saver.save(self.sess, save_path)
        print('Model Saved: ', save_path)


    def load_network(self, load_path):
        self.saver.restore(self.sess, load_path)


    def get_weights(self):
        xs = np.zeros(shape=(32, 227, 227, 3))
        actions = np.zeros(dtype=int, shape=[xs.shape[0]])
        ys = np.zeros(shape=[xs.shape[0]], dtype=np.float32)
        return self.sess.run(self.weights,
                             feed_dict={self.batch_size: xs.shape[0], self.learning_rate: 0,
                                        self.X1: xs,
                                        self.target: ys, self.actions: actions})








