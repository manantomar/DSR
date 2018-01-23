#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import argparse
import random
import time
import sys
import os

import numpy as np
import cv2
import tensorflow as tf
import itertools

def MakeDir(path):
    try:
        os.makedirs(path)
    except:
        pass

lab = True
load_model = True
train = False
test_display = True
test_write_video = False
path_work_dir = "/home/manan/Downloads/lab/python/"
vizdoom_path = "/usr/local/lib/python2.7/dist-packages/vizdoom/"
vizdoom_scenario = vizdoom_path + "scenarios/simpler_basic.wad"
logs_path = "/home/manan/Downloads/lab/python/model_lab_dsr_ae_small/aws/"

# Lab parameters.
if (lab):
    from env_lab import EnvLab

    learning_rate = 0.00020  # 0.001
    discount_factor = 0.99
    step_num = int(5e5)  # int(1e6)
    replay_memory_size = int(1e6)
    replay_memory_batch_size = 64
    num_layers = 32

    # Exploration rate.
    start_eps = 1.0
    end_eps = 0.1
    eps_decay_iter = 0.33 * step_num

    frame_repeat = 10  # 4
    channels = 3
    resolution = (40, 40) + (channels,)  # Original: 240x320

    model_path = path_work_dir + "model_lab_dsr_ae_small/aws/"
    save_each = 0.001 * step_num
    step_load = 1000

# Vizdoom parameters.
if (not lab):
    from env_vizdoom import EnvVizDoom

    learning_rate = 0.00015
    discount_factor = 0.99
    step_num = int(1e5)
    replay_memory_size = int(1e5)
    replay_memory_batch_size = 64
    num_layers = 256

    frame_repeat = 10
    channels = 3
    resolution = (40, 40) + (channels,) # Original: 480x640

    start_eps = 1.0
    end_eps = 0.1
    eps_decay_iter = 0.33 * step_num

    model_path = path_work_dir + "model_vizdoom_dsr_ae_small/iter_11/"
    save_each = 0.01 * step_num
    step_load = 100

MakeDir(model_path)
model_name = model_path #+ "dqn"

# Global variables.
env = None

def PrintStat(elapsed_time, step, step_num, train_scores):
    steps_per_s = 1.0 * step / elapsed_time
    steps_per_m = 60.0 * step / elapsed_time
    steps_per_h = 3600.0 * step / elapsed_time
    steps_remain = step_num - step
    remain_h = int(steps_remain / steps_per_h)
    remain_m = int((steps_remain - remain_h * steps_per_h) / steps_per_m)
    remain_s = int((steps_remain - remain_h * steps_per_h - remain_m * steps_per_m) / steps_per_s)
    elapsed_h = int(elapsed_time / 3600)
    elapsed_m = int((elapsed_time - elapsed_h * 3600) / 60)
    elapsed_s = int((elapsed_time - elapsed_h * 3600 - elapsed_m * 60))
    print("{}% | Steps: {}/{}, {:.2f}M step/h, {:02}:{:02}:{:02}/{:02}:{:02}:{:02}".format(
        100.0 * step / step_num, step, step_num, steps_per_h / 1e6,
        elapsed_h, elapsed_m, elapsed_s, remain_h, remain_m, remain_s), file=sys.stderr)

    mean_train = 0
    std_train = 0
    min_train = 0
    max_train = 0
    if (len(train_scores) > 0):
        train_scores = np.array(train_scores)
        mean_train = train_scores.mean()
        std_train = train_scores.std()
        min_train = train_scores.min()
        max_train = train_scores.max()
    print("Episodes: {} Rewards: mean: {:.2f}, std: {:.2f}, min: {:.2f}, max: {:.2f}".format(
        len(train_scores), mean_train, std_train, min_train, max_train), file=sys.stderr)

def Preprocess(img):
    #cv2.imshow("frame-train", img)
    #cv2.waitKey(20)
    if (channels == 1):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (resolution[1], resolution[0]))
    #cv2.imshow("frame-train", img)
    #cv2.waitKey(200)
    return np.reshape(img, resolution)

class ReplayMemory(object):
    def __init__(self, capacity):

        self.s = np.zeros((capacity,) + resolution, dtype=np.uint8)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def Add(self, s, action, isterminal, reward):

        self.s[self.pos, ...] = s
        self.a[self.pos] = action
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def Get(self, sample_size):

        idx = random.sample(xrange(0, self.size-2), sample_size)
        idx2 = []
        for i in idx:
            idx2.append(i + 1)
        return self.s[idx], self.a[idx], self.s[idx2], self.isterminal[idx], self.r[idx]

class Model(object):
    def __init__(self, session, actions_count, scope):

        self.session = session
        self.scope = scope

        # Create the input.
        self.s_ = tf.placeholder(shape=[None] + list(resolution), dtype=tf.float32)
        self.mu_ = tf.placeholder(shape=[None, num_layers], dtype=tf.float32)
        self.r_ = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.a_ = tf.placeholder(shape=[None, 2], dtype=tf.int32)
        self.mode = tf.placeholder(shape=None, dtype=tf.int32, name="mode")

        # Create the network.
        conv1 = tf.contrib.layers.conv2d(self.s_, num_outputs=8, kernel_size=[3, 3], stride=[2, 2], padding='SAME')
        conv2 = tf.contrib.layers.conv2d(conv1, num_outputs=16, kernel_size=[3, 3], stride=[2, 2])#, padding='VALID')
        conv2_flat = tf.contrib.layers.flatten(conv2)
        self.fc1 = tf.contrib.layers.fully_connected(conv2_flat, num_outputs=num_layers)

        fc2 = tf.contrib.layers.fully_connected(self.fc1, num_outputs=1600)

        '''deconv0 = tf.reshape(self.fc1, shape=[-1, 1, 1, 128])
        deconv1 = tf.layers.conv2d_transpose(deconv0, filters=16, kernel_size=[4, 4], strides=[1,1])#, padding='VALID')
        deconv2 = tf.layers.conv2d_transpose(deconv1, filters=8, kernel_size=[4, 4], strides=[2,2])#, padding='VALID')
        deconv3 = tf.layers.conv2d_transpose(deconv2, filters=4, kernel_size=[4, 4], strides=[2,2], padding='SAME')
        self.reconstruct = tf.layers.conv2d_transpose(deconv3, filters=3, kernel_size=[4, 4], strides=[2,2], activation=tf.nn.tanh, padding='SAME')'''

        deconv0 = tf.reshape(fc2, shape=[-1, 10, 10, 16])
        deconv1 = tf.contrib.layers.conv2d_transpose(deconv0, num_outputs=8, kernel_size=[3, 3], stride=[2,2])#, padding='VALID')
        self.reconstruct = tf.contrib.layers.conv2d_transpose(deconv1, num_outputs=3, kernel_size=[3, 3], stride=[2,2], activation_fn=tf.nn.tanh)#, padding='SAME')
        #self.reconstruct = tf.contrib.layers.conv2d_transpose(deconv2, num_outputs=3, kernel_size=[3, 3], stride=[2,2], activation_fn=tf.nn.tanh)#, padding='SAME')


        print("conv1", conv1)
        print("conv2", conv2)

        print("fc1", self.fc1)
        print("deconv0", deconv0)
        print("deconv1", deconv1)
        #print("deconv2", deconv2)
        #print("deconv3", deconv3)
        print("deconv3", self.reconstruct)

        self.fc1_stopgrad = tf.stop_gradient(self.fc1)

        with tf.variable_scope("regression") as s:
            self.r = tf.contrib.layers.fully_connected(self.fc1, num_outputs=1, activation_fn=None, scope=s)
            #self.q = tf.contrib.layers.fully_connected(mu, num_outputs)
        def add_sr_layer():
            x = tf.contrib.layers.fully_connected(self.fc1_stopgrad, num_outputs=num_layers)
            x = tf.contrib.layers.fully_connected(x, num_outputs=num_layers)
            #x = tf.contrib.layers.fully_connected(x, num_outputs=512)
            return x

        mu_list = []
        self.q = []

        for i in range(actions_count):
                mu_list.append(add_sr_layer())

                with tf.variable_scope("regression") as s:
                    self.q.append(tf.contrib.layers.fully_connected(mu_list[i], num_outputs=1, activation_fn=None, reuse=True, scope=s))

        self.mu = tf.stack(mu_list)
        self.mu = tf.transpose(self.mu, [1,0,2])
        #self.q = tf.contrib.layers.fully_connected(fc1, num_outputs=actions_count, activation_fn=None)
        #self.action = tf.reduce_sum(self.q)
        self.getmu = tf.gather_nd(self.mu, self.a_)
        self.action = tf.argmax(self.q)

        #self.loss = tf.losses.mean_squared_error(self.mu_, self.mu[self.action])
        self.loss_sr = tf.losses.mean_squared_error(self.mu_, tf.gather_nd(self.mu, self.a_))
        self.loss_r = tf.losses.mean_squared_error(self.r_, self.r)
        self.loss_g = tf.losses.mean_squared_error(self.reconstruct, self.s_)

        #self.loss_sr = tf.reduce_mean(tf.squared_difference(self.mu_, tf.gather_nd(self.mu, self.a_)))
        #self.loss_r = tf.reduce_mean(tf.squared_difference(self.r_, self.r))

        # 0 : SR | 1 : reward

        #self.loss = tf.cond(self.mode > 0, lambda: self.loss_r + self.loss_sr, lambda: self.loss_sr)
        self.loss = self.loss_sr + self.loss_r #+ self.loss_g
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate)
        self.train_step = self.optimizer.minimize(self.loss)

        self.reward_per_episode = tf.placeholder(tf.float32)
        tf.summary.scalar("Reward", self.reward_per_episode)
        self.summary_op = tf.summary.merge_all()

    def Learn(self, mu_target, state, action, reward, mode):

        state = state.astype(np.float32)
        reward = reward.reshape((replay_memory_batch_size,1))
        l, _ = self.session.run([self.loss, self.train_step], feed_dict={self.mu_: mu_target, self.s_ : state, self.a_: action, self.r_ : reward, self.mode : mode})
        return l

    def GetMU(self, state, action):

        state = state.astype(np.float32)
        return self.session.run(self.getmu, feed_dict={self.s_: state, self.a_ : action})

    def GetPhi(self, state):
        state = state.astype(np.float32)
        return self.session.run(self.fc1, feed_dict={self.s_: state})

    def GetReconstruction(self, state):
        state = state.astype(np.float32)
        return self.session.run(self.reconstruct, feed_dict={self.s_: state})

    def GetReward(self, state):
        state = state.astype(np.float32)
        return self.session.run(self.r, feed_dict={self.s_: state})

    def GetAction(self, state):

        state = state.astype(np.float32)
        #state = state.reshape([replay_memory_batch_size] + list(resolution))
        return self.session.run(self.action, feed_dict={self.s_: state})[0][0]

class Agent(object):

    def __init__(self, num_actions):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = False
        config.allow_soft_placement = True

        self.session = tf.Session(config=config)

        with tf.variable_scope("model"):
            self.model = Model(self.session, num_actions, "model")
        with tf.variable_scope("target"):
            self.target = Model(self.session, num_actions, "target")
        self.nonzeroMemory = ReplayMemory(int(5e4))
        self.memory = ReplayMemory(replay_memory_size)

        self.rewards = 0

        self.writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        self.saver = tf.train.Saver(max_to_keep=3)
        if (load_model):
            model_name_curr = model_name + "_{:04}".format(step_load)
            print("Loading model from: ", model_name_curr)
            self.saver.restore(self.session, model_name_curr)
        else:
            init = tf.global_variables_initializer()
            self.session.run(init)

        self.num_actions = num_actions
        self.num_steps = 1

    def copy_model_parameters(self, estimator1, estimator2):
        """
        Copies the model parameters of one estimator to another.
        Args:
        sess: Tensorflow session instance
        estimator1: Estimator to copy the paramters from
        estimator2: Estimator to copy the parameters to
        """
        e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            update_ops.append(op)

        self.session.run(update_ops)

    def LearnFromMemory(self, mode):

        if (self.memory.size > 2*replay_memory_batch_size and self.nonzeroMemory.size > 2*replay_memory_batch_size):
            #print("sampling")
            if (random.random() <= 0.3):
                s1, a, s2, isterminal, r = self.nonzeroMemory.Get(replay_memory_batch_size)
            else:
                s1, a, s2, isterminal, r = self.memory.Get(replay_memory_batch_size)

            #q = self.model.GetQ(s1)
            #q2 = np.max(self.model.GetQ(s2), axis=1)
            #q[np.arange(q.shape[0]), a] = r + (1 - isterminal) * discount_factor * q2
            #self.model.Learn(s1, q)
            #print("action chosen", self.model.GetAction(s2).shape)
            #print("replay size", a)
            #mu[np.arange(mu.shape[0]), 512] = self.model.GetPhi(s1) + (1 - isterminal) * discount_factor * mu2
            #print("s2", s2.shape)
            a = np.asarray(zip(np.arange(replay_memory_batch_size), a))
            mu = self.model.GetMU(s1, a)
            mu2 = self.model.GetMU(s2, np.asarray(zip(np.arange(replay_memory_batch_size), itertools.repeat(self.model.GetAction(s2)))))
            #mu2 = self.target.GetMU(s2, np.asarray(zip(np.arange(replay_memory_batch_size), itertools.repeat(self.target.GetAction(s2)))))
            #mu2 = self.model.GetMU(s2, np.asarray(zip(np.arange(replay_memory_batch_size), self.model.GetAction(s2))))
            done = (1 - isterminal).reshape((replay_memory_batch_size, 1)) + np.zeros((replay_memory_batch_size, num_layers))

            self.model.Learn(self.model.GetPhi(s1) + done * discount_factor * mu2, s1, a, r, mode)

    def GetAction(self, state):

        if (random.random() <= 0.05):
            a = random.randint(0, self.num_actions-1)
        else:
            a = self.model.GetAction(np.reshape(state, (1, 40, 40, 3)))

        return a

    def Step(self, iteration):

        s = Preprocess(env.Observation())

        # Epsilon-greedy.
        if (iteration < eps_decay_iter):
            eps = start_eps - iteration / eps_decay_iter * (start_eps - end_eps)
        else:
            eps = end_eps

        if (random.random() <= eps):
            a = random.randint(0, self.num_actions-1)
        else:
            a = self.model.GetAction(np.reshape(s, (1, 40, 40, 3)))

        reward = env.Act(a, frame_repeat)
        self.rewards += reward

        isterminal = not env.IsRunning()

        if (reward != 0):
          self.nonzeroMemory.Add(s, a, isterminal, reward)
        else :
            self.memory.Add(s, a, isterminal, reward)

#        if (iteration > 20000):
#            self.num_steps = self.num_steps or 50000
#            self.num_steps = int(self.num_steps / 2)

#            for i in range(self.num_steps):
#            self.LearnFromMemory(1)

#        else :
#            if (iteration % 2 == 0):
#            for i in range(5000):
#            self.LearnFromMemory(0)

#        else:
#            for i in range(5000):'''
        self.LearnFromMemory(1)

        if (iteration % 5000 == 0):
            print("updating target network now...")
            self.copy_model_parameters(self.model, self.target)

        print(self.model.GetReward(np.reshape(s, (1, 40, 40, 3)))[0])
        print("reward", reward)
        cv2.imshow("reconstruction", self.model.GetReconstruction(np.reshape(s, (1, 40, 40, 3)))[0])
        cv2.imshow("input", np.reshape(s, (1, 40, 40, 3))[0])
        cv2.waitKey(5)


    def Train(self):

        print("Starting training.")
        start_time = time.time()
        train_scores = []
        env.Reset()
        for step in xrange(1, step_num+1):
            self.Step(step)
            if (not env.IsRunning()):
                summary = self.session.run(self.model.summary_op, feed_dict={self.model.reward_per_episode: self.rewards})
                self.writer.add_summary(summary, step)
                train_scores.append(self.rewards)
                self.rewards = 0
                env.Reset()

            if (step % save_each == 0):
                model_name_curr = model_name + "_{:04}".format(int(step / save_each))
                print("\nSaving the network weigths to:", model_name_curr, file=sys.stderr)
                self.saver.save(self.session, model_name_curr)

                PrintStat(time.time() - start_time, step, step_num, train_scores)

                train_scores = []

        env.Reset()

def Test(agent):
    if (test_write_video):
        size = (640, 480)
        fps = 30.0 #/ frame_repeat
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # cv2.cv.CV_FOURCC(*'XVID')
        out_video = cv2.VideoWriter(path_work_dir + "test.avi", fourcc, fps, size)

    reward_total = 0
    num_episodes = 30
    while (num_episodes != 0):
        if (not env.IsRunning()):
            env.Reset()
            print("Total reward: {}".format(reward_total))
            reward_total = 0
            num_episodes -= 1

        state_raw = env.Observation()

        state = Preprocess(state_raw)
        action = agent.GetAction(state)
        #print("reward", agent.model.GetReward(state))

        for _ in xrange(frame_repeat):
            # Display.
            if (test_display):
                cv2.imshow("frame-test", state_raw)
                cv2.waitKey(20)

            if (test_write_video):
                out_video.write(state_raw)

            reward = env.Act(action, 1)
            reward_total += reward

            if (not env.IsRunning()):
                break

            state_raw = env.Observation()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="the GPU to use")
    args = parser.parse_args()

    if (args.gpu):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if (lab):
        env = EnvLab(80, 80, 60, "seekavoid_arena_01") #seekavoid_arena_01
    else:
        env = EnvVizDoom(vizdoom_scenario)

    agent = Agent(env.NumActions())

    if (train):
        agent.Train()

    Test(agent)
